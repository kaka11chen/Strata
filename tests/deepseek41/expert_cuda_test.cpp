#include "strata/deepseek41/expert_cpu.hpp"
#include "strata/deepseek41/expert_cuda.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>

using namespace strata;
using namespace strata::deepseek41;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) {
    ++checks;
    failures += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
float bf16(float x) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
}

// The pinned GPU quantizer uses fast division. At half-integer boundaries it can
// choose the neighboring code to a scalar CPU division (observed at +/-63.5).
// Check scales exactly and admit a one-code difference ONLY within 2e-5 of a tie.
// Use those checked GPU activation bytes for the independent FP64 matrix oracle.
struct ReferenceQuantizer {
    core::DeviceArena arena{5120 * 4 + 5120 / 32 * 36 + 256};
    Stream stream;
    float* values = static_cast<float*>(arena.alloc(5120 * 4));
    void* quantized = arena.alloc(5120 / 32 * 36);
    bool valid = true;
    size_t blocks = 0, boundary_codes = 0;

    std::vector<float> operator()(std::vector<float> x) {
        std::vector<uint8_t> packed(x.size() / 32 * 36);
        ck(cudaMemcpyAsync(values, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
        kernels::native_quantize_q8_1(values, quantized, int(x.size()), 1, stream.value);
        ck(cudaMemcpyAsync(packed.data(), quantized, packed.size(), cudaMemcpyDeviceToHost, stream.value));
        ck(cudaStreamSynchronize(stream.value));
        for (size_t b = 0; b < x.size() / 32; ++b) {
            ++blocks;
            float amax = 0.f;
            for (size_t j = 0; j < 32; ++j)
                amax = std::max(amax, std::abs(x[32 * b + j]));
            const float d = amax / 127.f;
            uint16_t stored;
            std::memcpy(&stored, packed.data() + b * 36, sizeof(stored));
            valid = valid && stored == ggml_fp32_to_fp16(d);
            for (size_t j = 0; j < 32; ++j) {
                const float ratio = d == 0.f ? 0.f : x[32 * b + j] / d;
                const int expected = int(std::round(ratio));
                const int actual = int(int8_t(packed[b * 36 + 4 + j]));
                const float fraction = std::abs(ratio) - std::floor(std::abs(ratio));
                valid =
                    valid && actual >= -127 && actual <= 127 &&
                    (actual == expected || (std::abs(actual - expected) == 1 && std::abs(fraction - 0.5f) <= 2e-5f));
                boundary_codes += actual != expected;
                x[32 * b + j] = ggml_fp16_to_fp32(stored) * actual;
            }
        }
        return x;
    }
};

// Original MXFP4 decoder + FP64 dots: no Strata projection or weight conversion.
// NVFP4 dots do not use Q8_1's sum field. Quantization is checked separately above.
std::vector<float> projection(const ExpertPart& weight, int width, int rows, const std::vector<float>& input,
                              ReferenceQuantizer& quantize) {
    const auto x = quantize(input);
    const size_t row_bytes = size_t(width / 32) * 17;
    std::vector<float> w(size_t(width), 0.f), y(size_t(rows), 0.f);
    for (int r = 0; r < rows; ++r) {
        ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(weight.data.data() + size_t(r) * row_bytes, w.data(), width);
        double sum = 0;
        for (int j = 0; j < width; ++j)
            sum += double(w[size_t(j)]) * x[size_t(j)];
        y[size_t(r)] = bf16(float(sum));
    }
    return y;
}
std::vector<float> reference(const Geometry& g, const ExpertView& weights, std::vector<float> x, float route,
                             ReferenceQuantizer& quantize) {
    for (float& v : x)
        v = bf16(v);
    const auto gate = projection(weights.parts[0], g.hidden, g.expert_width, x, quantize);
    const auto up = projection(weights.parts[1], g.hidden, g.expert_width, x, quantize);
    std::vector<float> hidden(size_t(g.expert_width), 0.f);
    for (size_t i = 0; i < hidden.size(); ++i) {
        const float a = std::min(gate[i], 10.f), b = std::clamp(up[i], -10.f, 10.f);
        hidden[i] = bf16(((a / (1.f + std::exp(-a))) * b) * route);
    }
    return projection(weights.parts[2], g.expert_width, g.hidden, hidden, quantize);
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name,
             double l2_limit, double peak_limit) {
    double error2 = 0, norm2 = 0, max_abs = 0, peak = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i];
        error2 += delta * delta;
        norm2 += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta));
        peak = std::max(peak, std::abs(double(expected[i])));
    }
    const double l2 = std::sqrt(error2 / std::max(norm2, 1e-30));
    std::printf("  %s: relative_L2=%.8g max_abs=%.8g peak=%.8g\n", name.c_str(), l2, max_abs, peak);
    check(std::isfinite(l2) && l2 <= l2_limit && max_abs <= 1e-5 + peak_limit * peak, name);
}
void rejects(const char* name, const std::function<void()>& run) {
    try {
        run();
        check(false, name);
    } catch (const std::invalid_argument&) {
        check(true, name);
    }
}
void verify_weight_values(const Geometry& g, const ExpertView& weights) {
    std::vector<uint8_t> packed(nvfp4_expert_bytes(g));pack_nvfp4_expert(g,weights,packed);
    const auto scales=nvfp4_expert_scales(packed);const size_t part=(packed.size()-nvfp4_scale_tail_bytes)/3;
    for (size_t role = 0; role < 3; ++role) {
        const int width = role == 2 ? g.expert_width : g.hidden, rows = role == 2 ? g.hidden : g.expert_width;
        std::vector<float> a(size_t(width), 0.f), b(size_t(width), 0.f);
        bool equal = true;
        for (int r = 0; r < rows; ++r) {
            ggml_get_type_traits(GGML_TYPE_MXFP4)
                ->to_float(weights.parts[role].data.data() + size_t(r) * size_t(width / 32) * 17, a.data(), width);
            ggml_get_type_traits(GGML_TYPE_NVFP4)
                ->to_float(packed.data() + role * part + size_t(r) * size_t(width / 64) * 36, b.data(), width);
            for(float& value:b)value*=scales[role];
            equal = equal && a == b;
        }
        check(equal, "all weight values unchanged for role " + std::to_string(role));
    }
}
void run_cases(const Geometry& g, const ExpertView& weights, const char* label) {
    CpuExpert cpu(g, weights);
    verify_weight_values(g, weights);
    CudaExpert gpu(g, weights);
    ReferenceQuantizer quantize;
    Stream stream;
    const size_t bytes = size_t(g.hidden) * sizeof(float);
    core::DeviceArena io(bytes * 2 + 256);
    auto* dx = static_cast<float*>(io.alloc(bytes));
    auto* dy = static_cast<float*>(io.alloc(bytes));
    const size_t device_bytes = gpu.device_bytes();
    std::vector<float> input(size_t(g.hidden), 0.f), actual(input), cpu_output(input);
    for (int i = 0; i < g.hidden; ++i)
        input[size_t(i)] = std::sin(float(i) * 0.17f) + 0.3f * std::cos(float(i) * 0.071f);
    auto run = [&](const std::vector<float>& x, float route) {
        ck(cudaMemcpyAsync(dx, x.data(), bytes, cudaMemcpyHostToDevice, stream.value));
        gpu.forward(dx, route, dy, stream.value);
        ck(cudaMemcpyAsync(actual.data(), dy, bytes, cudaMemcpyDeviceToHost, stream.value));
        ck(cudaStreamSynchronize(stream.value));
    };
    // Limits fixed before real GPU execution: 0.3% against the Q8_1 oracle, 3% across
    // Q8_0/Q8_1 implementations (which can round exact half-integers differently).
    for (int c = 0; c < 3; ++c) {
        const float route = c == 0 ? 1.f : c == 1 ? 0.37f : 1.5f;
        if (c == 2)
            for (float& v : input)
                v *= 16.f;
        run(input, route);
        compare(actual, reference(g, weights, input, route, quantize),
                std::string(label) + " Q8_1 oracle case " + std::to_string(c), 0.003, 0.02);
        cpu.forward(input, route, cpu_output);
        compare(actual, cpu_output, std::string(label) + " CPU Q8_0 case " + std::to_string(c), 0.03, 0.08);
    }
    check(quantize.valid,
          std::string(label) + " Q8_1 scales exact; code differences limited to half-integer boundaries");
    std::printf("  %s: checked %zu Q8_1 blocks; %zu boundary code differences vs scalar division\n", label,
                quantize.blocks, quantize.boundary_codes);
    run(input, 0.f);
    check(std::all_of(actual.begin(), actual.end(), [](float v) { return v == 0.f; }),
          std::string(label) + " zero route");
    std::fill(input.begin(), input.end(), 0.f);
    run(input, 1.f);
    check(std::all_of(actual.begin(), actual.end(), [](float v) { return v == 0.f; }),
          std::string(label) + " fresh input at reused device address");
    check(gpu.device_bytes() == device_bytes, std::string(label) + " fixed arena capacity");
    rejects("reject null stream", [&] { gpu.forward(dx, 1.f, dy, nullptr); });
    rejects("reject null input", [&] { gpu.forward(nullptr, 1.f, dy, stream.value); });
    rejects("reject negative route", [&] { gpu.forward(dx, -1.f, dy, stream.value); });
    rejects("reject route above 1017 bound", [&] { gpu.forward(dx, 2.f, dy, stream.value); });
    rejects("reject NaN route", [&] { gpu.forward(dx, std::numeric_limits<float>::quiet_NaN(), dy, stream.value); });
    std::printf("  %s: MXFP4=%zu NVFP4=%zu scratch=%zu arena=%zu bytes\n", label, weights.bytes(), gpu.weight_bytes(),
                gpu.scratch_bytes(), gpu.device_bytes());
}
void synthetic_tests() {
    Geometry g;
    g.layers = g.experts = g.top_k = 1;
    g.hidden = 128;
    g.expert_width = 64;
    std::array<std::vector<uint8_t>, 3> storage;
    ExpertView weights;
    std::mt19937 rng(717);
    for (size_t role = 0; role < 3; ++role) {
        auto& bytes = storage[role];
        bytes.resize(size_t(g.hidden) * size_t(g.expert_width) / 32 * 17);
        for (size_t b = 0; b < bytes.size(); b += 17) {
            bytes[b] = 119 + uint8_t(rng() % 4);
            for (size_t j = 1; j < 17; ++j)
                bytes[b + j] = uint8_t(rng());
        }
        weights.parts[role].type = 39;
        weights.parts[role].data = bytes;
    }
    run_cases(g, weights, "synthetic");
    for(size_t role=0;role<3;++role){
        const uint8_t old_scale=storage[role][0],old_codes=storage[role][1];
        storage[role][0]=117;storage[role][1]=0x31;
        const auto label="synthetic-global-half-role"+std::to_string(role);
        run_cases(g,weights,label.c_str());
        storage[role][0]=old_scale;storage[role][1]=old_codes;
    }
    // Reproduce the discovered boundary case without needing the real checkpoint.
    ReferenceQuantizer quantize;
    std::vector<float> ties(128, 0.f);
    const float maxima[] = {1.2578125f, 1.15625f, 0.83203125f, 1.f};
    for (size_t b = 0; b < 4; ++b) {
        ties[32 * b] = maxima[b];
        ties[32 * b + 1] = maxima[b] * 0.5f;
        ties[32 * b + 2] = -maxima[b] * 0.5f;
        ties[32 * b + 31] = -maxima[b];
    }
    (void)quantize(ties);
    check(quantize.valid, "crafted positive/negative Q8_1 half-integer boundaries");
    const auto zeros = quantize(std::vector<float>(128, 0.f));
    check(quantize.valid && std::all_of(zeros.begin(), zeros.end(), [](float v) { return v == 0.f; }),
          "all-zero Q8_1 blocks");
    auto broken = weights;
    broken.parts[0].type = 42;
    rejects("reject unsupported GPU weight type", [&] { CudaExpert bad(g, broken); });
    broken = weights;
    broken.parts[0].data = broken.parts[0].data.first(17);
    rejects("reject short GPU weight matrix", [&] { CudaExpert bad(g, broken); });
    storage[0][0] = 116;
    rejects("reject unsupported NVFP4 scale conversion", [&] { CudaExpert bad(g, weights); });
}
void real_test(const char* path) {
    const auto model = GgufModel::open(path);
    const auto* arch = model.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-expert-fixture")
        throw std::runtime_error("expected a prepared single-expert fixture");
    Geometry g;
    g.layers = g.experts = g.top_k = 1;
    g.hidden = 5120;
    g.expert_width = 2304;
    ExpertIndex index(model, g);
    const auto* prefix = model.meta().get("strata.deepseek41.source_prefix");
    std::printf("Real expert: %s; original MXFP4 values, GPU NVFP4/Q8_1, CPU MXFP4/Q8_0\n",
                prefix ? prefix->s.c_str() : "unknown");
    run_cases(g, index.view(0, 0), "real-1017");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 1 && !(argc == 3 && std::string(argv[1]) == "--real-expert"))
            throw std::runtime_error("usage: deepseek41_expert_cuda_test [--real-expert FIXTURE.gguf]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::fprintf(stderr, "No CUDA device; GPU test not run\n");
            return 77;
        }
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0));
        std::printf("GPU: %s, SM %d%d\n", prop.name, prop.major, prop.minor);
        synthetic_tests();
        if (argc == 3)
            real_test(argv[2]);
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
