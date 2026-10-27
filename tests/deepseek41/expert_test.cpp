#include "strata/deepseek41/expert_cpu.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
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
void rejects(const char* name, const std::function<void()>& run) {
    try {
        run();
        check(false, name);
    } catch (const std::exception&) {
        check(true, name);
    }
}
float bf16(float x) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(x));
}

// Oracle uses ggml's format decoders but NONE of Strata's projection functions.
// Sum dequantized products in FP64. Q8_0 rounding is part of the declared baseline.
std::vector<float> projection_reference(const ExpertPart& weights, int width, int rows, const std::vector<float>& x) {
    std::vector<uint8_t> q(size_t(ggml_row_size(GGML_TYPE_Q8_0, width)));
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), q.data(), width);
    std::vector<float> activation(size_t(width), 0.f), weight(size_t(width), 0.f), out(size_t(rows), 0.f);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(q.data(), activation.data(), width);
    const size_t row_bytes = ggml_row_size(GGML_TYPE_MXFP4, width);
    for (int r = 0; r < rows; ++r) {
        ggml_get_type_traits(GGML_TYPE_MXFP4)
            ->to_float(weights.data.data() + size_t(r) * row_bytes, weight.data(), width);
        double sum = 0;
        for (int k = 0; k < width; ++k)
            sum += double(weight[size_t(k)]) * double(activation[size_t(k)]);
        out[size_t(r)] = bf16(float(sum));
    }
    return out;
}
struct ClampCounts {
    int positive_gate = 0, negative_gate = 0, positive_up = 0, negative_up = 0;
};
std::vector<float> reference(const Geometry& g, const ExpertView& weights, const std::vector<float>& input, float route,
                             ClampCounts& counts) {
    auto x = input;
    for (float& v : x)
        v = bf16(v);
    const auto gate = projection_reference(weights.parts[0], g.hidden, g.expert_width, x);
    const auto up = projection_reference(weights.parts[1], g.hidden, g.expert_width, x);
    std::vector<float> hidden(size_t(g.expert_width), 0.f);
    for (int i = 0; i < g.expert_width; ++i) {
        counts.positive_gate += gate[size_t(i)] > 10.f;
        counts.negative_gate += gate[size_t(i)] < -10.f;
        counts.positive_up += up[size_t(i)] > 10.f;
        counts.negative_up += up[size_t(i)] < -10.f;
        const float a = std::min(gate[size_t(i)], 10.f);
        const float b = std::clamp(up[size_t(i)], -10.f, 10.f);
        hidden[size_t(i)] = bf16(((a / (1.f + std::exp(-a))) * b) * route);
    }
    return projection_reference(weights.parts[2], g.expert_width, g.hidden, hidden);
}
void compare(const Geometry& g, const ExpertView& weights, const std::vector<float>& input, float route,
             CpuExpert& expert, const std::string& name, ClampCounts& counts) {
    std::vector<float> actual(size_t(g.hidden), 0.f);
    expert.forward(input, route, actual);
    const auto expected = reference(g, weights, input, route, counts);
    double error2 = 0, norm2 = 0, max_abs = 0, peak = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = double(actual[i]) - expected[i];
        error2 += d * d;
        norm2 += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(d));
        peak = std::max(peak, std::abs(double(expected[i])));
    }
    const double relative_l2 = std::sqrt(error2 / std::max(norm2, 1e-30));
    std::printf("  %s: relative_L2=%.8g max_abs=%.8g reference_peak=%.8g\n", name.c_str(), relative_l2, max_abs, peak);
    // Fixed before real-weight execution. BF16 boundaries can turn a reduction's last-bit difference into one ULP.
    check(std::isfinite(relative_l2) && relative_l2 <= 0.002 && max_abs <= 1e-5 + 0.02 * peak, name);
}
struct Synthetic {
    std::array<std::vector<uint8_t>, 3> storage;
    ExpertView weights;
    Synthetic(const Geometry& g) {
        std::mt19937 rng(717);
        for (int role = 0; role < 3; ++role) {
            const int width = role == 2 ? g.expert_width : g.hidden;
            const int rows = role == 2 ? g.hidden : g.expert_width;
            auto& bytes = storage[size_t(role)];
            bytes.resize(size_t(width) * size_t(rows) / 32 * 17);
            for (size_t b = 0; b < bytes.size(); b += 17) {
                bytes[b] = 119 + uint8_t(rng() % 4);
                for (size_t j = 1; j < 17; ++j)
                    bytes[b + j] = uint8_t(rng());
            }
            weights.parts[size_t(role)].data = bytes;
            weights.parts[size_t(role)].type = GGML_TYPE_MXFP4;
        }
    }
};
void run_cases(const Geometry& g, const ExpertView& weights, const char* label) {
    CpuExpert expert(g, weights);
    const auto scratch = expert.scratch_bytes();
    std::vector<float> input(size_t(g.hidden), 0.f);
    for (int i = 0; i < g.hidden; ++i)
        input[size_t(i)] = std::sin(float(i) * 0.17f) + 0.3f * std::cos(float(i) * 0.071f);
    ClampCounts counts;
    compare(g, weights, input, 1.f, expert, std::string(label) + " ordinary input", counts);
    compare(g, weights, input, 0.37f, expert, std::string(label) + " route before down quantization", counts);
    for (float& x : input)
        x *= 16.f;
    compare(g, weights, input, 1.5f, expert, std::string(label) + " clamp-active input", counts);
    check(counts.positive_gate && counts.negative_gate && counts.positive_up && counts.negative_up,
          std::string(label) + " covers gate upper-only and up two-sided clamp branches");
    std::vector<float> output(size_t(g.hidden), -1.f);
    expert.forward(input, 0.f, output);
    check(std::all_of(output.begin(), output.end(), [](float x) { return x == 0.f; }),
          std::string(label) + " zero route");
    std::fill(input.begin(), input.end(), 0.f);
    expert.forward(input, 1.f, output);
    check(std::all_of(output.begin(), output.end(), [](float x) { return x == 0.f; }),
          std::string(label) + " reused input address gets fresh activation");
    check(scratch == expert.scratch_bytes(), std::string(label) + " scratch size stays fixed");
    std::printf("  %s: scratch_bytes=%zu weight_bytes=%zu\n", label, scratch, weights.bytes());
}
void synthetic_tests() {
    Geometry g;
    g.layers = 1;
    g.experts = 1;
    g.top_k = 1;
    g.hidden = 128;
    g.expert_width = 64;
    Synthetic fixture(g);
    run_cases(g, fixture.weights, "synthetic");
    CpuExpert expert(g, fixture.weights);
    std::vector<float> input(size_t(g.hidden), 1.f), output(size_t(g.hidden), 77.f);
    rejects("reject wrong input width", [&] { expert.forward(std::span(input).first(32), 1.f, output); });
    rejects("reject negative route", [&] { expert.forward(input, -1.f, output); });
    rejects("reject route above 1017 bound", [&] { expert.forward(input, 2.f, output); });
    rejects("reject nonfinite route", [&] { expert.forward(input, std::numeric_limits<float>::quiet_NaN(), output); });
    input[0] = std::numeric_limits<float>::quiet_NaN();
    rejects("reject nonfinite input", [&] { expert.forward(input, 1.f, output); });
    input[0] = 1e10f;
    rejects("reject input that overflows Q8_0 scale", [&] { expert.forward(input, 1.f, output); });
    check(std::all_of(output.begin(), output.end(), [](float x) { return x == 77.f; }),
          "invalid calls leave output untouched");
    auto broken = fixture.weights;
    broken.parts[0].type = 42;
    rejects("reject unimplemented expert format", [&] { CpuExpert bad(g, broken); });
    broken = fixture.weights;
    broken.parts[1].data = broken.parts[1].data.first(17);
    rejects("reject truncated projection", [&] { CpuExpert bad(g, broken); });
    fixture.storage[0][0] = 255;
    rejects("reject nonfinite weight scale", [&] { CpuExpert bad(g, fixture.weights); });
}
void real_test(const char* path) {
    const auto model = GgufModel::open(path);
    const auto* arch = model.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-expert-fixture")
        throw std::runtime_error("expected a prepared single-expert fixture");
    Geometry g;
    g.layers = 1;
    g.experts = 1;
    g.top_k = 1;
    g.hidden = 5120;
    g.expert_width = 2304;
    const ExpertIndex index(model, g);
    strata::kernels::cpu::NativeFmt fmt;
    std::string err;
    check(!strata::kernels::cpu::native_fmt(39, 39, 5120, 2304, fmt, err) &&
              err.find("pool's buffers") != std::string::npos,
          "existing Qwen pool rejects this shape; the adapter uses caller-sized buffers");
    const auto* prefix = model.meta().get("strata.deepseek41.source_prefix");
    std::printf("Real expert source: %s; baseline: MXFP4 weights + ggml Q8_0 activations\n",
                prefix ? prefix->s.c_str() : "unknown");
    run_cases(g, index.view(0, 0), "real-1017");
}
} // namespace
int main(int argc, char** argv) {
    try {
        ggml_cpu_init();
        synthetic_tests();
        if (argc == 3 && std::string(argv[1]) == "--real-expert")
            real_test(argv[2]);
        else if (argc != 1)
            throw std::runtime_error("usage: deepseek41_expert_test [--real-expert FIXTURE.gguf]");
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
