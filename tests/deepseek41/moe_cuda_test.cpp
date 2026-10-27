#include "strata/deepseek41/moe_cuda.hpp"
#include "strata/deepseek41/expert_source.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "../core/gguf_fixture.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) {
    ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void ck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void rejects(const std::string& name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::exception& e) { check(true, name + ": " + e.what()); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct TempDir {
    fs::path path = fs::temp_directory_path() /
        ("strata-ds41-moe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() { std::error_code e; fs::remove_all(path, e); }
};
float bf16(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
float fp8(unsigned code) {
    const unsigned e = (code >> 3) & 15, f = code & 7;
    const float v = e ? std::ldexp(1.f + float(f) / 8.f, int(e) - 7) : float(f) / 512.f;
    return code & 128 ? -v : v;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
std::vector<float> download(const float* p, int n, cudaStream_t stream) {
    std::vector<float> y(static_cast<size_t>(n));
    ck(cudaMemcpyAsync(y.data(), p, y.size() * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return y;
}
// Independent FP8 oracle: frexp scale selection and exhaustive nearest-grid
// search/tie-to-even. Does not use CUDA casts or the production scale bit math.
std::vector<uint16_t> quantize(const std::vector<float>& x) {
    std::vector<uint16_t> result(x.size());
    for (size_t block = 0; block < x.size(); block += 32) {
        float amax = 1e-4f;
        for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::abs(bf16(x[block + j])));
        int exponent = 0;
        const float mantissa = std::frexp(amax * (1.f / 448.f), &exponent);
        const float scale = std::ldexp(1.f, exponent - (mantissa == .5f));
        for (size_t j = 0; j < 32; ++j) {
            const float value = bf16(x[block + j]);
            const float mag = std::abs(value / scale);
            float best = INFINITY; unsigned code = 0;
            for (unsigned candidate = 0; candidate < 127; ++candidate) {
                const float error = std::abs(mag - fp8(candidate));
                if (error < best || (error == best && candidate % 2 == 0)) {
                    best = error; code = candidate;
                }
            }
            result[block + j] = kernels::bf16_from_f32(std::copysign(fp8(code) * scale, value));
        }
    }
    return result;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name) {
    double error2 = 0., norm2 = 0., maximum = 0., peak = 0.;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i];
        error2 += delta * delta; norm2 += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
    }
    const double l2 = std::sqrt(error2 / std::max(norm2, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), l2, maximum, peak);
    check(std::isfinite(l2) && l2 <= .003 && maximum <= 1e-5 + .02 * peak, name);
}
std::vector<float> projection(const Fp8MatrixView& w, const std::vector<float>& x) {
    const auto q = quantize(x);
    std::vector<float> result(size_t(w.rows), 0.f);
    for (int r = 0; r < w.rows; ++r) {
        double sum = 0.;
        for (int c = 0; c < w.cols; ++c) {
            const float weight = std::ldexp(fp8(w.values[size_t(r) * w.cols + c]),
                int(w.scales[size_t(r / 32) * (w.cols / 32) + c / 32]) - 127);
            sum += double(weight) * kernels::f32_from_bf16(q[size_t(c)]);
        }
        result[size_t(r)] = bf16(float(sum));
    }
    return result;
}
std::vector<float> shared_reference(const SharedExpertWeights& w, const std::vector<float>& x) {
    const auto gate = projection(w.parts[0], x), up = projection(w.parts[1], x);
    std::vector<float> hidden(gate.size());
    for (size_t i = 0; i < hidden.size(); ++i) {
        const float g = std::min(gate[i], 10.f), u = std::clamp(up[i], -10.f, 10.f);
        hidden[i] = bf16((g / (1.f + std::exp(-g))) * u);
    }
    return projection(w.parts[2], hidden);
}
void quantization_tests() {
    Stream stream;
    core::DeviceArena arena(5120 * 6 + 512);
    auto* input = static_cast<float*>(arena.alloc(5120 * 4));
    auto* output = static_cast<uint16_t*>(arena.alloc(5120 * 2));
    std::vector<float> values;
    for (int c = 0; c < 127; ++c) {
        values.push_back(fp8(unsigned(c))); values.push_back(-fp8(unsigned(c)));
        if (c < 126) {
            const float middle = (fp8(unsigned(c)) + fp8(unsigned(c + 1))) * .5f;
            values.push_back(middle); values.push_back(-middle);
        }
    }
    // Anchor each block at 448 to expose every representable code and midpoint.
    std::vector<float> x((values.size() + 30) / 31 * 32, 448.f);
    for (size_t i = 0; i < values.size(); ++i) x[(i / 31) * 32 + i % 31] = values[i];
    for (float scale : {1.f, .125f, 256.f, 1e-8f, 0.f}) {
        auto scaled = x;
        for (float& v : scaled) v *= scale;
        const auto expected = quantize(scaled);
        std::vector<uint16_t> actual(expected.size());
        ck(cudaMemcpyAsync(input, scaled.data(), scaled.size() * 4, cudaMemcpyHostToDevice, stream.value));
        quantize_fp8_bf16(input, output, int(scaled.size()), stream.value);
        ck(cudaMemcpyAsync(actual.data(), output, actual.size() * 2, cudaMemcpyDeviceToHost, stream.value));
        ck(cudaStreamSynchronize(stream.value));
        check(actual == expected, "FP8 grid/ties/signs/floor match independent oracle at scale " + std::to_string(scale));
    }
    rejects("FP8 non-block width rejected", [&] { quantize_fp8_bf16(input, output, 33, stream.value); });
    rejects("FP8 overlapping buffers rejected", [&] { quantize_fp8_bf16(input, reinterpret_cast<uint16_t*>(input), 32, stream.value); });
    rejects("FP8 null stream rejected", [&] { quantize_fp8_bf16(input, output, 32, nullptr); });
}
void shared_cases(const Geometry& g, const SharedExpertWeights& weights, const std::vector<std::vector<float>>& inputs,
                  const std::vector<std::vector<float>>& expected = {}, const GgufModel* golden = nullptr) {
    CudaSharedExpert expert(g, weights);
    Stream stream;
    core::DeviceArena io(size_t(g.hidden) * 10 + 512);
    auto* x = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* y = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* q = static_cast<uint16_t*>(io.alloc(size_t(g.hidden) * 2));
    const auto bytes = expert.device_bytes();
    for (size_t c = 0; c < inputs.size(); ++c) {
        ck(cudaMemcpyAsync(x, inputs[c].data(), inputs[c].size() * 4, cudaMemcpyHostToDevice, stream.value));
        expert.forward(x, y, stream.value);
        const auto actual = download(y, g.hidden, stream.value);
        compare(actual, expected.empty() ? shared_reference(weights, inputs[c]) : expected[c],
                "shared expert case " + std::to_string(c));
        check(std::all_of(actual.begin(), actual.end(), [](float v) { return std::isfinite(v) && v == bf16(v); }),
              "shared output is finite BF16-valued F32");
        if (golden) {
            const auto* t = golden->find("expected.quantized." + std::to_string(c));
            if (!t || !golden->in_bounds(*t, 0) || tensor_payload_bytes(*t) != size_t(g.hidden) * 2)
                throw std::runtime_error("Missing quantization oracle");
            std::vector<uint16_t> actual_q(size_t(g.hidden), 0);
            quantize_fp8_bf16(x, q, g.hidden, stream.value);
            ck(cudaMemcpyAsync(actual_q.data(), q, actual_q.size() * 2, cudaMemcpyDeviceToHost, stream.value));
            ck(cudaStreamSynchronize(stream.value));
            check(std::memcmp(actual_q.data(), golden->shard(0).tensor_data(*t), actual_q.size() * 2) == 0,
                  "real input FP8 activation bytes match independent NumPy oracle");
        }
    }
    check(expert.device_bytes() == bytes && expert.weight_bytes() == size_t(g.hidden) * g.expert_width * 6,
          "shared weights and scratch allocations stay fixed");
    std::printf("shared weight bytes=%zu device bytes=%zu\n", expert.weight_bytes(), expert.device_bytes());
    rejects("shared input/output overlap rejected", [&] { expert.forward(x, x, stream.value); });
    auto bad = weights; bad.parts[0].rows += 32;
    rejects("shared weight shape mismatch rejected", [&] { CudaSharedExpert wrong(g, bad); });
}
void moe_cases(const Geometry& g, const ExpertIndex& index, const SharedExpertWeights& weights,
               const std::vector<uint16_t>& router, const std::vector<float>& bias, const std::vector<float>& input) {
    Stream stream;
    core::DeviceArena io(size_t(g.hidden) * 12 + 512);
    auto* x = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* y = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* shared_out = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    ck(cudaMemcpyAsync(x, input.data(), input.size() * 4, cudaMemcpyHostToDevice, stream.value));
    Mxfp4ExpertSource ref_source(g, index);
    CudaRoutedLayer routed(g, 0, router, bias, ref_source, 6);
    CudaSharedExpert shared(g, weights);
    const auto route = routed.forward(x, y, stream.value);
    shared.forward(x, shared_out, stream.value);
    auto expected = download(y, g.hidden, stream.value);
    const auto shared_values = download(shared_out, g.hidden, stream.value);
    const auto routed_values = expected;
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = bf16(expected[i] + shared_values[i]);
    check(!same(expected, routed_values), "fixture detects omitted shared branch/final BF16 cast");
    for (int slots : {6, 2}) {
        Mxfp4ExpertSource source(g, index);
        CudaMoE moe(g, 0, router, bias, source, slots, weights);
        const auto allocation = moe.device_bytes();
        for (int iteration = 0; iteration < 3; ++iteration) {
            const auto actual_route = moe.forward(x, y, stream.value);
            check(actual_route.ids == route.ids && actual_route.weights == route.weights,
                  "MoE preserves routed IDs and weights");
            check(same(download(y, g.hidden, stream.value), expected),
                  "MoE is BF16(routed + unweighted shared), slots=" + std::to_string(slots) +
                  " pass=" + std::to_string(iteration));
        }
        check(moe.device_bytes() == allocation, "MoE allocations stable across cold/warm/reloads");
        check(slots == 6 ? moe.stats().misses == 6 && moe.stats().hits == 12 && source.reads() == 6
                         : moe.stats().misses == 18 && moe.stats().evictions == 16 && source.reads() == 18,
              "MoE retains exact expert cache read/eviction accounting");
        std::printf("MoE slots=%d device bytes=%zu expert H2D=%llu\n", slots, allocation,
                    (unsigned long long)moe.stats().expert_h2d_bytes);
    }
}
void synthetic() {
    Geometry g; g.layers = 1; g.hidden = g.expert_width = 64; g.experts = 384; g.top_k = 6;
    std::array<std::vector<uint8_t>, 3> values, scales;
    SharedExpertWeights weights;
    for (int role = 0; role < 3; ++role) {
        values[role].resize(64 * 64); scales[role] = {124, 125, 126, 123};
        for (size_t i = 0; i < values[role].size(); ++i)
            values[role][i] = uint8_t(48 + (i + size_t(role) * 11) % 64 + ((i / 64 + role) % 3 == 0 ? 128 : 0));
        weights.parts[role] = {values[role], scales[role], 64, 64};
    }
    std::vector<float> input(64), negative(64), zero(64, 0.f);
    for (int i = 0; i < 64; ++i) { input[size_t(i)] = .5f + std::sin(float(i) * .17f); negative[size_t(i)] = -16.f * input[size_t(i)]; }
    shared_cases(g, weights, {input, negative, zero});
    TempDir temp;
    std::vector<fixture::Tensor> tensors;
    for (const char* role : {"gate", "up", "down"})
        tensors.push_back({std::string("blk.0.ffn_") + role + "_exps.weight", {64, 64, 384}, 39, uint8_t(tensors.size() + 1)});
    const auto path = temp.path / "routed.gguf";
    const auto info = fixture::write(path, {}, tensors);
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        for (size_t role = 0; role < 3; ++role) {
            std::vector<uint8_t> data(64 * 64 / 32 * 17 * 384);
            for (size_t i = 0; i < data.size(); ++i) data[i] = fixture::pattern(uint8_t(role + 1), i);
            for (size_t i = 0; i < data.size(); i += 17) data[i] = uint8_t(120 + (i / 17) % 3);
            f.seekp(std::streamoff(info.data_start + info.offsets[role]));
            f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        }
    }
    const auto model = GgufModel::open(path.string()); const ExpertIndex index(model, g);
    std::vector<uint16_t> router(384 * 64, 0); std::vector<float> bias(384, -100.f);
    for (int e = 0; e < 6; ++e) { bias[size_t(e)] = float(e); router[size_t(e) * 64] = kernels::bf16_from_f32(float(e) * .25f); }
    moe_cases(g, index, weights, router, bias, input);
    rejects("untagged raw shared weights rejected", [&] { read_shared_expert(model, g, 0); });
    struct Missing : core::ExpertSource { const uint8_t* blob(int64_t, int64_t) override { return nullptr; } } missing;
    CudaMoE broken(g, 0, router, bias, missing, 2, weights);
    Stream stream; core::DeviceArena io(512);
    auto* x = static_cast<float*>(io.alloc(256)); auto* y = static_cast<float*>(io.alloc(256));
    ck(cudaMemcpy(x, input.data(), 256, cudaMemcpyHostToDevice));
    rejects("MoE propagates missing selected expert", [&] { broken.forward(x, y, stream.value); });
    rejects("MoE cannot reuse a failed execution", [&] { broken.forward(x, y, stream.value); });
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, uint32_t type, size_t n) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != type || !model.in_bounds(*t, shard) || tensor_payload_bytes(*t) != n * sizeof(T))
        throw std::runtime_error("Invalid fixture tensor " + name);
    std::vector<T> out(n); std::memcpy(out.data(), model.shard(shard).tensor_data(*t), n * sizeof(T)); return out;
}
void real(const char* shared_path, const char* routed_path) {
    const auto shared = GgufModel::open(shared_path), routed = GgufModel::open(routed_path);
    const auto* arch = shared.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-shared-fixture") throw std::runtime_error("Expected prepared shared fixture");
    Geometry g; g.layers = 40; g.hidden = 5120; g.expert_width = 2304; g.experts = 384; g.top_k = 6;
    const auto weights = read_shared_expert(shared, g, 0);
    for (int role = 0; role < 3; ++role) {
        const std::string name = role == 0 ? "gate" : role == 1 ? "up" : "down";
        check(fp8_block32_to_bf16(weights.parts[size_t(role)]) ==
              tensor<uint16_t>(shared, "expected." + name + ".bf16", 30, size_t(g.hidden) * g.expert_width),
              "all real " + name + " weight values exactly match independent torch FP8 decoder");
    }
    std::vector<std::vector<float>> inputs, expected;
    for (int i = 0; i < 3; ++i) {
        inputs.push_back(tensor<float>(shared, "input." + std::to_string(i), 0, 5120));
        expected.push_back(tensor<float>(shared, "expected.output." + std::to_string(i), 0, 5120));
    }
    std::printf("Real shared expert: official Expert.forward + independent block32 CPU oracle\n");
    shared_cases(g, weights, inputs, expected, &shared);
    const ExpertIndex index(routed, g, {0});
    const auto router = tensor<uint16_t>(routed, "router.weight", 30, 5120 * 384);
    const auto bias = tensor<float>(routed, "router.bias", 0, 384);
    check(inputs[0] == tensor<float>(routed, "input", 0, 5120), "shared/routed real fixtures use identical seeded input");
    moe_cases(g, index, weights, router, bias, inputs[0]);
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 1 && !(argc == 4 && std::string(argv[1]) == "--real"))
            throw std::runtime_error("usage: deepseek41_moe_cuda_test [--real SHARED.gguf ROUTED.gguf]");
        int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        quantization_tests(); synthetic(); if (argc == 4) real(argv[2], argv[3]);
        std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
