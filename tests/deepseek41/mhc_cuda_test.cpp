#include "strata/deepseek41/mhc_cuda.hpp"
#include "strata/deepseek41/expert_source.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>

using namespace strata;
using namespace strata::deepseek41;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); }
void ck(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void rejects(const std::string& name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::exception& e) { check(true, name + ": " + e.what()); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
float bf16(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
std::vector<float> download(const float* p, size_t n, cudaStream_t stream) {
    std::vector<float> out(n);
    ck(cudaMemcpyAsync(out.data(), p, n * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return out;
}
void upload(float* dst, const std::vector<float>& x, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, x.data(), x.size() * 4, cudaMemcpyHostToDevice, stream));
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name, bool coefficients = false) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double e2 = 0., n2 = 0., maximum = 0., peak = 0.; bool pointwise = true, finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i];
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        e2 += delta * delta; n2 += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        pointwise &= std::abs(delta) <= 2e-6 + 2e-5 * std::abs(double(expected[i]));
    }
    const double l2 = std::sqrt(e2 / std::max(n2, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), l2, maximum, peak);
    check(finite && (coefficients ? pointwise : l2 <= .003 && maximum <= 1e-5 + .02 * peak), name);
}
std::vector<float> normalized(const std::vector<float>& x, const std::vector<float>& pre,
                              std::span<const uint16_t> norm, float eps = 1e-20f) {
    const size_t n = norm.size(); std::vector<float> out(n); double squares = 0.;
    for (size_t d = 0; d < n; ++d) {
        float sum = 0.f;
        for (size_t i = 0; i < 4; ++i) sum += pre[i] * bf16(x[i * n + d]);
        out[d] = bf16(sum); squares += double(out[d]) * out[d];
    }
    const float inv = 1.f / std::sqrt(float(squares / n) + eps);
    for (size_t d = 0; d < n; ++d) out[d] = bf16((out[d] * inv) * kernels::f32_from_bf16(norm[d]));
    return out;
}
struct Reference { std::vector<float> pre, post, comb, norm, output; };
std::vector<float> post_reference(const std::vector<float>& residual, const std::vector<float>& sublayer,
                                  const std::vector<float>& post, const std::vector<float>& comb, bool transpose = false) {
    const size_t n = sublayer.size(); std::vector<float> out(4 * n);
    for (size_t dest = 0; dest < 4; ++dest)
        for (size_t d = 0; d < n; ++d) {
            float mixed = 0.f;
            for (size_t source = 0; source < 4; ++source)
                mixed += comb[transpose ? dest * 4 + source : source * 4 + dest] * bf16(residual[source * n + d]);
            out[dest * n + d] = bf16(post[dest] * bf16(sublayer[d]) + mixed);
        }
    return out;
}
Reference reference(const MhcWeights& w, const std::vector<float>& x, const std::vector<float>& incoming,
                    const std::vector<float>& sublayer) {
    Reference r; r.pre.resize(4); r.post.resize(4); r.comb.resize(16);
    // Independent scalar FP64 projection/statistic, then F32 graph operations.
    double squares = 0.;
    for (float v : x) squares += double(bf16(v)) * bf16(v);
    const float inv = 1.f / std::sqrt(float(squares / x.size()) + 1e-20f);
    std::array<float, 24> mixes{};
    for (size_t row = 0; row < 24; ++row) {
        double sum = 0.;
        for (size_t col = 0; col < x.size(); ++col) sum += double(w.projection[row * x.size() + col]) * bf16(x[col]);
        mixes[row] = float(sum) * inv;
    }
    for (int i = 0; i < 4; ++i) {
        r.pre[size_t(i)] = 1.f / (1.f + std::exp(-(mixes[size_t(i)] * w.scale[0] + w.base[size_t(i)]))) + 1e-6f;
        r.post[size_t(i)] = 2.f / (1.f + std::exp(-(mixes[size_t(i + 4)] * w.scale[1] + w.base[size_t(i + 4)])));
    }
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) { const size_t i = size_t(row * 4 + col); r.comb[i] = mixes[8 + i] * w.scale[2] + w.base[8 + i]; }
        const float maximum = *std::max_element(r.comb.begin() + row * 4, r.comb.begin() + row * 4 + 4);
        float sum = 0.f;
        for (int col = 0; col < 4; ++col) { auto& v = r.comb[size_t(row * 4 + col)]; v = std::exp(v - maximum); sum += v; }
        for (int col = 0; col < 4; ++col) r.comb[size_t(row * 4 + col)] = r.comb[size_t(row * 4 + col)] / sum + 1e-6f;
    }
    auto column_norm = [&] {
        for (int col = 0; col < 4; ++col) {
            float sum = 0.f;
            for (int row = 0; row < 4; ++row) sum += r.comb[size_t(row * 4 + col)];
            for (int row = 0; row < 4; ++row) r.comb[size_t(row * 4 + col)] /= sum + 1e-6f;
        }
    };
    column_norm();
    for (int iteration = 1; iteration < 20; ++iteration) {
        for (int row = 0; row < 4; ++row) {
            const float sum = std::accumulate(r.comb.begin() + row * 4, r.comb.begin() + row * 4 + 4, 0.f);
            for (int col = 0; col < 4; ++col) r.comb[size_t(row * 4 + col)] /= sum + 1e-6f;
        }
        column_norm();
    }
    r.norm = normalized(x, incoming, w.norm);
    r.output = post_reference(x, sublayer, r.post, r.comb);
    return r;
}
void run_cases(const Geometry& g, const MhcWeights& weights, const std::vector<std::vector<float>>& inputs,
                const std::vector<std::vector<float>>& incoming, const std::vector<std::vector<float>>& sublayers,
                const std::vector<Reference>& golden = {}, const std::string& label = "synthetic") {
    Stream stream, other;
    CudaMhc mhc(g, weights);
    core::DeviceArena arena(size_t(g.hidden) * 36 + 1024);
    auto* x = static_cast<float*>(arena.alloc(size_t(g.hidden) * 16));
    auto* pre = static_cast<float*>(arena.alloc(16));
    auto* sub = static_cast<float*>(arena.alloc(size_t(g.hidden) * 4));
    auto* y = static_cast<float*>(arena.alloc(size_t(g.hidden) * 16));
    const auto allocation = mhc.device_bytes();
    rejects("finish without prepare rejected", [&] { mhc.finish(sub, y, stream.value); });
    rejects("null stream rejected", [&] { mhc.prepare(x, pre, nullptr); });
    rejects("overlapping mHC inputs rejected", [&] { mhc.prepare(x, x, stream.value); });
    for (size_t c = 0; c < inputs.size(); ++c) {
        const auto expected = golden.empty() ? reference(weights, inputs[c], incoming[c], sublayers[c]) : golden[c];
        upload(x, inputs[c], stream.value); upload(pre, incoming[c], stream.value); upload(sub, sublayers[c], stream.value);
        const auto state = mhc.prepare(x, pre, stream.value);
        const auto name = label + " case=" + std::to_string(c);
        compare(download(state.next_pre, 4, stream.value), expected.pre, name + " next_pre", true);
        compare(download(state.post, 4, stream.value), expected.post, name + " post", true);
        compare(download(state.comb, 16, stream.value), expected.comb, name + " comb", true);
        compare(download(state.normalized, size_t(g.hidden), stream.value), expected.norm, name + " normalized");
        if (c == 0) {
            rejects("duplicate prepare rejected", [&] { mhc.prepare(x, pre, stream.value); });
            rejects("cross-stream finish rejected", [&] { mhc.finish(sub, y, other.value); });
            rejects("overlapping finish buffers rejected", [&] { mhc.finish(sub, sub, stream.value); });
            check(!same(expected.norm, normalized(inputs[c], expected.pre, weights.norm)), "fixture distinguishes incoming from newly computed pre");
            check(!same(expected.output, post_reference(inputs[c], sublayers[c], expected.post, expected.comb, true)),
                  "fixture distinguishes source/destination matrix orientation");
        }
        // Saved residual must not depend on the caller's buffer after prepare.
        ck(cudaMemsetAsync(x, 0, size_t(g.hidden) * 16, stream.value));
        mhc.finish(sub, y, stream.value);
        const auto output = download(y, size_t(g.hidden) * 4, stream.value);
        compare(output, expected.output, name + " output");
        check(std::all_of(output.begin(), output.end(), [](float v) { return std::isfinite(v) && v == bf16(v); }), "mHC residual is finite and BF16-valued");
    }
    rejects("instance cannot switch execution stream", [&] { mhc.prepare(x, pre, other.value); });
    check(mhc.device_bytes() == allocation, "mHC owned device allocation remains fixed");
    std::printf("%s mHC device bytes=%zu (cuBLAS resources excluded)\n", label.c_str(), allocation);
}
void synthetic() {
    Geometry g; g.layers = 1; g.hidden = 64;
    std::vector<float> projection(96 * 64), scale{.5f, -.7f, .6f}, base(24);
    std::vector<uint16_t> norm(64);
    for (size_t i = 0; i < projection.size(); ++i) projection[i] = .037f * std::sin(float(i) * .13f);
    for (size_t i = 0; i < base.size(); ++i) base[i] = 1.5f * std::cos(float(i) * 1.3f);
    for (size_t i = 0; i < norm.size(); ++i) norm[i] = kernels::bf16_from_f32(.75f + float(i % 7) / 16.f);
    const MhcWeights weights{projection, scale, base, norm};
    std::vector<float> x(256), scaled(256), tiny(256), zero(256, 0.f), sub(64);
    for (size_t i = 0; i < x.size(); ++i) { x[i] = bf16(std::sin(float(i) * .17f) + float(i / 64) * .4f); scaled[i] = bf16(x[i] * 16); tiny[i] = bf16(x[i] * 1e-12f); }
    for (size_t i = 0; i < sub.size(); ++i) sub[i] = bf16(.3f * std::cos(float(i) * .11f));
    const std::vector<float> identity{1, 0, 0, 0}, mixed{.1f, .7f, .2f, .4f};
    check(!same(normalized(tiny, mixed, norm), normalized(tiny, mixed, norm, 1e-6f)), "tiny input detects wrong default RMS epsilon");
    run_cases(g, weights, {x, x, scaled, tiny, zero}, {identity, mixed, mixed, mixed, identity}, {sub, sub, sub, sub, sub});
    auto bad = weights; bad.projection = std::span(projection).first(projection.size() - 1);
    rejects("wrong projection shape rejected", [&] { CudaMhc nope(g, bad); });
    scale[0] = NAN;
    rejects("nonfinite mHC weight rejected", [&] { CudaMhc nope(g, weights); });
    scale[0] = .5f; norm[0] = 0x7f80;
    rejects("nonfinite norm weight rejected", [&] { CudaMhc nope(g, weights); });
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, uint32_t type, size_t count) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != type || !model.in_bounds(*t, shard) || tensor_payload_bytes(*t) != count * sizeof(T))
        throw std::runtime_error("Invalid fixture tensor " + name);
    std::vector<T> out(count); std::memcpy(out.data(), model.shard(shard).tensor_data(*t), count * sizeof(T)); return out;
}
void integration(const Geometry& g, const GgufModel& fixture, const MhcWeights& weights,
                   const char* shared_path, const char* routed_path) {
    const auto shared = GgufModel::open(shared_path), routed = GgufModel::open(routed_path);
    const auto shared_weights = read_shared_expert(shared, g, 0);
    const ExpertIndex index(routed, g, {0});
    const auto router = tensor<uint16_t>(routed, "router.weight", 30, 5120 * 384);
    const auto bias = tensor<float>(routed, "router.bias", 0, 384);
    const auto residual = tensor<float>(fixture, "ffn.0.input", 0, 20480), incoming = tensor<float>(fixture, "ffn.0.incoming", 0, 4);
    const auto normalized_input = tensor<float>(fixture, "ffn.0.normalized", 0, 5120);
    const auto post = tensor<float>(fixture, "ffn.0.post", 0, 4), comb = tensor<float>(fixture, "ffn.0.comb", 0, 16);
    const auto next_pre = tensor<float>(fixture, "ffn.0.pre", 0, 4);
    check(same(normalized_input, tensor<float>(routed, "input", 0, 5120)), "routed fixture uses actual reference-normalized FFN input");
    Stream stream;
    core::DeviceArena io(12 * size_t(g.hidden) * 4 + 1024);
    auto* x = static_cast<float*>(io.alloc(size_t(g.hidden) * 16));
    auto* prev = static_cast<float*>(io.alloc(16));
    auto* next = static_cast<float*>(io.alloc(16));
    auto* y = static_cast<float*>(io.alloc(size_t(g.hidden) * 16));
    auto* norm = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    auto* moe_y = static_cast<float*>(io.alloc(size_t(g.hidden) * 4));
    upload(x, residual, stream.value); upload(prev, incoming, stream.value); upload(norm, normalized_input, stream.value);
    Mxfp4ExpertSource reference_source(g, index);
    CudaMoE reference_moe(g, 0, router, bias, reference_source, 6, shared_weights);
    const auto expected_route = reference_moe.forward(norm, moe_y, stream.value);
    const auto expected = post_reference(residual, download(moe_y, 5120, stream.value), post, comb);
    const auto* ids = routed.meta().get("strata.deepseek41.selected_ids");
    if (!ids || ids->items.size() != 6) throw std::runtime_error("Missing official route metadata");
    const auto route_weights = tensor<float>(routed, "expected.weights", 0, 6);
    bool route_ok = true;
    for (int k = 0; k < 6; ++k) route_ok &= expected_route.ids[k] == int32_t(ids->items[size_t(k)].u) && std::abs(expected_route.weights[k] - route_weights[size_t(k)]) <= 2e-6f;
    check(route_ok, "reference-normalized input preserves official six route IDs and weights");
    std::vector<float> first;
    for (int slots : {6, 2}) {
        Mxfp4ExpertSource source(g, index);
        CudaMoE moe(g, 0, router, bias, source, slots, shared_weights);
        CudaMhc mhc(g, weights);
        const auto allocation = mhc.device_bytes() + moe.device_bytes();
        for (int pass = 0; pass < 3; ++pass) {
            const auto route = mhc.forward_moe(moe, x, prev, y, next, stream.value);
            check(route.ids == expected_route.ids, "mHC/MoE retains original route IDs");
            compare(std::vector<float>(route.weights.begin(), route.weights.end()), route_weights, "mHC/MoE official route weights", true);
            const auto actual = download(y, 20480, stream.value);
            compare(actual, expected, "real complete FFN slots=" + std::to_string(slots) + " pass=" + std::to_string(pass));
            compare(download(next, 4, stream.value), next_pre, "FFN returns delayed pre for next sublayer", true);
            if (first.empty()) first = actual; else check(same(first, actual), "complete FFN cold/warm/eviction outputs bit-identical");
        }
        check(mhc.device_bytes() + moe.device_bytes() == allocation, "complete FFN owned allocations remain fixed");
        check(slots == 6 ? source.reads() == 6 && moe.stats().hits == 12 && moe.stats().misses == 6
                         : source.reads() == 18 && moe.stats().misses == 18 && moe.stats().evictions == 16,
              "complete FFN cache reads and evictions unchanged");
        std::printf("complete FFN slots=%d owned device bytes=%zu expert H2D=%llu\n", slots, allocation,
                    (unsigned long long)moe.stats().expert_h2d_bytes);
        rejects("FFN output/input overlap rejected", [&] { mhc.forward_moe(moe, x, prev, x, next, stream.value); });
    }
    struct Missing : core::ExpertSource { const uint8_t* blob(int64_t, int64_t) override { return nullptr; } } missing;
    CudaMoE broken(g, 0, router, bias, missing, 2, shared_weights); CudaMhc mhc(g, weights);
    rejects("FFN propagates missing selected expert", [&] { mhc.forward_moe(broken, x, prev, y, next, stream.value); });
    rejects("failed mHC instance cannot be reused", [&] { mhc.prepare(x, prev, stream.value); });
}
void real(const char* mhc_path, const char* shared_path, const char* routed_path) {
    const auto fixture = GgufModel::open(mhc_path);
    const auto* arch = fixture.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-mhc-fixture") throw std::runtime_error("Expected mHC fixture");
    Geometry g; g.layers = 40; g.hidden = 5120; g.expert_width = 2304; g.experts = 384; g.top_k = 6;
    for (const auto branch : {MhcBranch::ffn, MhcBranch::attention}) {
        const std::string kind = branch == MhcBranch::ffn ? "ffn" : "attn";
        const auto weights = read_mhc_weights(fixture, g, 0, branch);
        std::vector<std::vector<float>> inputs, incoming, sublayers; std::vector<Reference> expected;
        for (int c = 0; c < 4; ++c) {
            const std::string p = kind + "." + std::to_string(c) + ".";
            inputs.push_back(tensor<float>(fixture, p + "input", 0, 20480));
            incoming.push_back(tensor<float>(fixture, p + "incoming", 0, 4));
            sublayers.push_back(tensor<float>(fixture, p + "sublayer", 0, 5120));
            expected.push_back({tensor<float>(fixture, p + "pre", 0, 4), tensor<float>(fixture, p + "post", 0, 4),
                tensor<float>(fixture, p + "comb", 0, 16), tensor<float>(fixture, p + "normalized", 0, 5120), tensor<float>(fixture, p + "output", 0, 20480)});
        }
        run_cases(g, weights, inputs, incoming, sublayers, expected, "real " + kind);
    }
    integration(g, fixture, read_mhc_weights(fixture, g, 0, MhcBranch::ffn), shared_path, routed_path);
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 1 && !(argc == 5 && std::string(argv[1]) == "--real"))
            throw std::runtime_error("usage: deepseek41_mhc_cuda_test [--real MHC.gguf SHARED.gguf ROUTED.gguf]");
        int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        synthetic(); if (argc == 5) real(argv[2], argv[3], argv[4]);
        std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
