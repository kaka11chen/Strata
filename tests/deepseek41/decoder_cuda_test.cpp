#include "strata/deepseek41/decoder_cuda.hpp"
#include "strata/deepseek41/expert_source.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
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
constexpr int hidden = 5120, residual_size = 4 * hidden;
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); }
void ck(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void rejects(const std::string& name, const std::function<void()>& fn, bool execution = false) {
    try { fn(); check(false, name); }
    catch (const std::invalid_argument&) { check(!execution, name); }
    catch (const std::exception& e) { check(execution, name + ": " + e.what()); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
Geometry geometry() {
    Geometry g; g.layers = 40; g.hidden = hidden; g.expert_width = 2304; g.experts = 384; g.top_k = 6; g.compress_ratios = {0}; return g;
}
float bf(float value) { return kernels::f32_from_bf16(kernels::bf16_from_f32(value)); }
void upload(float* dst, const std::vector<float>& input, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, input.data(), input.size() * 4, cudaMemcpyHostToDevice, stream));
}
std::vector<float> download(const float* data, size_t n, cudaStream_t stream) {
    std::vector<float> out(n); ck(cudaMemcpyAsync(out.data(), data, n * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return out;
}
std::vector<float> cache(const uint16_t* data, cudaStream_t stream) {
    std::vector<uint16_t> raw(128 * 512); ck(cudaMemcpyAsync(raw.data(), data, raw.size() * 2, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); std::vector<float> out; out.reserve(raw.size());
    for (auto v : raw) out.push_back(kernels::f32_from_bf16(v));
    return out;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name, bool coeff = false) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double error = 0., norm = 0., peak = 0., max_abs = 0.; bool finite = true, rounded = true, pointwise = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i]; error += delta * delta; norm += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]); rounded &= actual[i] == bf(actual[i]);
        pointwise &= std::abs(delta) <= 2e-6 + 2e-5 * std::abs(double(expected[i]));
    }
    const double rel = std::sqrt(error / std::max(norm, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), rel, max_abs, peak);
    check(finite && (coeff ? pointwise : rounded && rel <= .005 && max_abs <= 1e-5 + .03 * peak), name);
}
// CPU residual join; tensors use official source-row/destination-column order.
std::vector<float> join(std::span<const float> residual, std::span<const float> sublayer,
                         std::span<const float> post, std::span<const float> comb) {
    const size_t tokens = sublayer.size() / hidden; std::vector<float> out(tokens * residual_size);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t dest = 0; dest < 4; ++dest)
            for (size_t d = 0; d < hidden; ++d) {
                float mixed = 0.f;
                for (size_t source = 0; source < 4; ++source)
                    mixed += comb[t * 16 + source * 4 + dest] * residual[t * residual_size + source * hidden + d];
                out[t * residual_size + dest * hidden + d] = bf(post[t * 4 + dest] * bf(sublayer[t * hidden + d]) + mixed);
            }
    return out;
}
std::vector<float> collapse_norm(const std::vector<float>& residual, const std::vector<float>& pre) {
    const size_t tokens = pre.size() / 4; std::vector<float> out(tokens * hidden);
    for (size_t t = 0; t < tokens; ++t) {
        double squares = 0.;
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.f;
            for (int s = 0; s < 4; ++s) sum += pre[t * 4 + s] * residual[t * residual_size + size_t(s) * hidden + d];
            out[t * hidden + d] = bf(sum); squares += double(bf(sum)) * bf(sum);
        }
        const float inv = 1.f / std::sqrt(float(squares / hidden) + 1e-20f);
        for (int d = 0; d < hidden; ++d) out[t * hidden + d] = bf(out[t * hidden + d] * inv);
    }
    return out;
}
struct Matrix {
    std::vector<uint8_t> values, scales; int rows, cols;
    Matrix(int r, int c, bool diagonal) : values(size_t(r) * c, 0), scales(size_t(r / 32) * (c / 32), 127), rows(r), cols(c) {
        if (diagonal) for (int i = 0; i < r; ++i) values[size_t(i) * c + i % c] = 0x30; // 0.5
    }
    Fp8MatrixView view() const { return {values, scales, rows, cols}; }
};
struct ConstantMhc {
    std::vector<float> fn = std::vector<float>(96 * hidden, 0.f);
    std::array<float, 3> scale{0.f, 0.f, 0.f};
    std::array<float, 24> base{};
    std::vector<uint16_t> norm = std::vector<uint16_t>(hidden, kernels::bf16_from_f32(1.f));
    explicit ConstantMhc(float shift) {
        for (int i = 0; i < 8; ++i) base[size_t(i)] = shift + float(i) * .2f;
    }
    MhcWeights view() const { return {fn, scale, base, norm}; }
    std::array<float, 4> pre() const {
        std::array<float, 4> v{};
        for (size_t i = 0; i < 4; ++i) v[i] = 1.f / (1.f + std::exp(-base[i])) + 1e-6f;
        return v;
    }
    std::vector<float> post(int count) const {
        std::vector<float> v(size_t(count) * 4);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 2.f / (1.f + std::exp(-base[4 + i % 4]));
        return v;
    }
    std::vector<float> comb(int count) const {
        float v = .25f + 1e-6f;
        // All equal logits: initial row softmax+eps, column norm, 19 row/column pairs.
        for (int step = 0; step < 39; ++step) v /= 4.f * v + 1e-6f;
        return std::vector<float>(size_t(count) * 16, v);
    }
};
struct ZeroSource : core::ExpertSource {
    std::vector<uint8_t> bytes;
    int calls = 0; bool missing = false;
    explicit ZeroSource(const Geometry& g) : bytes(nvfp4_expert_bytes(g), 0) {
        const std::array<float,4> scales{1.f,1.f,1.f,0.f};
        std::memcpy(bytes.data()+bytes.size()-nvfp4_scale_tail_bytes,scales.data(),nvfp4_scale_tail_bytes);
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (layer != 0 || expert < 0 || expert >= 384) throw std::runtime_error("Unexpected original expert ID");
        ++calls; return missing ? nullptr : bytes.data();
    }
};
struct SyntheticWeights {
    std::array<Matrix, 5> attention{Matrix(1280,5120,true), Matrix(32768,1280,true), Matrix(512,5120,true), Matrix(8192,4096,true), Matrix(5120,8192,true)};
    std::array<Matrix, 3> shared{Matrix(2304,5120,false), Matrix(2304,5120,false), Matrix(5120,2304,false)};
    std::vector<uint16_t> qnorm = std::vector<uint16_t>(1280, kernels::bf16_from_f32(1.f));
    std::vector<uint16_t> kvnorm = std::vector<uint16_t>(512, kernels::bf16_from_f32(1.f));
    std::vector<uint16_t> router = std::vector<uint16_t>(384 * hidden, 0);
    std::vector<float> sinks = std::vector<float>(64, 0.f), bias = std::vector<float>(384);
    ConstantMhc attn_mhc{-.7f}, ffn_mhc{-1.4f};
    SyntheticWeights() { for (int i = 0; i < 384; ++i) bias[size_t(i)] = -float(i); }
    DecoderWeights view() const {
        DecoderWeights w; w.attention_mhc = attn_mhc.view(); w.ffn_mhc = ffn_mhc.view();
        for (size_t i = 0; i < 5; ++i) w.attention.projections[i] = attention[i].view();
        w.attention.q_norm = qnorm; w.attention.kv_norm = kvnorm; w.attention.sink = sinks;
        for (size_t i = 0; i < 3; ++i) w.shared.parts[i] = shared[i].view();
        w.router = router; w.bias = bias; return w;
    }
};
void synthetic() {
    const auto g = geometry(); SyntheticWeights storage; const auto w = storage.view(); ZeroSource source(g);
    constexpr int count = 66;
    std::vector<float> input(count * residual_size), incoming(count * 4);
    for (int t = 0; t < count; ++t) {
        for (int s = 0; s < 4; ++s) {
            incoming[size_t(t) * 4 + s] = .1f + .13f * float((t + s) % 7);
            for (int d = 0; d < hidden; ++d)
                input[size_t(t) * residual_size + s * hidden + d] = bf(std::sin(float(d) * .03f + float(t) * .2f + float(s) * .7f) + .1f * std::cos(float(d) * .07f) + .125f * float(s));
        }
    }
    Stream stream, other;
    core::DeviceArena io(size_t(count) * hidden * 48 + 8192);
    auto* x = static_cast<float*>(io.alloc(input.size() * 4));
    auto* pre = static_cast<float*>(io.alloc(incoming.size() * 4));
    auto* y = static_cast<float*>(io.alloc(input.size() * 4));
    auto* next = static_cast<float*>(io.alloc(incoming.size() * 4));
    auto* norm = static_cast<float*>(io.alloc(size_t(count) * hidden * 4));
    auto* attn_output = static_cast<float*>(io.alloc(size_t(count) * hidden * 4));
    upload(x, input, stream.value); upload(pre, incoming, stream.value);
    upload(norm, collapse_norm(input, incoming), stream.value);
    std::vector<float> expected, expected_cache;
    {
        CudaAttention attention(w.attention);
        attention.forward(norm, attn_output, 65, 0, stream.value);
        attention.forward(norm + 65 * hidden, attn_output + 65 * hidden, 1, 65, stream.value);
        auto middle = join(input, download(attn_output, size_t(count) * hidden, stream.value), storage.attn_mhc.post(count), storage.attn_mhc.comb(count));
        expected = join(middle, std::vector<float>(size_t(count) * hidden, 0.f), storage.ffn_mhc.post(count), storage.ffn_mhc.comb(count));
        expected_cache = cache(attention.kv_cache(), stream.value);
    }
    CudaDecoder decoder(g, w, source, 6);
    const auto bytes = decoder.device_bytes();
    std::vector<Route> routes(count);
    rejects("null stream rejected", [&] { decoder.forward(x, pre, y, next, 1, 0, nullptr); });
    rejects("null input rejected", [&] { decoder.forward(nullptr, pre, y, next, 1, 0, stream.value); });
    rejects("residual input/output overlap rejected", [&] { decoder.forward(x, pre, x, next, 1, 0, stream.value); });
    rejects("coefficient input/output overlap rejected", [&] { decoder.forward(x, pre, y, pre, 1, 0, stream.value); });
    rejects("zero tokens rejected", [&] { decoder.forward(x, pre, y, next, 0, 0, stream.value); });
    rejects("missing prefill rejected", [&] { decoder.forward(x, pre, y, next, 1, 1, stream.value); });
    rejects("sequence bound checked before access", [&] { decoder.forward(x, pre, y, next, 1048577, 0, stream.value); });
    rejects("host route span must match tokens", [&] { decoder.forward(x, pre, y, next, 1, 0, stream.value, routes); });
    check(source.calls == 0 && decoder.next_position() == 0, "validation failures leave cache and position untouched");
    decoder.forward(x, pre, y, next, 65, 0, stream.value, std::span(routes).first(65));
    decoder.forward(x + 65 * residual_size, pre + 65 * 4, y + 65 * residual_size, next + 65 * 4, 1, 65, stream.value, std::span(routes).last(1));
    const auto output = download(y, input.size(), stream.value);
    compare(output, expected, "65-token streamed prefill + decode matches batch-attention/CPU-join composition");
    compare(cache(decoder.kv_cache(), stream.value), expected_cache, "decoder preserves batch-prefill KV state");
    const auto expected_pre = storage.ffn_mhc.pre();
    std::vector<float> expected_next(incoming.size());
    for (size_t i = 0; i < expected_next.size(); ++i) expected_next[i] = expected_pre[i % 4];
    compare(download(next, incoming.size(), stream.value), expected_next, "returns FFN coefficients for each token's next layer", true);
    bool correct_routes = true;
    for (const auto& route : routes)
        for (int k = 0; k < 6; ++k) correct_routes &= route.ids[size_t(k)] == k && std::abs(route.weights[size_t(k)] - .25f) < 1e-6f;
    check(correct_routes, "all original top-6 IDs and route weights retained");
    check(source.calls == 6 && decoder.stats().misses == 6 && decoder.stats().hits == 390, "expert cache reused throughout prefill and decode");
    const auto before = cache(decoder.kv_cache(), stream.value);
    rejects("noncontiguous decode rejected", [&] { decoder.forward(x, pre, y, next, 1, 67, stream.value); });
    rejects("second prefill rejected", [&] { decoder.forward(x, pre, y, next, 1, 0, stream.value); });
    rejects("multi-token continuation rejected", [&] { decoder.forward(x, pre, y, next, 2, 66, stream.value); });
    rejects("stream switch rejected", [&] { decoder.forward(x, pre, y, next, 1, 66, other.value); });
    rejects("reset cannot switch streams", [&] { decoder.reset(other.value); });
    check(decoder.next_position() == count && same(before, cache(decoder.kv_cache(), stream.value)), "invalid continuation preserves live KV and position");
    decoder.reset(stream.value);
    check(decoder.next_position() == 0 && same(cache(decoder.kv_cache(), stream.value), std::vector<float>(128 * 512, 0.f)), "reset clears sequence state");
    decoder.forward(x, pre, y, next, 65, 0, stream.value);
    decoder.forward(x + 65 * residual_size, pre + 65 * 4, y + 65 * residual_size, next + 65 * 4, 1, 65, stream.value);
    check(same(output, download(y, input.size(), stream.value)), "reset/replay is byte-identical with warm expert cache");
    check(source.calls == 6 && decoder.stats().hits == 786 && decoder.device_bytes() == bytes, "reset retains immutable expert cache and fixed allocation");
    decoder.reset(stream.value);
    auto changed = incoming; std::reverse(changed.begin(), changed.begin() + 4); upload(pre, changed, stream.value);
    decoder.forward(x, pre, y, next, 1, 0, stream.value);
    check(!same(download(y, residual_size, stream.value), std::vector<float>(output.begin(), output.begin() + residual_size)), "incoming mHC coefficients affect attention/output");
    rejects("wrong decoder geometry rejected", [&] { auto bad = g; bad.hidden = 64; CudaDecoder invalid(bad, w, source, 2); });
    {
        ZeroSource missing(g); missing.missing = true; CudaDecoder broken(g, w, missing, 2);
        rejects("missing original expert fails complete layer", [&] { broken.forward(x, pre, y, next, 1, 0, stream.value); }, true);
        check(broken.next_position() == 0, "failed layer does not publish a completed position");
        rejects("partial execution cannot be reused", [&] { broken.forward(x, pre, y, next, 1, 0, stream.value); }, true);
        rejects("reset cannot revive a poisoned layer", [&] { broken.reset(stream.value); }, true);
    }
    std::printf("synthetic_decoder_device_bytes=%zu\n", bytes);
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, uint32_t type, size_t n) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != type || tensor_payload_bytes(*t) != n * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Missing/invalid fixture tensor " + name);
    std::vector<T> out(n); std::memcpy(out.data(), model.shard(shard).tensor_data(*t), out.size() * sizeof(T)); return out;
}
void real(const char* fixture_path, const char* mhc_path, const char* attn_path, const char* shared_path, const char* routed_path) {
    const auto g = geometry();
    const auto fixture = GgufModel::open(fixture_path), mhc = GgufModel::open(mhc_path), attn = GgufModel::open(attn_path),
               shared = GgufModel::open(shared_path), routed = GgufModel::open(routed_path);
    const auto* arch = fixture.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-decoder-fixture") throw std::runtime_error("Expected prepared decoder fixture");
    const auto router = tensor<uint16_t>(routed, "router.weight", 30, hidden * 384);
    const auto bias = tensor<float>(routed, "router.bias", 0, 384);
    DecoderWeights weights{read_mhc_weights(mhc, g, 0, MhcBranch::attention), read_mhc_weights(mhc, g, 0, MhcBranch::ffn),
        read_attention_weights(attn, g, 0), read_shared_expert(shared, g, 0), router, bias};
    const ExpertIndex index(routed, g, {0});
    Stream stream, other;
    core::DeviceArena io(3 * size_t(residual_size) * 12 + 8192);
    auto* x = static_cast<float*>(io.alloc(3 * size_t(residual_size) * 4));
    auto* y = static_cast<float*>(io.alloc(3 * size_t(residual_size) * 4));
    auto* pre = static_cast<float*>(io.alloc(3 * 16));
    auto* next = static_cast<float*>(io.alloc(3 * 16));
    auto* norm = static_cast<float*>(io.alloc(hidden * 4));
    auto* moe_output = static_cast<float*>(io.alloc(hidden * 4));
    std::vector<std::vector<float>> expected;
    std::vector<std::vector<Route>> expected_routes;
    {
        Mxfp4ExpertSource source(g, index); CudaMoE moe(g, 0, router, bias, source, 6, weights.shared);
        for (int c = 0; c < 3; ++c) {
            const int tokens = c == 0 ? 3 : 1;
            const auto prefix = "case." + std::to_string(c) + ".";
            const auto normalized = tensor<float>(fixture, prefix + "ffn_input", 0, size_t(tokens) * hidden);
            const auto ids = tensor<float>(fixture, prefix + "route_ids", 0, size_t(tokens) * 6);
            const auto mixing = tensor<float>(fixture, prefix + "route_weights", 0, size_t(tokens) * 6);
            std::vector<float> outputs; std::vector<Route> routes;
            for (int t = 0; t < tokens; ++t) {
                ck(cudaMemcpyAsync(norm, normalized.data() + size_t(t) * hidden, hidden * 4, cudaMemcpyHostToDevice, stream.value));
                const auto route = moe.forward(norm, moe_output, stream.value);
                const auto value = download(moe_output, hidden, stream.value); outputs.insert(outputs.end(), value.begin(), value.end());
                bool correct = true;
                for (size_t k = 0; k < 6; ++k)
                    correct &= route.ids[k] == int32_t(ids[size_t(t) * 6 + k]) && std::abs(route.weights[k] - mixing[size_t(t) * 6 + k]) <= 2e-6f;
                check(correct, prefix + "native MoE baseline retains official routes token=" + std::to_string(t));
                routes.push_back(route);
            }
            expected.push_back(join(tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size), outputs,
                                    tensor<float>(fixture, prefix + "post", 0, size_t(tokens) * 4),
                                    tensor<float>(fixture, prefix + "comb", 0, size_t(tokens) * 16)));
            expected_routes.push_back(routes);
        }
    }
    std::vector<std::vector<float>> first_outputs, first_pre;
    for (int slots : {6, 2}) {
        Mxfp4ExpertSource source(g, index); CudaDecoder decoder(g, weights, source, slots);
        const size_t bytes = decoder.device_bytes();
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) decoder.reset(stream.value);
            for (int c = 0; c < 3; ++c) {
                const int tokens = c == 0 ? 3 : 1, pos = c == 0 ? 0 : c + 2;
                const auto prefix = "case." + std::to_string(c) + ".";
                const auto label = "real decoder slots=" + std::to_string(slots) + " pass=" + std::to_string(pass) + " case=" + std::to_string(c);
                upload(x, tensor<float>(fixture, prefix + "input", 0, size_t(tokens) * residual_size), stream.value);
                upload(pre, tensor<float>(fixture, prefix + "incoming", 0, size_t(tokens) * 4), stream.value);
                std::vector<Route> routes(static_cast<size_t>(tokens));
                decoder.forward(x, pre, y, next, tokens, pos, stream.value, routes);
                const auto actual = download(y, size_t(tokens) * residual_size, stream.value);
                const auto coefficients = download(next, size_t(tokens) * 4, stream.value);
                compare(actual, expected[size_t(c)], label + " CPU graph + native MoE output");
                compare(coefficients, tensor<float>(fixture, prefix + "next_pre", 0, size_t(tokens) * 4), label + " next-layer coefficients", true);
                compare(cache(decoder.kv_cache(), stream.value), tensor<float>(fixture, prefix + "cache", 0, 128 * 512), label + " KV state");
                bool route_ids = true, route_weights = true;
                for (int t = 0; t < tokens; ++t)
                    for (size_t k = 0; k < 6; ++k) {
                        route_ids &= routes[size_t(t)].ids[k] == expected_routes[size_t(c)][size_t(t)].ids[k];
                        route_weights &= std::abs(routes[size_t(t)].weights[k] - expected_routes[size_t(c)][size_t(t)].weights[k]) <= 1e-4f;
                    }
                check(route_ids && route_weights, label + " original routes and weights");
                check(decoder.next_position() == pos + tokens && decoder.device_bytes() == bytes, label + " position and fixed memory");
                if (slots == 6 && pass == 0) { first_outputs.push_back(actual); first_pre.push_back(coefficients); }
                else {
                    check(same(actual, first_outputs[size_t(c)]), label + " output invariant to reset and eviction");
                    check(same(coefficients, first_pre[size_t(c)]), label + " coefficients invariant to reset and eviction");
                }
            }
        }
        const auto stats = decoder.stats();
        check(stats.hits + stats.misses == 60 && stats.expert_h2d_bytes == stats.misses * nvfp4_expert_bytes(g) &&
              stats.router_d2h_bytes == 10 * 384 * 4 && uint64_t(source.reads()) == stats.misses && stats.evictions > 0,
              "complete layer uses existing cache accounting slots=" + std::to_string(slots));
        if (slots == 2) check(stats.misses == 60 && stats.evictions == 58, "two-slot cache forces every original expert load without changing output");
        rejects("real decoder rejects cross-stream reset", [&] { decoder.reset(other.value); });
        std::printf("real_decoder_slots=%d device_bytes=%zu expert_h2d_bytes=%llu hits=%llu misses=%llu evictions=%llu\n", slots, bytes,
                    (unsigned long long)stats.expert_h2d_bytes, (unsigned long long)stats.hits,
                    (unsigned long long)stats.misses, (unsigned long long)stats.evictions);
    }
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 7 && std::string(argv[1]) == "--real"))) {
        std::fprintf(stderr, "usage: deepseek41_decoder_cuda_test [--real DECODER.gguf MHC.gguf ATTENTION.gguf SHARED.gguf ROUTED.gguf]\n"); return 2;
    }
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 1) synthetic(); else real(argv[2], argv[3], argv[4], argv[5], argv[6]);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
