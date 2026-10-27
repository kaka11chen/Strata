#include "strata/deepseek41/attention_cuda.hpp"
#include "strata/deepseek41/shared_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int dim = 512, heads = 64, qs = dim * heads;
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void rejects(const std::string& name, const std::function<void()>& fn) {
    try { fn(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
std::vector<float> download(const float* data, size_t n, cudaStream_t stream) {
    std::vector<float> out(n); ck(cudaMemcpyAsync(out.data(), data, n * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return out;
}
std::vector<float> cache(const uint16_t* data, cudaStream_t stream) {
    std::vector<uint16_t> bits(128 * dim); ck(cudaMemcpyAsync(bits.data(), data, bits.size() * 2, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); std::vector<float> out; out.reserve(bits.size());
    for (auto v : bits) out.push_back(kernels::f32_from_bf16(v));
    return out;
}
void upload(float* dst, const std::vector<float>& src, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, src.data(), src.size() * 4, cudaMemcpyHostToDevice, stream));
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name, bool full = false) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double err = 0., norm = 0., peak = 0., maximum = 0.; bool finite = true, rounded = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i]; err += delta * delta; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]); rounded &= actual[i] == bf(actual[i]);
    }
    const double rel = std::sqrt(err / std::max(norm, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), rel, maximum, peak);
    check(finite && rounded && rel <= (full ? .005 : .003) && maximum <= 1e-5 + (full ? .03 : .02) * peak, name);
}
void rope(float* x, int nheads, int pos, bool inverse) {
    for (int h = 0; h < nheads; ++h)
        for (int pair = 0; pair < 32; ++pair) {
            const float freq = 1.f / std::pow(10000.f, float(pair) / 32.f);
            const float theta = float(pos) * freq, c = std::cos(theta), s = (inverse ? -1.f : 1.f) * std::sin(theta);
            const int at = h * dim + 448 + 2 * pair; const float a = x[at], b = x[at + 1];
            x[at] = bf(a * c - b * s); x[at + 1] = bf(a * s + b * c);
        }
}
void fp8(std::vector<float>& values) {
    std::array<float, 127> grid{};
    for (int c = 0; c < 127; ++c) grid[size_t(c)] = c < 8 ? float(c) / 512.f : std::ldexp(1.f + float(c & 7) / 8.f, (c >> 3) - 7);
    for (size_t start = 0; start < values.size(); start += 32) {
        float maximum = 1e-4f;
        for (size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(values[start + j]));
        int exponent; const float mantissa = std::frexp(maximum / 448.f, &exponent);
        const float scale = std::ldexp(1.f, exponent - (mantissa == .5f));
        for (size_t j = 0; j < 32; ++j) {
            const float value = values[start + j], mag = std::min(std::abs(value) / scale, 448.f);
            const int upper = int(std::lower_bound(grid.begin(), grid.end(), mag) - grid.begin()), lower = std::max(upper - 1, 0);
            const float a = mag - grid[size_t(lower)], b = grid[size_t(upper)] - mag;
            const int code = b < a || (b == a && upper % 2 == 0) ? upper : lower;
            values[start + j] = bf(std::copysign(grid[size_t(code)] * scale, value));
        }
    }
}
// Scalar CPU oracle: double dot products, explicit official 64-candidate rounds.
// Stores all historical KV; no ring-state implementation is shared with the GPU.
std::vector<float> reference(const std::vector<float>& queries, const std::vector<float>& keys,
                             const std::vector<float>& sinks, int position, int prefill_width) {
    std::vector<float> q(queries.begin() + size_t(position) * qs, queries.begin() + size_t(position + 1) * qs);
    rope(q.data(), heads, position, false);
    std::array<int, 128> ids{};
    for (int i = 0; i < 128; ++i) {
        if (prefill_width) {
            const int id = std::max(0, position - 127) + i;
            ids[size_t(i)] = i < prefill_width && id <= position ? id : -1;
        } else {
            const int physical = (position + 1 + i) % 128;
            const int absolute = position - (position - physical + 128) % 128;
            ids[size_t(i)] = absolute >= 0 ? absolute : -1;
        }
    }
    std::vector<float> out(qs, 0.f);
    for (int h = 0; h < heads; ++h) {
        float maximum = -1e30f, denominator = 0.f;
        for (int chunk = 0; chunk < 2; ++chunk) {
            std::array<float, 64> scores{}, prob{};
            const float previous = maximum;
            for (int j = 0; j < 64; ++j) {
                const int id = ids[size_t(chunk * 64 + j)];
                if (id < 0) { scores[size_t(j)] = -INFINITY; continue; }
                double sum = 0.;
                for (int d = 0; d < dim; ++d) sum += double(q[size_t(h) * dim + d]) * keys[size_t(id) * dim + d];
                scores[size_t(j)] = float(sum) * float(1. / std::sqrt(512.));
                maximum = std::max(maximum, scores[size_t(j)]);
            }
            const float correction = std::exp(previous - maximum);
            float sum = 0.f;
            for (size_t j = 0; j < 64; ++j) { const float p = std::exp(scores[j] - maximum); sum += p; prob[j] = bf(p); }
            denominator = denominator * correction + sum;
            for (int d = 0; d < dim; ++d) {
                double value = 0.;
                for (int j = 0; j < 64; ++j) {
                    const int id = ids[size_t(chunk * 64 + j)];
                    if (id >= 0) value += double(prob[size_t(j)]) * keys[size_t(id) * dim + d];
                }
                auto& v = out[size_t(h) * dim + d]; v = v * correction + float(value);
            }
        }
        denominator += std::exp(sinks[size_t(h)] - maximum);
        for (int d = 0; d < dim; ++d) out[size_t(h) * dim + d] = bf(out[size_t(h) * dim + d] / denominator);
    }
    rope(out.data(), heads, position, true); return out;
}
void synthetic() {
    constexpr int count = 257;
    std::mt19937 gen(1017); std::uniform_real_distribution<float> uniform(-.7f, .7f);
    std::vector<float> q(size_t(count) * qs), kv(size_t(count) * dim), sinks(heads);
    for (auto& v : q) v = bf(uniform(gen));
    for (auto& v : kv) v = bf(uniform(gen));
    for (int h = 0; h < heads; ++h) sinks[size_t(h)] = float(h - 32) / 16.f;
    sinks[0] = 80.f; sinks[1] = -80.f;
    auto keys = kv;
    for (int t = 0; t < count; ++t) rope(keys.data() + size_t(t) * dim, 1, t, false);
    fp8(keys);
    Stream stream, other;
    CudaWindowAttention attention(sinks);
    core::DeviceArena arena((q.size() * 2 + kv.size()) * 4 + 1024);
    auto* dq = static_cast<float*>(arena.alloc(q.size() * 4));
    auto* dk = static_cast<float*>(arena.alloc(kv.size() * 4));
    auto* out = static_cast<float*>(arena.alloc(q.size() * 4));
    upload(dq, q, stream.value); upload(dk, kv, stream.value);
    const auto bytes = attention.device_bytes();
    rejects("null stream refused", [&] { attention.forward(dq, dk, out, 1, 0, nullptr); });
    rejects("overlapping output refused", [&] { attention.forward(dq, dk, dq, 1, 0, stream.value); });
    rejects("query/key overlap refused", [&] { attention.forward(dq, dq, out, 1, 0, stream.value); });
    rejects("missing prefill refused", [&] { attention.forward(dq, dk, out, 1, 1, stream.value); });
    rejects("zero tokens refused", [&] { attention.forward(dq, dk, out, 0, 0, stream.value); });
    rejects("sequence limit refused before access", [&] { attention.forward(dq, dk, out, 1048577, 0, stream.value); });
    attention.forward(dq, dk, out, 129, 0, stream.value);
    for (int t : {0, 1, 63, 64, 127, 128})
        compare(download(out + size_t(t) * qs, qs, stream.value), reference(q, keys, sinks, t, 128), "prefill position=" + std::to_string(t));
    const auto prefill64 = download(out + 64 * qs, qs, stream.value);
    check(attention.next_position() == 129, "prefill advances the sequence once complete");
    rejects("noncontiguous decode refused", [&] { attention.forward(dq, dk, out, 1, 130, stream.value); });
    rejects("second prefill refused", [&] { attention.forward(dq, dk, out, 1, 0, stream.value); });
    rejects("multi-token continuation refused", [&] { attention.forward(dq, dk, out, 2, 129, stream.value); });
    rejects("stream switch refused", [&] { attention.forward(dq, dk, out, 1, 129, other.value); });
    check(attention.next_position() == 129, "rejected calls preserve position");
    for (int t = 129; t < count; ++t) {
        attention.forward(dq + size_t(t) * qs, dk + size_t(t) * dim, out + size_t(t) * qs, 1, t, stream.value);
        if (t == 129 || t == 130 || t == 191 || t == 255 || t == 256)
            compare(download(out + size_t(t) * qs, qs, stream.value), reference(q, keys, sinks, t, 0), "decode position=" + std::to_string(t));
    }
    std::vector<float> expected_cache(128 * dim);
    for (int t = count - 128; t < count; ++t)
        std::copy_n(keys.data() + size_t(t) * dim, dim, expected_cache.data() + (t % 128) * dim);
    compare(cache(attention.kv_cache(), stream.value), expected_cache, "ring contains only the last 128 tokens at their physical slots");
    check(attention.device_bytes() == bytes && attention.next_position() == count, "owned memory stays constant across two wraparounds");
    attention.reset(stream.value);
    check(attention.next_position() == 0 && same(cache(attention.kv_cache(), stream.value), std::vector<float>(128 * dim, 0.f)), "reset clears cache and position");
    for (int t = 0; t <= 130; ++t) {
        attention.forward(dq + size_t(t) * qs, dk + size_t(t) * dim, out + size_t(t) * qs, 1, t, stream.value);
        if (t == 0 || t == 1 || t == 63 || t == 64 || t == 127 || t == 128 || t == 130)
            compare(download(out + size_t(t) * qs, qs, stream.value), reference(q, keys, sinks, t, t == 0 ? 1 : 0), "incremental position=" + std::to_string(t));
    }
    check(!same(prefill64, download(out + 64 * qs, qs, stream.value)), "fixture exercises different prefill/decode BF16 chunk grouping");
    const auto warm_cache = cache(attention.kv_cache(), stream.value);
    attention.reset(stream.value);
    attention.forward(dq, dk, out, 131, 0, stream.value);
    check(same(warm_cache, cache(attention.kv_cache(), stream.value)), "prefill/decode produce identical final KV ring bytes");
    rejects("wrong sink width refused", [&] { CudaWindowAttention bad{std::span(sinks).first(63)}; });
    sinks[0] = INFINITY;
    rejects("nonfinite sink refused", [&] { CudaWindowAttention bad(sinks); });
    rejects("missing projection weights refused", [&] { CudaAttention bad(AttentionWeights{}); });
    std::printf("swa_device_bytes=%zu cache_bytes=%d\n", bytes, 128 * dim * 2);
}
std::vector<float> tensor(const GgufModel& model, const std::string& name, size_t n) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != 0 || tensor_payload_bytes(*t) != n * 4 || !model.in_bounds(*t, shard))
        throw std::runtime_error("Missing/invalid fixture tensor " + name);
    const auto* data = reinterpret_cast<const float*>(model.shard(shard).tensor_data(*t)); return {data, data + n};
}
void real(const char* path) {
    const auto model = GgufModel::open(path);
    Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios = {0};
    const auto w = read_attention_weights(model, g, 0);
    CudaAttention attention(w); CudaWindowAttention core(w.sink);
    Stream stream;
    core::DeviceArena arena(3 * 5 * qs * 4);
    auto* input = static_cast<float*>(arena.alloc(5 * 5120 * 4));
    auto* output = static_cast<float*>(arena.alloc(5 * 5120 * 4));
    auto* query = static_cast<float*>(arena.alloc(5 * qs * 4));
    auto* kv = static_cast<float*>(arena.alloc(5 * dim * 4));
    auto* o = static_cast<float*>(arena.alloc(5 * qs * 4));
    const size_t bytes = attention.device_bytes();
    std::vector<std::vector<float>> results;
    for (int c = 0; c < 3; ++c) {
        const int tokens = c == 0 ? 5 : 1, pos = c == 0 ? 0 : c + 4;
        const auto prefix = "case." + std::to_string(c) + ".";
        upload(input, tensor(model, prefix + "input", size_t(tokens) * 5120), stream.value);
        upload(query, tensor(model, prefix + "query", size_t(tokens) * qs), stream.value);
        upload(kv, tensor(model, prefix + "kv", size_t(tokens) * dim), stream.value);
        core.forward(query, kv, o, tokens, pos, stream.value);
        compare(download(o, size_t(tokens) * qs, stream.value), tensor(model, prefix + "core", size_t(tokens) * qs), prefix + "isolated SWA official-graph oracle");
        compare(cache(core.kv_cache(), stream.value), tensor(model, prefix + "cache", 128 * dim), prefix + "isolated KV oracle");
        attention.forward(input, output, tokens, pos, stream.value);
        results.push_back(download(output, size_t(tokens) * 5120, stream.value));
        compare(results.back(), tensor(model, prefix + "output", size_t(tokens) * 5120), prefix + "full attention official-graph oracle", true);
        compare(cache(attention.kv_cache(), stream.value), tensor(model, prefix + "cache", 128 * dim), prefix + "projected KV oracle", true);
        check(attention.next_position() == pos + tokens, prefix + "full attention sequence position");
    }
    check(attention.device_bytes() == bytes, "full attention memory unchanged after prefill/decode");
    attention.reset(stream.value);
    for (int c = 0; c < 3; ++c) {
        const int tokens = c == 0 ? 5 : 1, pos = c == 0 ? 0 : c + 4;
        upload(input, tensor(model, "case." + std::to_string(c) + ".input", size_t(tokens) * 5120), stream.value);
        attention.forward(input, output, tokens, pos, stream.value);
        check(same(results[size_t(c)], download(output, size_t(tokens) * 5120, stream.value)), "reset/replay full attention is byte-identical");
    }
    attention.reset(stream.value);
    ck(cudaMemsetAsync(input, 0, 5120 * 4, stream.value)); attention.forward(input, output, 1, 0, stream.value);
    compare(download(output, 5120, stream.value), tensor(model, "zero.output", 5120), "zero input through norms and all projections", true);
    upload(query, tensor(model, "wide.input", 32768), stream.value);
    auto* quantized = reinterpret_cast<uint16_t*>(o);
    quantize_fp8_bf16(query, quantized, 32768, stream.value);
    std::vector<uint16_t> bits(32768);
    ck(cudaMemcpyAsync(bits.data(), quantized, bits.size() * 2, cudaMemcpyDeviceToHost, stream.value));
    ck(cudaStreamSynchronize(stream.value));
    size_t shard = 0; const auto* t = model.find("wide.quantized", &shard);
    check(t && t->type == 30 && tensor_payload_bytes(*t) == bits.size() * 2 && model.in_bounds(*t, shard) &&
          std::memcmp(bits.data(), model.shard(shard).tensor_data(*t), bits.size() * 2) == 0, "32768-value FP8 activation is exact against independent oracle");
    rejects("oversized FP8 activation refused", [&] { quantize_fp8_bf16(query, quantized, 32800, stream.value); });
    std::printf("attention_weight_bytes=%zu attention_device_bytes=%zu\n", attention.weight_bytes(), bytes);
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 3 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0)); synthetic(); if (argc == 3) real(argv[2]);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
