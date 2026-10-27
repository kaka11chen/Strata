#include "strata/deepseek41/compressed_attention_cuda.hpp"
#include "strata/deepseek41/compressed_kv_cuda.hpp"
#include "strata/deepseek41/index_key_cuda.hpp"
#include "strata/deepseek41/index_query_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "../../src/deepseek41/compressed_rope.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

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
template<class T> void upload(T* dst, const std::vector<T>& src, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
}
template<class T> std::vector<T> download(const T* src, size_t n, cudaStream_t stream) {
    std::vector<T> out(n); ck(cudaMemcpyAsync(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return out;
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double err = 0., norm = 0., peak = 0., maximum = 0.; bool finite = true, rounded = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i]; err += delta * delta; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]); rounded &= actual[i] == bf(actual[i]);
    }
    const double rel = std::sqrt(err / std::max(norm, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), rel, maximum, peak);
    check(finite && rounded && rel <= .003 && maximum <= 1e-5 + .02 * peak, name);
}
std::vector<float> expand(const std::vector<uint16_t>& x) {
    std::vector<float> y; y.reserve(x.size()); for (auto v : x) y.push_back(kernels::f32_from_bf16(v)); return y;
}
// With Q=0, p=1 exactly. KV has only channel 0 nonzero, avoiding any rotary
// arithmetic in this oracle. This detects lost/duplicated entries and sink use.
void closed_form(const float* out, int position, const std::vector<int>& groups,
                  const std::vector<float>& sink, cudaStream_t stream, const std::string& name) {
    const float windows = float(std::min(position + 1, 128));
    float numerator = windows;
    for (int id : groups) numerator += float(4 + id % 3);
    const auto actual = download(out, qs, stream);
    std::vector<float> expected(qs, 0.f);
    for (int h = 0; h < heads; ++h)
        expected[size_t(h) * dim] = bf(numerator / (windows + float(groups.size()) + std::exp(sink[size_t(h)])));
    check(actual == expected, name);
}
void synthetic() {
    Stream stream, other;
    std::vector<float> sink(heads, 0.f); sink[0] = 80.f; sink[1] = -80.f;
    CudaCompressedAttention attention(sink);
    constexpr int tokens = 1026, groups = tokens / 2;
    core::DeviceArena arena(size_t(tokens) * (2 * qs + dim) * 4 + groups * dim * 2 + 13 * 512 * 4 + 2048);
    auto* q = static_cast<float*>(arena.alloc(size_t(tokens) * qs * 4));
    auto* kv = static_cast<float*>(arena.alloc(size_t(tokens) * dim * 4));
    auto* out = static_cast<float*>(arena.alloc(size_t(tokens) * qs * 4));
    auto* compressed = static_cast<uint16_t*>(arena.alloc(groups * dim * 2));
    auto* ids = static_cast<int32_t*>(arena.alloc(13 * 512 * 4));
    ck(cudaMemsetAsync(q, 0, size_t(tokens) * qs * 4, stream.value));
    std::vector<float> host_kv(size_t(tokens) * dim, 0.f);
    for (int t = 0; t < tokens; ++t) host_kv[size_t(t) * dim] = 1.f;
    std::vector<uint16_t> host_compressed(groups * dim, 0);
    for (int g = 0; g < groups; ++g) host_compressed[size_t(g) * dim] = kernels::bf16_from_f32(float(4 + g % 3));
    upload(kv, host_kv, stream.value); upload(compressed, host_compressed, stream.value);
    const auto bytes = attention.device_bytes();
    std::vector<int32_t> selected(13 * 6, -1);
    for (int t = 0; t < 13; ++t)
        for (int j = 0; j < (t + 1) / 2; ++j) selected[size_t(t) * 6 + j] = j + 13;
    upload(ids, selected, stream.value);
    const CompressedAttentionSelection selection{compressed, 6, ids, 6, 13};
    const std::vector<float> sentinel(13 * qs, 17.f);
    upload(out, sentinel, stream.value);
    const auto empty_ring = download(attention.kv_cache(), 128 * dim, stream.value);
    auto call = [&](CompressedAttentionSelection s) { attention.forward(q, kv, out, 13, 0, s, stream.value); };
    rejects("null stream", [&] { attention.forward(q, kv, out, 13, 0, selection, nullptr); });
    rejects("zero tokens", [&] { attention.forward(q, kv, out, 0, 0, selection, stream.value); });
    rejects("context overflow", [&] { attention.forward(q, kv, out, 1048577, 0, selection, stream.value); });
    rejects("missing initial prefill", [&] { attention.forward(q, kv, out, 1, 1, selection, stream.value); });
    rejects("wrong group count", [&] { auto s = selection; --s.groups; call(s); });
    rejects("negative group count", [&] { auto s = selection; s.groups = -1; call(s); });
    rejects("wrong concatenation offset", [&] { auto s = selection; --s.offset; call(s); });
    rejects("oversized selection", [&] { auto s = selection; s.count = 513; call(s); });
    rejects("negative selection count", [&] { auto s = selection; s.count = -1; call(s); });
    rejects("null compressed storage", [&] { auto s = selection; s.kv = nullptr; call(s); });
    rejects("null indices", [&] { auto s = selection; s.indices = nullptr; call(s); });
    rejects("unaligned indices", [&] { auto s = selection; s.indices = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(ids) + 1); call(s); });
    rejects("input/output overlap", [&] { attention.forward(q, kv, q, 13, 0, selection, stream.value); });
    rejects("query/window overlap", [&] { attention.forward(q, q, out, 13, 0, selection, stream.value); });
    rejects("owned ring alias", [&] { auto s = selection; s.kv = attention.kv_cache(); call(s); });
    rejects("pointer range overflow", [&] { attention.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), kv, out, 13, 0, selection, stream.value); });
    std::vector<std::pair<std::string, std::vector<int32_t>>> malformed;
    auto bad = selected; bad[0] = 13; malformed.push_back({"future group on first row", bad});
    bad = selected; bad.back() = 19; malformed.push_back({"out-of-range group on last row", bad});
    bad = selected; bad.back() = -2; malformed.push_back({"negative index other than -1", bad});
    bad = selected; bad.back() = 17; malformed.push_back({"duplicate group", bad});
    bad = selected; std::swap(bad[72], bad[73]); malformed.push_back({"unordered group IDs", bad});
    bad = selected; bad[72] = -1; malformed.push_back({"valid ID after padding", bad});
    bad = selected; bad[72] = 0; malformed.push_back({"unshifted group ID", bad});
    for (const auto& [name, values] : malformed) {
        upload(ids, values, other.value);
        rejects(name, [&] { attention.forward(q, kv, out, 13, 0, selection, other.value); });
        check(attention.next_position() == 0 && same(download(out, sentinel.size(), other.value), sentinel) &&
              same(download(attention.kv_cache(), 128 * dim, other.value), empty_ring), name + " preserves entire batch state");
        check(same(download(ids, values.size(), other.value), values), name + " leaves caller indices intact");
    }
    upload(ids, selected, stream.value);
    call(selection); // rejected device indices above must not bind the other stream
    for (int t = 0; t < 13; ++t) {
        std::vector<int> active; for (int j = 0; j < (t + 1) / 2; ++j) active.push_back(j);
        closed_form(out + size_t(t) * qs, t, active, sink, stream.value, "joint prefill numerator/denominator token=" + std::to_string(t));
    }
    const auto good = download(out, 13 * qs, stream.value);
    const auto ring = download(attention.kv_cache(), 128 * dim, stream.value);
    rejects("noncontiguous decode", [&] { attention.forward(q, kv, out, 1, 14, {compressed, 7, ids, 0, 128}, stream.value); });
    rejects("multi-token continuation", [&] { attention.forward(q, kv, out, 2, 13, {compressed, 7, ids, 0, 128}, stream.value); });
    rejects("stream switch", [&] { attention.forward(q, kv, out, 1, 13, {compressed, 7, ids, 0, 128}, other.value); });
    rejects("reset stream switch", [&] { attention.reset(other.value); });
    bad = {135}; upload(ids, bad, stream.value);
    rejects("future decode group", [&] { attention.forward(q, kv, out, 1, 13, {compressed, 7, ids, 1, 128}, stream.value); });
    check(attention.next_position() == 13 && same(good, download(out, 13 * qs, stream.value)) &&
          same(ring, download(attention.kv_cache(), 128 * dim, stream.value)), "bad decode preserves output/history/position");
    for (int pos = 13; pos <= 130; ++pos) {
        const int count = (pos + 1) / 2;
        std::vector<int> active; selected.assign(size_t(count), -1);
        for (int j = 0; j < count; j += 2) { selected[active.size()] = j + 128; active.push_back(j); }
        upload(ids, selected, stream.value);
        attention.forward(q, kv, out, 1, pos, {compressed, count, ids, count, 128}, stream.value);
        if (pos == 13 || pos == 63 || pos == 64 || pos >= 127)
            closed_form(out, pos, active, sink, stream.value, "sparse decode/wrap token=" + std::to_string(pos));
    }
    attention.reset(stream.value);
    check(attention.next_position() == 0 && same(empty_ring, download(attention.kv_cache(), 128 * dim, stream.value)), "reset clears ring and position");
    // Build history without selected compressed entries, then exercise all ten
    // 64-candidate chunks with a 512-item sparse subset of 513 stored groups.
    attention.forward(q, kv, out, 1024, 0, {compressed, 512, nullptr, 0, 1024}, stream.value);
    for (int pos = 1024; pos <= 1025; ++pos) {
        std::vector<int> active(512); selected.resize(512);
        for (int j = 0; j < 512; ++j) active[size_t(j)] = j;
        if (pos == 1025) active.back() = 512;
        for (int j = 0; j < 512; ++j) selected[size_t(j)] = active[size_t(j)] + 128;
        upload(ids, selected, stream.value);
        attention.forward(q, kv, out, 1, pos, {compressed, (pos + 1) / 2, ids, 512, 128}, stream.value);
        closed_form(out, pos, active, sink, stream.value, "full 512 selections token=" + std::to_string(pos));
    }
    check(attention.device_bytes() == bytes && attention.next_position() == tokens, "bounded owned memory through 512 selections");
    check(same(download(compressed, host_compressed.size(), stream.value), host_compressed) &&
          same(download(kv, host_kv.size(), stream.value), host_kv) &&
          download(q, size_t(tokens) * qs, stream.value) == std::vector<float>(size_t(tokens) * qs, 0.f), "caller Q/KV/cache unchanged");
    attention.reset(stream.value);
    // Two window entries and one compressed entry belong to ONE 64-slot chunk.
    // Channel 1 cancels exactly after BF16 p rounding in that joint chunk. If
    // the window is separately padded to 128, its old numerator is rescaled in
    // F32 and the cancellation fails, even if a loose global error gate passes.
    std::vector<float> mixed_q(2 * qs, 0.f), mixed_kv(2 * dim, 0.f);
    std::vector<uint16_t> mixed_compressed(dim, 0);
    for (int h = 0; h < 2 * heads; ++h) mixed_q[size_t(h) * dim] = 32.f;
    mixed_kv[0] = mixed_kv[1] = mixed_kv[dim] = mixed_kv[dim + 1] = 1.f;
    mixed_compressed[0] = kernels::bf16_from_f32(2.f);
    mixed_compressed[1] = kernels::bf16_from_f32(-2.f * bf(std::exp(-32.f * float(1. / std::sqrt(512.)))));
    upload(q, mixed_q, stream.value); upload(kv, mixed_kv, stream.value);
    upload(compressed, mixed_compressed, stream.value); upload(ids, std::vector<int32_t>{-1, 2}, stream.value);
    attention.forward(q, kv, out, 2, 0, {compressed, 1, ids, 1, 2}, stream.value);
    const auto mixed = download(out, 2 * qs, stream.value);
    bool cancellation = true;
    for (int h = 0; h < heads; ++h) cancellation &= mixed[qs + size_t(h) * dim + 1] == 0.f;
    check(mixed[size_t(2) * dim + 1] > 0.f && cancellation, "short prefill joins both sources before 64-candidate BF16 rounding");
    rejects("wrong sink width", [&] { CudaCompressedAttention bad_sink{std::span(sink).first(63)}; });
    sink[0] = INFINITY; rejects("nonfinite sink", [&] { CudaCompressedAttention bad_sink(sink); });
    std::printf("compressed_attention_device_bytes=%zu ring_bytes=%d\n", bytes, 128 * dim * 2);
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, size_t n) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != (sizeof(T) == 2 ? 30u : 0u) || tensor_payload_bytes(*t) != n * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Missing/invalid fixture tensor " + name);
    const auto* data = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*t)); return {data, data + n};
}
int number(const GgufModel& m, const std::string& name, int maximum) {
    const auto* v = m.meta().get(name);
    if (!v || v->type != MetaType::U32 || v->u > uint64_t(maximum)) throw std::runtime_error("Invalid fixture number " + name);
    return int(v->u);
}
std::vector<int32_t> indices(const GgufModel& m, const std::string& name, size_t n) {
    const auto src = tensor<float>(m, name, n); std::vector<int32_t> result;
    for (float v : src) {
        if (!std::isfinite(v) || v < -1 || v > 1048576 || v != std::floor(v)) throw std::runtime_error("Invalid fixture index");
        result.push_back(int32_t(v));
    }
    return result;
}
void real(const char* path) {
    const auto model = GgufModel::open(path);
    Stream stream;
    CudaCompressedAttention attention(tensor<float>(model, "blk.2.attn_sink", heads));
    core::DeviceArena arena(40 * 1024 * 1024);
    auto* q = static_cast<float*>(arena.alloc(131 * qs * 4));
    auto* kv = static_cast<float*>(arena.alloc(131 * dim * 4));
    auto* out = static_cast<float*>(arena.alloc(131 * qs * 4));
    auto* compressed = static_cast<uint16_t*>(arena.alloc(65 * dim * 2));
    auto* ids = static_cast<int32_t*>(arena.alloc(131 * 65 * 4));
    const auto host_q = expand(tensor<uint16_t>(model, "sequence.q", 131 * qs));
    const auto host_kv = expand(tensor<uint16_t>(model, "sequence.kv", 131 * dim));
    upload(q, host_q, stream.value); upload(kv, host_kv, stream.value);
    const auto host_compressed = tensor<uint16_t>(model, "sequence.compressed", 65 * dim);
    upload(compressed, host_compressed, stream.value);
    const int plans = number(model, "fixture.plans", 8);
    check(plans == 6, "all short-prefill/chunk-boundary/wrap plans present");
    for (int plan = 0; plan < plans; ++plan) {
        attention.reset(stream.value);
        const int first = number(model, "plan." + std::to_string(plan) + ".prefill", 131);
        for (int pos = 0; pos < 131;) {
            const int length = pos == 0 ? first : 1, groups = (pos + length) / 2;
            const auto prefix = "plan." + std::to_string(plan) + ".at." + std::to_string(pos) + ".";
            if (groups) upload(ids, indices(model, prefix + "indices", size_t(length) * groups), stream.value);
            attention.forward(q + size_t(pos) * qs, kv + size_t(pos) * dim, out, length, pos,
                              {compressed, groups, ids, groups, pos == 0 ? length : 128}, stream.value);
            if (model.find(prefix + "output"))
                compare(download(out, size_t(length) * qs, stream.value), expand(tensor<uint16_t>(model, prefix + "output", size_t(length) * qs)), prefix + "core");
            pos += length;
        }
        check(same(download(attention.kv_cache(), 128 * dim, stream.value), tensor<uint16_t>(model, "plan." + std::to_string(plan) + ".ring", 128 * dim)), "plan " + std::to_string(plan) + " final window ring byte exact");
        check(attention.next_position() == 131, "plan completes correct position");
    }
    check(same(download(q, host_q.size(), stream.value), host_q) && same(download(kv, host_kv.size(), stream.value), host_kv) &&
          same(download(compressed, host_compressed.size(), stream.value), host_compressed), "reference plan caller inputs untouched");
    const auto rope_input = expand(tensor<uint16_t>(model, "rope.input", qs));
    const int rope_cases = number(model, "fixture.rope_cases", 8);
    check(rope_cases == 6, "all compressed main-head RoPE positions present");
    for (int c = 0; c < rope_cases; ++c) {
        const auto prefix = "rope." + std::to_string(c) + ".";
        const int pos = number(model, prefix + "position", 1048575);
        for (bool inverse : {false, true}) {
            upload(out, rope_input, stream.value);
            detail::rotate_compressed(out, heads, pos, inverse, stream.value);
            const auto actual = download(out, qs, stream.value);
            const auto expected = expand(tensor<uint16_t>(model, prefix + (inverse ? "inverse" : "forward"), qs));
            size_t different = 0; int max_ulp = 0; bool valid = true;
            for (size_t i = 0; i < actual.size(); ++i) {
                const int a = int(kernels::bf16_from_f32(actual[i]));
                const int b = int(kernels::bf16_from_f32(expected[i]));
                const int ulp = actual[i] == expected[i] ? 0 : std::abs(a - b);
                max_ulp = std::max(max_ulp, ulp);
                // libdevice/CPU sin can differ by an FP32 ULP; cancellation can
                // expose that at a BF16 boundary. Unrotated channels stay exact.
                valid &= std::isfinite(actual[i]) && actual[i] == bf(actual[i]) &&
                         ulp <= (i % dim >= 448 ? 1 : 0);
                if (actual[i] != expected[i]) {
                    if (different < 8) {
                        const size_t at = i / 2 * 2;
                        std::printf("  RoPE mismatch pos=%d inverse=%d at=%zu pair=%zu a=%a b=%a actual=%a expected=%a\n",
                            pos, int(inverse), i, (i % dim - 448) / 2, double(rope_input[at]), double(rope_input[at + 1]),
                            double(actual[i]), double(expected[i]));
                    }
                    ++different;
                }
            }
            std::printf("  %s%s different=%zu/%d max_bf16_ulp=%d\n", prefix.c_str(), inverse ? "inverse" : "forward", different, qs, max_ulp);
            check(valid, prefix + (inverse ? "inverse" : "forward") + " exact prefix, rotary <=1 BF16 ULP");
        }
    }
    Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
    CudaCompressor compressor(read_compressor_weights(model, g, 2));
    CudaCompressedKvCache cache(8);
    CudaIndexKeyCache keys(read_index_key_weights(model, g, 2), 8);
    CudaIndexQuery indexer(read_index_query_weights(model, g, 2), 8);
    auto* x = static_cast<float*>(arena.alloc(15 * 5120 * 4));
    auto* qr = static_cast<float*>(arena.alloc(15 * 1280 * 4));
    auto* latent = static_cast<float*>(arena.alloc(7 * dim * 4));
    const auto host_x = expand(tensor<uint16_t>(model, "integration.x", 15 * 5120));
    const auto host_qr = expand(tensor<uint16_t>(model, "integration.qr", 15 * 1280));
    upload(x, host_x, stream.value); upload(qr, host_qr, stream.value);
    attention.reset(stream.value);
    for (int pos : {0, 13, 14}) {
        const int length = pos == 0 ? 13 : 1, offset = pos == 0 ? 13 : 128;
        const auto prefix = "integration.at." + std::to_string(pos) + ".";
        const auto batch = compressor.forward(x + pos * 5120, latent, length, pos, stream.value);
        if (batch.count)
            compare(download(latent, size_t(batch.count) * dim, stream.value), expand(tensor<uint16_t>(model, prefix + "latent", size_t(batch.count) * dim)), prefix + "original latent");
        keys.append(latent, batch, stream.value); cache.append(latent, batch, stream.value);
        const int groups = cache.size();
        check(groups == (pos + length) / 2 && keys.size() == groups, prefix + "cache counts synchronized");
        check(same(download(cache.data(), size_t(groups) * dim, stream.value), tensor<uint16_t>(model, prefix + "compressed", size_t(groups) * dim)), prefix + "compressed KV byte exact");
        check(same(download(keys.data(), size_t(groups) * 128, stream.value), tensor<uint16_t>(model, prefix + "keys", size_t(groups) * 128)), prefix + "index keys byte exact");
        for (int t = 0; t < length; ++t) {
            const auto chosen = indexer.forward(x + (pos + t) * 5120, qr + (pos + t) * 1280, keys.data(), groups, pos + t, offset, stream.value);
            ck(cudaMemcpyAsync(ids + t * groups, chosen.indices, size_t(chosen.count) * 4, cudaMemcpyDeviceToDevice, stream.value));
        }
        check(same(download(ids, size_t(length) * groups, stream.value), indices(model, prefix + "indices", size_t(length) * groups)), prefix + "indexer IDs exact");
        const auto saved_cache = download(cache.data(), size_t(groups) * dim, stream.value);
        attention.forward(q + size_t(pos) * qs, kv + size_t(pos) * dim, out, length, pos,
                          {cache.data(), groups, ids, groups, offset}, stream.value);
        compare(download(out, size_t(length) * qs, stream.value), expand(tensor<uint16_t>(model, prefix + "output", size_t(length) * qs)), prefix + "6a through 6e");
        check(same(download(attention.kv_cache(), 128 * dim, stream.value), tensor<uint16_t>(model, prefix + "ring", 128 * dim)), prefix + "window ring byte exact");
        check(same(saved_cache, download(cache.data(), size_t(groups) * dim, stream.value)), prefix + "attention leaves compressed cache unchanged");
        check(attention.next_position() == compressor.next_position(), prefix + "positions synchronized");
    }
    check(same(download(x, host_x.size(), stream.value), host_x) && same(download(qr, host_qr.size(), stream.value), host_qr), "integration source inputs unchanged");
    std::printf("compressed_attention_device_bytes=%zu\n", attention.device_bytes());
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 3 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 3) real(argv[2]); else synthetic();
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
