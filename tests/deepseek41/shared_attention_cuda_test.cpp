#include "strata/deepseek41/layer3_attention_cuda.hpp"
#include "strata/deepseek41/attention_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr size_t hidden = 5120;
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
template<class T> std::vector<T> download(const T* p, size_t n, cudaStream_t s) {
    std::vector<T> v(n); if (n) ck(cudaMemcpyAsync(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, s));
    ck(cudaStreamSynchronize(s)); return v;
}
template<class T> void upload(T* p, const std::vector<T>& v, cudaStream_t s) {
    ck(cudaMemcpyAsync(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, s));
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}
template<class T> bool zero(const std::vector<T>& v) { return std::all_of(v.begin(), v.end(), [](T x) { return x == 0; }); }
std::vector<float> expand(const std::vector<uint16_t>& v) {
    std::vector<float> out; out.reserve(v.size()); for (auto x : v) out.push_back(kernels::f32_from_bf16(x)); return out;
}
struct History {
    std::vector<uint16_t> ring, compressed, keys;
    int next, groups;
    bool operator==(const History& b) const {
        return same(ring, b.ring) && same(compressed, b.compressed) && same(keys, b.keys) && next == b.next && groups == b.groups;
    }
};
History history(const CudaLayer2Attention& a, cudaStream_t s) {
    return {download(a.kv_cache(), 128 * 512, s), download(a.compressed_kv_cache(), size_t(a.capacity()) * 512, s),
            download(a.index_key_cache(), size_t(a.capacity()) * 128, s), a.next_position(), a.compressed_groups()};
}
struct Weights {
    std::array<std::vector<uint8_t>, 5> values, scales;
    std::vector<uint16_t> qnorm, kvnorm, compkv, gate, compnorm, key, keynorm, head_weights;
    std::vector<uint8_t> query, query_scales;
    std::vector<float> sink;
    AttentionWeights attention;
    Weights() : qnorm(1280, 0x3f80), kvnorm(512, 0x3f80), compkv(512 * hidden, 0), gate(compkv.size(), 0),
                compnorm(512, 0x3f80), key(128 * 512, 0), keynorm(128, 0x3f80), head_weights(32 * hidden, 0),
                query(4096 * 1280, 0), query_scales(query.size() / 1024, 127), sink(64, 0.f) {
        const std::array<int, 5> rows{1280, 32768, 512, 8192, 5120}, cols{5120, 1280, 5120, 4096, 8192};
        for (size_t i = 0; i < 5; ++i) {
            values[i].assign(size_t(rows[i]) * cols[i], 0); scales[i].assign(values[i].size() / 1024, 127);
            attention.projections[i] = {values[i], scales[i], rows[i], cols[i]};
        }
        values[0][0] = values[2][0] = values[4][0] = 0x38;
        for (int h = 0; h < 64; ++h) values[1][size_t(h) * 512 * 1280] = 0x38;
        for (int g = 0; g < 8; ++g) values[3][size_t(g) * 1024 * 4096] = 0x38;
        compkv[0] = key[0] = 0x3f80;
        attention.q_norm = qnorm; attention.kv_norm = kvnorm; attention.sink = sink; attention.layer = 2;
    }
    CompressorWeights compressor() const { return {compkv, gate, compnorm}; }
    IndexKeyWeights index_key() const { return {key, keynorm}; }
    IndexQueryWeights index_query() const { return {{query, query_scales, 4096, 1280}, head_weights}; }
};
void synthetic() {
    Weights w; auto w3 = w.attention; w3.layer = 3;
    rejects("consumer refuses source weights", [&] { CudaLayer3Attention bad(w.attention); });
    rejects("pure-SWA refuses consumer weights", [&] { CudaAttention bad(w3); });
    for (int limit : {-1, 1048577})
        rejects("invalid shared prefill capacity", [&] { CudaLayer2Attention bad(w.attention, {}, {}, {}, 4, 0, limit); });
    Stream s, other;
    core::DeviceArena buffers(10 * hidden * 16 + 1024);
    auto* x2 = static_cast<float*>(buffers.alloc(10 * hidden * 4));
    auto* x3 = static_cast<float*>(buffers.alloc(10 * hidden * 4));
    auto* y2 = static_cast<float*>(buffers.alloc(10 * hidden * 4));
    auto* y3 = static_cast<float*>(buffers.alloc(10 * hidden * 4));
    std::vector<float> input2(10 * hidden, 0.f), input3(input2), sentinel(input2.size(), -37.f);
    for (int t = 0; t < 10; ++t) {
        input2[size_t(t) * hidden] = float((t % 3 ? 1 : -1) * (t + 1));
        input3[size_t(t) * hidden] = float((t % 2 ? 1 : -1) * (t + 2));
    }
    upload(x2, input2, s.value); upload(x3, input3, s.value);
    upload(y2, sentinel, s.value); upload(y3, sentinel, s.value);
    size_t standalone_bytes = 0;
    {
        CudaLayer2Attention disabled(w.attention, w.compressor(), w.index_key(), w.index_query(), 4);
        standalone_bytes = disabled.device_bytes();
        disabled.forward(x2, y2, 1, 0, s.value);
        rejects("publication disabled unless explicitly reserved", [&] { disabled.shared_batch(); });
    }
    CudaLayer2Attention source(w.attention, w.compressor(), w.index_key(), w.index_query(), 4, 0, 3);
    CudaLayer3Attention consumer(w3);
    const auto source_bytes = source.device_bytes(), consumer_bytes = consumer.device_bytes();
    check(source_bytes == standalone_bytes + 256, "shared index allocation is explicit and aligned at startup");
    upload(y2, sentinel, s.value);
    const auto empty = history(source, s.value);
    const auto empty_ring = download(consumer.kv_cache(), 128 * 512, s.value);
    rejects("unpublished source", [&] { source.shared_batch(); });
    rejects("empty publication handle", [&] { consumer.forward(x3, y3, 3, 0, {}, s.value); });
    rejects("shared prefill limit checked before source advances", [&] { source.forward(x2, y2, 4, 0, other.value); });
    check(history(source, s.value) == empty && same(download(y2, sentinel.size(), s.value), sentinel), "rejected publication leaves all source state/output unchanged");
    source.forward(x2, y2, 3, 0, s.value);
    auto first = source.shared_batch(), first_copy = first;
    const auto source_first = history(source, s.value);
    rejects("consumer mismatched batch length", [&] { consumer.forward(x3, y3, 2, 0, first, s.value); });
    rejects("consumer null stream", [&] { consumer.forward(x3, y3, 3, 0, first, nullptr); });
    rejects("consumer different source stream", [&] { consumer.forward(x3, y3, 3, 0, first, other.value); });
    rejects("consumer null input", [&] { consumer.forward(nullptr, y3, 3, 0, first, s.value); });
    rejects("consumer null output", [&] { consumer.forward(x3, nullptr, 3, 0, first, s.value); });
    rejects("consumer input/output overlap", [&] { consumer.forward(x3, x3, 3, 0, first, s.value); });
    rejects("consumer unaligned input", [&] { consumer.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(x3) + 1), y3, 3, 0, first, s.value); });
    rejects("consumer pointer overflow", [&] { consumer.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), y3, 3, 0, first, s.value); });
    rejects("consumer context overflow", [&] { consumer.forward(x3, y3, 1048577, 0, first, s.value); });
    rejects("consumer zero tokens", [&] { consumer.forward(x3, y3, 0, 0, first, s.value); });
    rejects("consumer missing initial prefill", [&] { consumer.forward(x3, y3, 1, 1, first, s.value); });
    for (const auto* view : {source.kv_cache(), source.compressed_kv_cache(), source.index_key_cache(), consumer.kv_cache()})
        rejects("consumer cannot write either layer's cache", [&] {
            consumer.forward(x3, reinterpret_cast<float*>(const_cast<uint16_t*>(view)), 3, 0, first, s.value);
        });
    check(consumer.next_position() == 0 && same(download(consumer.kv_cache(), 128 * 512, s.value), empty_ring) &&
          same(download(y3, sentinel.size(), s.value), sentinel) && history(source, s.value) == source_first,
          "consumer argument rejections preserve both histories and output");
    consumer.forward(x3, y3, 3, 0, first_copy, s.value);
    const auto first_output = download(y3, 3 * hidden, s.value);
    const auto own_ring = download(consumer.kv_cache(), 128 * 512, s.value);
    check(!zero(first_output) && !same(own_ring, source_first.ring) && history(source, s.value) == source_first,
          "consumer has its own window/output and leaves source caches unchanged");
    rejects("duplicate consumption refused", [&] { consumer.forward(x3, y3, 3, 0, first, s.value); });
    rejects("consumer reset stream switch", [&] { consumer.reset(other.value); });
    consumer.reset(s.value);
    check(history(source, s.value) == source_first && zero(download(consumer.kv_cache(), 128 * 512, s.value)), "consumer reset touches only its own ring/binding");
    consumer.forward(x3, y3, 3, 0, first_copy, s.value);
    check(same(download(y3, 3 * hidden, s.value), first_output), "source batch remains valid after consumer reset/replay");
    source.forward(x2 + 3 * hidden, y2 + 3 * hidden, 1, 3, s.value);
    auto decode3 = source.shared_batch();
    source.forward(x2 + 4 * hidden, y2 + 4 * hidden, 1, 4, s.value);
    rejects("publication expires on source advance", [&] { consumer.forward(x3 + 3 * hidden, y3 + 3 * hidden, 1, 3, decode3, s.value); });
    rejects("source ahead of consumer refused", [&] { consumer.forward(x3 + 3 * hidden, y3 + 3 * hidden, 1, 3, source.shared_batch(), s.value); });
    source.reset(s.value);
    rejects("source reset clears publication", [&] { source.shared_batch(); });
    source.forward(x2, y2, 3, 0, s.value);
    source.forward(x2 + 3 * hidden, y2 + 3 * hidden, 1, 3, s.value);
    const auto before_epoch_reject = history(source, s.value);
    rejects("same position from a new source sequence refused", [&] { consumer.forward(x3 + 3 * hidden, y3 + 3 * hidden, 1, 3, source.shared_batch(), s.value); });
    rejects("same-position old publication still refused", [&] { consumer.forward(x3 + 3 * hidden, y3 + 3 * hidden, 1, 3, decode3, s.value); });
    check(consumer.next_position() == 3 && same(download(consumer.kv_cache(), 128 * 512, s.value), own_ring) &&
          same(download(y3, 3 * hidden, s.value), first_output) && history(source, s.value) == before_epoch_reject,
          "stale/sequence rejection changes neither layer");
    source.reset(s.value); consumer.reset(s.value);
    source.forward(x2, y2, 3, 0, s.value); consumer.forward(x3, y3, 3, 0, source.shared_batch(), s.value);
    {
        auto second = std::make_unique<CudaLayer2Attention>(w.attention, w.compressor(), w.index_key(), w.index_query(), 4, 0, 3);
        second->forward(x2, y2, 3, 0, s.value); second->forward(x2 + 3 * hidden, y2 + 3 * hidden, 1, 3, s.value);
        rejects("same-position different source refused", [&] { consumer.forward(x3 + 3 * hidden, y3 + 3 * hidden, 1, 3, second->shared_batch(), s.value); });
        second->reset(s.value); second->forward(x2, y2, 3, 0, s.value);
        auto expired = second->shared_batch(); second.reset(); consumer.reset(s.value);
        rejects("destroyed source rejected without dereferencing GPU storage", [&] { consumer.forward(x3, y3, 3, 0, expired, s.value); });
    }
    source.reset(s.value); consumer.reset(s.value);
    source.forward(x2, y2, 3, 0, s.value); consumer.forward(x3, y3, 3, 0, source.shared_batch(), s.value);
    for (int pos = 3; pos < 9; ++pos) {
        source.forward(x2 + size_t(pos) * hidden, y2 + size_t(pos) * hidden, 1, pos, s.value);
        const auto batch = source.shared_batch(); const auto before = history(source, s.value);
        if (pos == 8) {
            rejects("source cache capacity rejection preserves publication", [&] { source.forward(x2 + 9 * hidden, y2 + 9 * hidden, 1, 9, s.value); });
            rejects("source rejected reset preserves publication", [&] { source.reset(other.value); });
        }
        consumer.forward(x3 + size_t(pos) * hidden, y3 + size_t(pos) * hidden, 1, pos, batch, s.value);
        check(history(source, s.value) == before, "decode consumer leaves shared histories unchanged at " + std::to_string(pos));
    }
    check(source.next_position() == 9 && consumer.next_position() == 9 && source.compressed_groups() == 4 &&
          source.device_bytes() == source_bytes && consumer.device_bytes() == consumer_bytes, "paired positions and startup allocation stay bounded through full-cache pending token");
    check(same(download(x2, input2.size(), s.value), input2) && same(download(x3, input3.size(), s.value), input3), "both caller inputs unchanged");
    source.reset(s.value); consumer.reset(s.value);
    check(history(source, s.value) == empty && same(download(consumer.kv_cache(), 128 * 512, s.value), empty_ring), "paired reset clears both layers");
    rejects("old first publication invalid after same-position reset", [&] { consumer.forward(x3, y3, 3, 0, first_copy, s.value); });
    std::printf("shared_source_device_bytes=%zu consumer_device_bytes=%zu index_reservation_bytes=%zu\n", source_bytes, consumer_bytes, source_bytes - standalone_bytes);
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, size_t n) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != (sizeof(T) == 2 ? 30u : 0u) || tensor_payload_bytes(*t) != n * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Missing/invalid fixture tensor " + name);
    const auto* p = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*t)); return {p, p + n};
}
int number(const GgufModel& m, const std::string& name, int maximum) {
    const auto* v = m.meta().get(name);
    if (!v || v->type != MetaType::U32 || v->u > uint64_t(maximum)) throw std::runtime_error("Invalid fixture number " + name);
    return int(v->u);
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name, bool full) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double err = 0., norm = 0., peak = 0., maximum = 0.; bool finite = true, rounded = true;
    size_t differences = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = double(actual[i]) - expected[i]; err += d * d; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(d)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        rounded &= actual[i] == kernels::f32_from_bf16(kernels::bf16_from_f32(actual[i]));
        differences += actual[i] != expected[i];
    }
    const double rel = std::sqrt(err / std::max(norm, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g different=%zu/%zu\n", name.c_str(), rel, maximum, peak, differences, actual.size());
    check(finite && rounded && rel <= (full ? .005 : .003) && maximum <= 1e-5 + (full ? .03 : .02) * peak, name);
}
void real(const char* source_path, const char* path) {
    const auto source_model = GgufModel::open(source_path), m = GgufModel::open(path);
    Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios.assign(40, 0); g.compress_ratios[2] = g.compress_ratios[3] = 2;
    g.sources.resize(40); g.sources[2] = g.sources[3] = {2, 2, -1};
    CudaLayer2Attention source(read_attention_weights(source_model, g, 2), read_compressor_weights(source_model, g, 2),
                              read_index_key_weights(source_model, g, 2), read_index_query_weights(source_model, g, 2), 66, 0, 129);
    CudaLayer3Attention consumer(read_attention_weights(m, g, 3));
    Stream s; core::DeviceArena buffers(131 * hidden * 16 + 1024);
    auto* x2 = static_cast<float*>(buffers.alloc(131 * hidden * 4));
    auto* x3 = static_cast<float*>(buffers.alloc(131 * hidden * 4));
    auto* y2 = static_cast<float*>(buffers.alloc(131 * hidden * 4));
    auto* y3 = static_cast<float*>(buffers.alloc(131 * hidden * 4));
    const auto input2 = expand(tensor<uint16_t>(m, "sequence.input", 131 * hidden));
    const auto input3 = expand(tensor<uint16_t>(m, "consumer.input", 131 * hidden));
    upload(x2, input2, s.value); upload(x3, input3, s.value);
    const auto source_bytes = source.device_bytes(), consumer_bytes = consumer.device_bytes();
    History baseline; std::vector<uint16_t> baseline_ring; std::vector<float> replay_source, replay_consumer;
    const int plans = number(m, "fixture.plans", 3);
    if (plans != 3) throw std::runtime_error("Incomplete shared fixture plans");
    for (int plan = 0; plan < plans; ++plan) {
        const std::string prefix = "plan." + std::to_string(plan) + ".";
        const int first = number(m, prefix + "prefill", 129), end = number(m, prefix + "tokens", 131);
        if (first <= 0 || first > end) throw std::runtime_error("Invalid fixture plan");
        source.reset(s.value); consumer.reset(s.value);
        for (int pos = 0; pos < end;) {
            const int tokens = pos ? 1 : first, groups = (pos + tokens) / 2;
            const size_t offset = size_t(pos) * hidden, n = size_t(tokens) * hidden;
            source.forward(x2 + offset, y2 + offset, tokens, pos, s.value);
            const auto batch = source.shared_batch(), copy = batch;
            const auto before = history(source, s.value);
            consumer.forward(x3 + offset, y3 + offset, tokens, pos, copy, s.value);
            const auto p = prefix + "at." + std::to_string(pos) + ".";
            compare(download(y2 + offset, n, s.value), expand(tensor<uint16_t>(m, p + "output", n)), p + "source_output", true);
            compare(download(y3 + offset, n, s.value), expand(tensor<uint16_t>(m, p + "consumer_output", n)), p + "consumer_output", true);
            compare(expand(before.ring), expand(tensor<uint16_t>(m, p + "ring", 128 * 512)), p + "source_ring", true);
            compare(expand(download(consumer.kv_cache(), 128 * 512, s.value)), expand(tensor<uint16_t>(m, p + "consumer_ring", 128 * 512)), p + "consumer_ring", true);
            if (groups) {
                compare(expand(download(source.compressed_kv_cache(), size_t(groups) * 512, s.value)),
                        expand(tensor<uint16_t>(m, p + "compressed", size_t(groups) * 512)), p + "shared_kv", false);
                compare(expand(download(source.index_key_cache(), size_t(groups) * 128, s.value)),
                        expand(tensor<uint16_t>(m, p + "keys", size_t(groups) * 128)), p + "source_keys", false);
            }
            check(history(source, s.value) == before, p + "consumer preserves all source storage");
            check(source.next_position() == pos + tokens && consumer.next_position() == pos + tokens && source.compressed_groups() == groups, p + "positions/groups");
            pos += tokens;
        }
        if (plan == 0) {
            baseline = history(source, s.value); baseline_ring = download(consumer.kv_cache(), 128 * 512, s.value);
            replay_source = download(y2, 9 * hidden, s.value); replay_consumer = download(y3, 9 * hidden, s.value);
        }
        if (plan == 1)
            check(history(source, s.value) == baseline && same(download(consumer.kv_cache(), 128 * 512, s.value), baseline_ring), "both layer histories match across prefill partitions");
    }
    check(source.device_bytes() == source_bytes && consumer.device_bytes() == consumer_bytes, "real sharing has fixed device allocation through ring wrap");
    check(same(download(x2, input2.size(), s.value), input2) && same(download(x3, input3.size(), s.value), input3), "real source/consumer inputs unchanged");
    check(!same(download(source.kv_cache(), 128 * 512, s.value), download(consumer.kv_cache(), 128 * 512, s.value)), "real windows stay layer-specific");
    source.reset(s.value); consumer.reset(s.value);
    for (int pos = 0; pos < 9; ++pos) {
        source.forward(x2 + size_t(pos) * hidden, y2 + size_t(pos) * hidden, 1, pos, s.value);
        consumer.forward(x3 + size_t(pos) * hidden, y3 + size_t(pos) * hidden, 1, pos, source.shared_batch(), s.value);
    }
    check(history(source, s.value) == baseline && same(download(consumer.kv_cache(), 128 * 512, s.value), baseline_ring) &&
          same(download(y2, 9 * hidden, s.value), replay_source) && same(download(y3, 9 * hidden, s.value), replay_consumer), "real paired reset/replay byte-identical");
    std::printf("shared_real_source_device_bytes=%zu consumer_device_bytes=%zu\n", source_bytes, consumer_bytes);
}
} // namespace
int main(int argc, char** argv) {
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        synthetic();
        if (argc == 4 && std::string(argv[1]) == "--real") real(argv[2], argv[3]);
        else if (argc != 1) throw std::runtime_error("Usage: shared_attention_cuda_test [--real source2.gguf shared23.gguf]");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
