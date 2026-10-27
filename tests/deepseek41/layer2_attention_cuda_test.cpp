#include "strata/deepseek41/layer2_attention_cuda.hpp"
#include "strata/deepseek41/attention_cuda.hpp"
#include "strata/deepseek41/layer3_attention_cuda.hpp"
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
    explicit Weights(int layer = 2) : qnorm(1280, 0x3f80), kvnorm(512, 0x3f80), compkv(512 * hidden, 0), gate(compkv.size(), 0),
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
        attention.q_norm = qnorm; attention.kv_norm = kvnorm; attention.sink = sink; attention.layer = layer;
    }
    CompressorWeights compressor() const { return {compkv, gate, compnorm, attention.layer}; }
    IndexKeyWeights index_key() const { return {key, keynorm, attention.layer}; }
    IndexQueryWeights index_query() const { return {{query, query_scales, 4096, 1280}, head_weights, attention.layer}; }
};
void synthetic(int layer) {
    std::printf("synthetic_source_layer=%d\n",layer);
    Weights w(layer);
    rejects("pure-SWA class refuses compressed source weights", [&] { CudaAttention a(w.attention); });
    rejects("source class refuses layer-0 weights", [&] { auto bad = w.attention; bad.layer = 0; CudaSourceAttention a(bad, w.compressor(), w.index_key(), w.index_query(), 3, 0, 0, layer); });
    for (int capacity : {0, -1, 524289})
        rejects("invalid capacity=" + std::to_string(capacity), [&] { CudaSourceAttention a(w.attention, w.compressor(), w.index_key(), w.index_query(), capacity, 0, 0, layer); });
    CudaSourceAttention a(w.attention, w.compressor(), w.index_key(), w.index_query(), 3, 0, 0, layer);
    Stream s, other;
    core::DeviceArena buffers(8 * hidden * 8 + 1024);
    auto* x = static_cast<float*>(buffers.alloc(8 * hidden * 4));
    auto* y = static_cast<float*>(buffers.alloc(8 * hidden * 4));
    std::vector<float> input(8 * hidden, 0.f), sentinel(input.size(), -37.f);
    const float values[] = {1.f, 2.f, -1.f, 4.f, 3.f, -2.f, 5.f, -4.f};
    for (int t = 0; t < 8; ++t) input[size_t(t) * hidden] = values[t];
    upload(x, input, s.value); upload(y, sentinel, s.value);
    const auto initial = history(a, s.value); const auto bytes = a.device_bytes();
    auto unchanged = [&](const History& before, const std::vector<float>& output, const std::string& name) {
        check(history(a, s.value) == before && same(download(y, output.size(), s.value), output), name);
    };
    rejects("oversized prefill checked before any state", [&] { a.forward(x, y, 8, 0, other.value); });
    rejects("null stream", [&] { a.forward(x, y, 3, 0, nullptr); });
    rejects("null input", [&] { a.forward(nullptr, y, 3, 0, other.value); });
    rejects("null output", [&] { a.forward(x, nullptr, 3, 0, other.value); });
    rejects("input/output alias", [&] { a.forward(x, x, 3, 0, other.value); });
    rejects("unaligned input", [&] { a.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(x) + 1), y, 3, 0, other.value); });
    rejects("pointer range overflow", [&] { a.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), y, 3, 0, other.value); });
    rejects("zero tokens", [&] { a.forward(x, y, 0, 0, other.value); });
    rejects("negative position", [&] { a.forward(x, y, 1, -1, other.value); });
    rejects("missing initial prefill", [&] { a.forward(x, y, 1, 1, other.value); });
    rejects("context overflow", [&] { a.forward(x, y, 1048577, 0, other.value); });
    for (const auto* p : {a.kv_cache(), a.compressed_kv_cache(), a.index_key_cache()})
        rejects("exposed cache alias", [&] { a.forward(reinterpret_cast<const float*>(p), y, 1, 0, other.value); });
    unchanged(initial, sentinel, "all initial argument rejections preserve all caches/output and do not bind stream");
    a.forward(x, y, 3, 0, s.value);
    check(a.next_position() == 3 && a.compressed_groups() == 1, "odd prefill leaves one pending token");
    const auto odd = history(a, s.value); const auto odd_output = download(y, sentinel.size(), s.value);
    check(!zero(odd.ring) && !zero(odd.compressed) && !zero(odd.keys) && !zero(download(y, 3 * hidden, s.value)), "sparse nonzero weights exercise every path");
    rejects("stream switch", [&] { a.forward(x, y, 1, 3, other.value); });
    rejects("reset stream switch", [&] { a.reset(other.value); });
    rejects("noncontiguous decode", [&] { a.forward(x, y, 1, 4, s.value); });
    rejects("second prefill", [&] { a.forward(x, y, 1, 0, s.value); });
    rejects("multi-token continuation", [&] { a.forward(x, y, 2, 3, s.value); });
    rejects("decode output cache alias", [&] { a.forward(x, reinterpret_cast<float*>(const_cast<uint16_t*>(a.compressed_kv_cache())), 1, 3, s.value); });
    unchanged(odd, odd_output, "invalid decode preserves pending compressor group and histories");
    for (int t = 3; t < 7; ++t) a.forward(x + size_t(t) * hidden, y + size_t(t) * hidden, 1, t, s.value);
    const auto full = history(a, s.value); const auto full_output = download(y, sentinel.size(), s.value);
    check(full.next == 7 && full.groups == 3 && a.capacity() == 3 && a.device_bytes() == bytes, "full cache accepts one pending token with bounded memory");
    rejects("next emission refused before any component advances", [&] { a.forward(x + 7 * hidden, y + 7 * hidden, 1, 7, s.value); });
    unchanged(full, full_output, "capacity rejection preserves all histories and output");
    a.reset(s.value);
    check(history(a, s.value) == initial, "reset clears full cache capacity and all sequence state");
    upload(y, sentinel, s.value);
    a.forward(x, y, 3, 0, s.value);
    for (int t = 3; t < 7; ++t) a.forward(x + size_t(t) * hidden, y + size_t(t) * hidden, 1, t, s.value);
    unchanged(full, full_output, "reset/replay after rejection is byte-identical");
    a.reset(s.value); a.forward(x, y, 3, 0, s.value); a.reset(s.value);
    std::vector<float> zeros(input.size(), 0.f); upload(x, zeros, s.value);
    a.forward(x, y, 2, 0, s.value);
    const auto clean = history(a, s.value);
    check(zero(download(y, 2 * hidden, s.value)) && zero(clean.ring) && zero(clean.compressed) && zero(clean.keys), "reset discards odd partial group before zero prefill");
    upload(x, input, s.value); a.reset(s.value); a.forward(x, y, 7, 0, s.value);
    check(history(a, s.value) == full, "prefill and decode produce identical compressed/key/window histories");
    check(same(download(x, input.size(), s.value), input), "caller input unchanged");
    std::printf("source_layer=%d device_bytes=%zu capacity=3 shared_prefill=0\n", layer, bytes);
}
// Source-14 publication is deliberately not consumed by future layers 15..19
// in this gate. Existing source-2/source-8 consumers must reject it.
void publications() {
    Weights w(14);Stream s,other;core::DeviceArena buffers(5*hidden*8);
    auto* x=static_cast<float*>(buffers.alloc(5*hidden*4));auto* y=static_cast<float*>(buffers.alloc(5*hidden*4));
    std::vector<float> input(5*hidden,0.f),sentinel(5*hidden,-23.f);
    for(size_t i=0;i<5;++i)input[i*hidden]=float(i+1);
    upload(x,input,s.value);upload(y,sentinel,s.value);
    rejects("legacy source constructor still defaults to layer 2",[&]{CudaLayer2Attention a(w.attention,w.compressor(),w.index_key(),w.index_query(),2);});
    for(int wrong : {2,8}) {
        auto c=w.compressor();c.layer=wrong;
        rejects("source-14 rejects another compressor",[&]{CudaSourceAttention a(w.attention,c,w.index_key(),w.index_query(),2,0,3,14);});
        auto k=w.index_key();k.layer=wrong;
        rejects("source-14 rejects another key projection",[&]{CudaSourceAttention a(w.attention,w.compressor(),k,w.index_query(),2,0,3,14);});
        auto q=w.index_query();q.layer=wrong;
        rejects("source-14 rejects another query projection",[&]{CudaSourceAttention a(w.attention,w.compressor(),w.index_key(),q,2,0,3,14);});
        auto aw=w.attention;aw.layer=wrong;
        rejects("source-14 rejects another attention projection",[&]{CudaSourceAttention a(aw,w.compressor(),w.index_key(),w.index_query(),2,0,3,14);});
    }
    for(int wrong : {3,13,15,20}) {
        auto aw=w.attention;auto c=w.compressor();auto k=w.index_key();auto q=w.index_query();aw.layer=c.layer=k.layer=q.layer=wrong;
        rejects("unsupported source rejected before allocation",[&]{CudaSourceAttention a(aw,c,k,q,2,0,3,wrong);});
    }
    auto source=std::make_unique<CudaSourceAttention>(w.attention,w.compressor(),w.index_key(),w.index_query(),2,0,3,14);
    rejects("no source-14 publication before successful forward",[&]{source->shared_batch();});
    const auto initial=history(*source,s.value);
    rejects("publication capacity checked before source mutation",[&]{source->forward(x,y,4,0,s.value);});
    check(history(*source,s.value)==initial && same(download(y,sentinel.size(),s.value),sentinel),"publication rejection preserves all source state and output");
    source->forward(x,y,3,0,s.value);const auto first=source->shared_batch(),copy=first;
    check(first.source_layer()==14 && copy.source_layer()==14,"completed source-14 prefill publishes its original identity");
    const auto saved=history(*source,s.value);const auto saved_output=download(y,sentinel.size(),s.value);
    rejects("rejected source call preserves publication",[&]{source->forward(x,y,1,4,s.value);});
    rejects("rejected reset preserves publication",[&]{source->reset(other.value);});
    check(first.source_layer()==14 && history(*source,s.value)==saved && same(download(y,sentinel.size(),s.value),saved_output),"source-14 argument rejection leaves borrowed handles and all state valid");
    for(int consumer_layer : {3,9}) {
        auto aw=w.attention;aw.layer=consumer_layer;CudaSharedAttention consumer(aw,0,consumer_layer);
        const auto ring=download(consumer.kv_cache(),128*512,s.value);
        rejects("existing group rejects source-14 at matching positions",[&]{consumer.forward(x,y,3,0,first,s.value);});
        check(consumer.next_position()==0 && same(download(consumer.kv_cache(),128*512,s.value),ring) &&
              history(*source,s.value)==saved && same(download(y,sentinel.size(),s.value),saved_output),"cross-group rejection preserves consumer, source and caller output");
    }
    {
        auto aw=w.attention;auto c=w.compressor();auto k=w.index_key();auto q=w.index_query();aw.layer=c.layer=k.layer=q.layer=8;
        CudaSourceAttention eight(aw,c,k,q,2,0,3,8);eight.forward(x,y,3,0,s.value);const auto b8=eight.shared_batch();const auto old8=history(eight,s.value);
        check(b8.source_layer()==8 && first.source_layer()==14 && history(*source,s.value)==saved,"independent source-8 forward preserves source-14 publication and caches");
        source->reset(s.value);rejects("source-14 reset expires copied handles",[&]{(void)copy.source_layer();});
        check(b8.source_layer()==8 && history(eight,s.value)==old8,"resetting source 14 preserves independent source 8");
        source->forward(x,y,3,0,s.value);const auto fresh=source->shared_batch();
        rejects("same-position replay does not revive old publication",[&]{(void)first.source_layer();});
        check(fresh.source_layer()==14 && history(*source,s.value)==saved && same(download(y,sentinel.size(),s.value),saved_output),"source-14 reset/replay is byte-identical with a fresh publication");
        eight.reset(s.value);check(fresh.source_layer()==14 && history(*source,s.value)==saved,"resetting source 8 preserves independent source 14");
    }
    auto previous=source->shared_batch();source->forward(x+3*hidden,y+3*hidden,1,3,s.value);
    rejects("source advance expires previous publication",[&]{(void)previous.source_layer();});
    previous=source->shared_batch();source->forward(x+4*hidden,y+4*hidden,1,4,s.value);
    rejects("pending-group decode also expires previous publication",[&]{(void)previous.source_layer();});
    const auto full=source->shared_batch();const auto full_state=history(*source,s.value);
    rejects("full source-14 capacity refuses next group before mutation",[&]{source->forward(x,y,1,5,s.value);});
    check(full.source_layer()==14 && history(*source,s.value)==full_state,"capacity rejection retains last completed publication");
    std::printf("source_layer=14 publication_device_bytes=%zu capacity=2 shared_prefill=3\n",source->device_bytes());
    source.reset();rejects("destroyed source expires publication without a GPU dereference",[&]{(void)full.source_layer();});
    CudaSourceAttention disabled(w.attention,w.compressor(),w.index_key(),w.index_query(),2,0,0,14);
    disabled.forward(x,y,3,0,s.value);rejects("source-14 publication remains opt-in",[&]{disabled.shared_batch();});
    check(history(disabled,s.value)==saved && same(download(y,3*hidden,s.value),std::vector<float>(saved_output.begin(),saved_output.begin()+3*hidden)),"disabled publication preserves source arithmetic and histories");
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
void real(const char* path, int layer) {
    const auto m = GgufModel::open(path);
    const auto* oracle = m.meta().get("fixture.linear_oracle");
    if (oracle && (oracle->type != MetaType::STRING || oracle->s != "fp64-linears-fp32-boundaries-v1"))
        throw std::runtime_error("Unsupported fixture linear oracle");
    std::printf("reference_linear_oracle=%s\n", oracle ? oracle->s.c_str() : "legacy-block32-fp32-bf16");
    Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios.assign(40, 0); g.compress_ratios[size_t(layer)] = 2;
    g.sources.resize(40);g.sources[size_t(layer)]={layer,layer,-1};
    const bool publish=layer!=2;
    CudaSourceAttention a(read_attention_weights(m, g, layer), read_compressor_weights(m, g, layer),
                          read_index_key_weights(m, g, layer), read_index_query_weights(m, g, layer), 66, 0, publish?129:0, layer);
    Stream s; core::DeviceArena buffers(131 * hidden * 8 + 1024);
    auto* x = static_cast<float*>(buffers.alloc(131 * hidden * 4));
    auto* y = static_cast<float*>(buffers.alloc(131 * hidden * 4));
    const auto input = expand(tensor<uint16_t>(m, "sequence.input", 131 * hidden)); upload(x, input, s.value);
    const auto bytes = a.device_bytes(); History baseline; std::vector<float> replay;
    const int plans = number(m, "fixture.plans", 5);
    if (plans != 5) throw std::runtime_error("Incomplete fixture plans");
    SharedAttentionBatch previous;bool had_publication=false;
    for (int plan = 0; plan < plans; ++plan) {
        const std::string p = "plan." + std::to_string(plan) + ".";
        const int first = number(m, p + "prefill", 129), end = number(m, p + "tokens", 131);
        if (first <= 0 || first > end) throw std::runtime_error("Invalid fixture plan");
        a.reset(s.value);
        if(publish && had_publication)rejects("real reset expires publication",[&]{(void)previous.source_layer();});
        for (int pos = 0; pos < end;) {
            const int tokens = pos ? 1 : first;
            a.forward(x + size_t(pos) * hidden, y + size_t(pos) * hidden, tokens, pos, s.value);
            if(publish) {
                if(had_publication)rejects("real advance/reset expires previous publication",[&]{(void)previous.source_layer();});
                previous=a.shared_batch();had_publication=true;check(previous.source_layer()==layer,"real publication keeps original source identity");
            }
            const auto prefix = p + "at." + std::to_string(pos) + ".";
            compare(download(y + size_t(pos) * hidden, size_t(tokens) * hidden, s.value),
                    expand(tensor<uint16_t>(m, prefix + "output", size_t(tokens) * hidden)), prefix + "output", true);
            compare(expand(download(a.kv_cache(), 128 * 512, s.value)), expand(tensor<uint16_t>(m, prefix + "ring", 128 * 512)), prefix + "ring", true);
            const int groups = (pos + tokens) / 2;
            if (groups) {
                compare(expand(download(a.compressed_kv_cache(), size_t(groups) * 512, s.value)),
                        expand(tensor<uint16_t>(m, prefix + "compressed", size_t(groups) * 512)), prefix + "compressed", false);
                compare(expand(download(a.index_key_cache(), size_t(groups) * 128, s.value)),
                        expand(tensor<uint16_t>(m, prefix + "keys", size_t(groups) * 128)), prefix + "keys", false);
            }
            check(a.next_position() == pos + tokens && a.compressed_groups() == groups, prefix + "position/groups");
            pos += tokens;
        }
        if (plan == 0) { baseline = history(a, s.value); replay = download(y, 15 * hidden, s.value); }
        if (plan > 0 && end == 15) check(history(a, s.value) == baseline, p + "histories byte-identical across prefill partitions");
    }
    check(a.device_bytes() == bytes && a.next_position() == 131, "real wrap/decode keeps startup allocation fixed");
    check(same(download(x, input.size(), s.value), input), "real input unchanged");
    a.reset(s.value);
    for (int pos = 0; pos < 15; ++pos) a.forward(x + size_t(pos) * hidden, y + size_t(pos) * hidden, 1, pos, s.value);
    check(history(a, s.value) == baseline && same(download(y, replay.size(), s.value), replay), "real reset/replay byte-identical");
    std::printf("source_layer=%d real_device_bytes=%zu capacity=66 shared_prefill=%d\n", layer, bytes, publish?129:0);
}
} // namespace
int main(int argc, char** argv) {
    int layer=2,arg=1;
    if(argc>1 && std::string(argv[1])=="--layer") {
        if(argc<3 || (std::string(argv[2])!="2" && std::string(argv[2])!="8" && std::string(argv[2])!="14"))return 2;
        layer=std::stoi(argv[2]);arg=3;
    }
    if(!(argc==arg || (argc==arg+2 && std::string(argv[arg])=="--real"))) {
        std::fprintf(stderr,"Usage: layer2_attention_cuda_test [--layer 2|8|14] [--real fixture.gguf]\n");return 2;
    }
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        synthetic(layer);
        if(layer==14)publications();
        if(argc!=arg)real(argv[arg+1],layer);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
