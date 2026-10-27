#include "strata/deepseek41/source20_attention_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <type_traits>
#include "strata/deepseek41/layer2_attention_cuda.hpp"

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
History history(const CudaSource20Attention& a, cudaStream_t s) {
    return {download(a.kv_cache(), 128 * 512, s), download(a.compressed_kv_cache(), size_t(a.capacity()) * 512, s),
            download(a.index_key_cache(), size_t(a.capacity()) * 128, s), a.next_position(), a.size()};
}
struct Weights {
    std::array<std::vector<uint8_t>, 5> values, scales;
    std::vector<uint16_t> qnorm, kvnorm, compkv, gate, compnorm, key, keynorm, head_weights;
    std::vector<uint8_t> query, query_scales;
    std::vector<float> sink;
    AttentionWeights attention;
    explicit Weights(int layer = 20) : qnorm(1280, 0x3f80), kvnorm(512, 0x3f80), compkv(512 * hidden, 0), gate(compkv.size(), 0),
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
    Source20CompressorWeights compressor() const { return {compkv, compnorm}; }
    Source20IndexKeyWeights index_key() const { return {key, keynorm}; }
    Source20IndexQueryWeights index_query() const { return {{query, query_scales, 4096, 1280}, head_weights}; }
};

static_assert(!std::is_convertible_v<Source20AttentionBatch, SharedAttentionBatch>);
void publication(const CudaSource20Attention& a, int start, int tokens, cudaStream_t stream) {
    const auto v = a.shared_batch().view(start,tokens,stream);
    const int count=start+tokens, stride=(count+7)/8;
    check(v.start==start && v.tokens==tokens && v.key_count==count && v.index_count==count &&
          v.offset==(start?128:tokens) && v.candidate_stride==stride &&
          v.kv==a.compressed_kv_cache() && v.keys==a.index_key_cache(), "publication shape and borrowed cache identity");
    const auto indices=download(v.indices,size_t(tokens)*v.index_count,stream);
    const auto candidates=download(v.candidates,size_t(tokens)*stride,stream);
    bool correct=true;
    for(int t=0;t<tokens;++t) {
        const int visible=start+t+1;
        for(int j=0;j<count;++j) correct &= indices[size_t(t)*count+j]==(j<visible?j+v.offset:-1);
        for(int j=0;j<stride;++j) correct &= candidates[size_t(t)*stride+j]==(j<(visible+7)/8?j:-1);
    }
    check(correct,"every prefill/decode publication row preserves causal indices and partial candidate blocks");
}
void synthetic() {
    Weights w;
    for(int layer:{0,2,14,21}) rejects("reject other projection owner",[&]{auto bad=w.attention;bad.layer=layer;CudaSource20Attention a(bad,w.compressor(),w.index_key(),w.index_query(),9,8);});
    for(const auto& limits:{std::pair{0,1},std::pair{-1,1},std::pair{1048577,1},std::pair{9,0},std::pair{9,10}})
        rejects("invalid startup limits",[&]{CudaSource20Attention a(w.attention,w.compressor(),w.index_key(),w.index_query(),limits.first,limits.second);});
    auto owner=std::make_unique<CudaSource20Attention>(w.attention,w.compressor(),w.index_key(),w.index_query(),9,8);
    auto& a=*owner;Stream s,other;
    core::DeviceArena buffers(10*hidden*8+1024);
    auto* x=static_cast<float*>(buffers.alloc(10*hidden*4));auto* y=static_cast<float*>(buffers.alloc(10*hidden*4));
    std::vector<float> input(10*hidden,0),sentinel(input.size(),-37);
    for(int t=0;t<10;++t)input[size_t(t)*hidden]=(t%2?-1.f:1.f)*(t+1);
    upload(x,input,s.value);upload(y,sentinel,s.value);
    const auto initial=history(a,s.value);const auto bytes=a.device_bytes();
    rejects("default publication invalid",[&]{Source20AttentionBatch{}.view(0,1,s.value);});
    rejects("no publication before forward",[&]{a.shared_batch();});
    rejects("prefill limit before any producer advance",[&]{a.forward(x,y,9,0,other.value);});
    rejects("null stream",[&]{a.forward(x,y,3,0,nullptr);});
    rejects("null input",[&]{a.forward(nullptr,y,3,0,other.value);});
    rejects("null output",[&]{a.forward(x,nullptr,3,0,other.value);});
    rejects("overlap",[&]{a.forward(x,x,3,0,other.value);});
    rejects("unaligned",[&]{a.forward(reinterpret_cast<const float*>(reinterpret_cast<const char*>(x)+1),y,3,0,other.value);});
    rejects("pointer overflow",[&]{a.forward(reinterpret_cast<const float*>(UINTPTR_MAX-3),y,3,0,other.value);});
    rejects("empty call",[&]{a.forward(x,y,0,0,other.value);});
    rejects("negative position",[&]{a.forward(x,y,1,-1,other.value);});
    rejects("missing prefill",[&]{a.forward(x,y,1,1,other.value);});
    for(const auto* p:{a.kv_cache(),a.compressed_kv_cache(),a.index_key_cache()})
        rejects("cache alias",[&]{a.forward(reinterpret_cast<const float*>(p),y,1,0,other.value);});
    check(history(a,s.value)==initial && same(download(y,sentinel.size(),s.value),sentinel),"all preflight rejections preserve output/history and leave stream unbound");
    a.forward(x,y,3,0,s.value);publication(a,0,3,s.value);
    const auto first=a.shared_batch(),copy=first;const auto v=first.view(0,3,s.value);
    const auto state=history(a,s.value);const auto output=download(y,sentinel.size(),s.value);
    check(a.size()==3 && a.next_position()==3 && !zero(state.compressed) && !zero(state.keys) && !zero(state.ring) &&
          !zero(download(y,3*hidden,s.value)),"odd prefill produces one KV/key per token and nonzero output");
    rejects("publication wrong position",[&]{first.view(1,3,s.value);});
    rejects("publication wrong tokens",[&]{first.view(0,2,s.value);});
    rejects("publication wrong stream",[&]{first.view(0,3,other.value);});
    rejects("publication null stream",[&]{first.view(0,3,nullptr);});
    rejects("stream switch",[&]{a.forward(x,y,1,3,other.value);});
    rejects("reset stream switch",[&]{a.reset(other.value);});
    rejects("position gap",[&]{a.forward(x,y,1,4,s.value);});
    rejects("multi-token continuation",[&]{a.forward(x,y,2,3,s.value);});
    for(const auto* p:{v.indices,v.candidates}) {
        rejects("publication input alias",[&]{a.forward(reinterpret_cast<const float*>(p),y,1,3,s.value);});
        rejects("publication output alias",[&]{a.forward(x,reinterpret_cast<float*>(const_cast<int32_t*>(p)),1,3,s.value);});
    }
    check(first.view(0,3,s.value).sequence==copy.view(0,3,s.value).sequence && history(a,s.value)==state &&
          same(download(y,sentinel.size(),s.value),output),"rejected calls preserve publication, history and output");
    for(int t=3;t<9;++t){a.forward(x+size_t(t)*hidden,y+size_t(t)*hidden,1,t,s.value);publication(a,t,1,s.value);}
    rejects("advance expires original handle",[&]{first.view(0,3,s.value);});
    rejects("advance expires copied handle",[&]{copy.view(0,3,s.value);});
    const auto full=history(a,s.value);const auto full_output=download(y,sentinel.size(),s.value);const auto last=a.shared_batch();
    rejects("full ratio-1 cache cannot accept an extra pending token",[&]{a.forward(x+9*hidden,y+9*hidden,1,9,s.value);});
    check(history(a,s.value)==full && same(download(y,sentinel.size(),s.value),full_output) && last.view(8,1,s.value).key_count==9,
          "capacity rejection preserves completed publication and state");
    a.reset(s.value);rejects("reset expires publication",[&]{last.view(8,1,s.value);});
    rejects("reset has no publication",[&]{a.shared_batch();});
    check(history(a,s.value)==initial,"reset clears full window, KV and key capacities");
    upload(y,sentinel,s.value);a.forward(x,y,3,0,s.value);
    check(a.shared_batch().view(0,3,s.value).sequence==v.sequence+1,"reset changes sequence identity");
    for(int t=3;t<9;++t)a.forward(x+size_t(t)*hidden,y+size_t(t)*hidden,1,t,s.value);
    check(history(a,s.value)==full && same(download(y,sentinel.size(),s.value),full_output),"reset replay is byte-identical");
    a.reset(s.value);a.forward(x,y,8,0,s.value);publication(a,0,8,s.value);a.forward(x+8*hidden,y+8*hidden,1,8,s.value);
    check(history(a,s.value)==full,"ratio-1 histories independent of odd/even prefill partition");
    check(a.device_bytes()==bytes && a.capacity()==9 && same(download(x,input.size(),s.value),input),"fixed startup memory and unchanged input");
    const auto expired=a.shared_batch();owner.reset();rejects("owner destruction expires publication without dereferencing GPU storage",[&]{expired.view(8,1,s.value);});
    std::printf("synthetic_device_bytes=%zu capacity=9 prefill=8\n",bytes);
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

std::vector<int32_t> candidates(const GgufModel& m,const std::string& name,size_t count) {
    size_t shard=0;const auto* t=m.find(name,&shard);
    if(!t || t->type!=24 || tensor_payload_bytes(*t)!=count*4 || !m.in_bounds(*t,shard))throw std::runtime_error("Invalid candidate fixture");
    std::vector<int32_t> out(count);std::memcpy(out.data(),m.shard(shard).tensor_data(*t),count*4);return out;
}
void real(const char* path) {
    const auto m=GgufModel::open(path);
    const auto* oracle=m.meta().get("fixture.linear_oracle");
    if(oracle && (oracle->type!=MetaType::STRING || oracle->s!="fp64-linears-fp32-boundaries-v1"))throw std::runtime_error("Unsupported oracle");
    std::printf("reference_linear_oracle=%s\n",oracle?oracle->s.c_str():"legacy-block32-fp32-bf16");
    Geometry g;g.layers=40;g.hidden=5120;g.compress_ratios.assign(40,0);g.compress_ratios[20]=1;g.sources.resize(40);g.sources[20]={20,20,-1};
    CudaSource20Attention a(read_attention_weights(m,g,20),read_source20_compressor_weights(m,g),
                            read_source20_index_key_weights(m,g),read_source20_index_query_weights(m,g),131,129);
    Stream s;core::DeviceArena buffers(131*hidden*8+1024);
    auto* x=static_cast<float*>(buffers.alloc(131*hidden*4));auto* y=static_cast<float*>(buffers.alloc(131*hidden*4));
    const auto input=expand(tensor<uint16_t>(m,"sequence.input",131*hidden));upload(x,input,s.value);
    const auto bytes=a.device_bytes();History baseline;std::vector<float> replay;
    const int plans=number(m,"fixture.plans",5);if(plans!=5)throw std::runtime_error("Incomplete fixture plans");
    Source20AttentionBatch previous;int previous_start=0,previous_tokens=0;bool had=false;
    for(int plan=0;plan<plans;++plan) {
        const auto prefix="plan."+std::to_string(plan)+".";
        const int first=number(m,prefix+"prefill",129),end=number(m,prefix+"tokens",131);
        if(first<=0 || first>end)throw std::runtime_error("Invalid fixture plan");
        a.reset(s.value);
        if(had)rejects("real reset expires publication",[&]{previous.view(previous_start,previous_tokens,s.value);});
        for(int pos=0;pos<end;) {
            const int tokens=pos?1:first;
            a.forward(x+size_t(pos)*hidden,y+size_t(pos)*hidden,tokens,pos,s.value);
            if(had)rejects("real advance expires old publication",[&]{previous.view(previous_start,previous_tokens,s.value);});
            previous=a.shared_batch();previous_start=pos;previous_tokens=tokens;had=true;
            const auto name=prefix+"at."+std::to_string(pos)+".";const auto v=previous.view(pos,tokens,s.value);
            compare(download(y+size_t(pos)*hidden,size_t(tokens)*hidden,s.value),expand(tensor<uint16_t>(m,name+"output",size_t(tokens)*hidden)),name+"output",true);
            compare(expand(download(a.kv_cache(),128*512,s.value)),expand(tensor<uint16_t>(m,name+"ring",128*512)),name+"ring",true);
            compare(expand(download(a.compressed_kv_cache(),size_t(v.key_count)*512,s.value)),expand(tensor<uint16_t>(m,name+"compressed",size_t(v.key_count)*512)),name+"compressed",false);
            compare(expand(download(a.index_key_cache(),size_t(v.key_count)*128,s.value)),expand(tensor<uint16_t>(m,name+"keys",size_t(v.key_count)*128)),name+"keys",false);
            const auto indices=download(v.indices,size_t(tokens)*v.index_count,s.value);const auto expected=tensor<float>(m,name+"shared_indices",indices.size());
            check(std::equal(indices.begin(),indices.end(),expected.begin()),name+"exact reference indices");
            check(same(download(v.candidates,size_t(tokens)*v.candidate_stride,s.value),candidates(m,name+"candidates",size_t(tokens)*v.candidate_stride)),name+"exact reference candidate blocks");
            publication(a,pos,tokens,s.value);
            check(a.next_position()==pos+tokens && a.size()==pos+tokens,name+"position/history");
            pos+=tokens;
        }
        if(plan==0){baseline=history(a,s.value);replay=download(y,15*hidden,s.value);}
        if(plan>0 && end==15)check(history(a,s.value)==baseline,"real histories byte-identical across prefill partitions");
    }
    check(a.device_bytes()==bytes && a.next_position()==131,"wrap/decode retain startup allocation");
    check(same(download(x,input.size(),s.value),input),"real input unchanged");
    a.reset(s.value);for(int t=0;t<15;++t)a.forward(x+size_t(t)*hidden,y+size_t(t)*hidden,1,t,s.value);
    check(history(a,s.value)==baseline && same(download(y,replay.size(),s.value),replay),"real reset replay byte-identical");
    std::printf("real_device_bytes=%zu capacity=131 prefill=129\n",bytes);
}
} // namespace
int main(int argc,char** argv) {
    if(!(argc==1 || (argc==3 && std::string(argv[1])=="--real")))return 2;
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    try {if(argc==1)synthetic();else real(argv[2]);}
    catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
