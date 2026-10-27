#include "decoder_test_utils.hpp"
#include "engram_fixture.hpp"
#include "strata/deepseek41/layer1_decoder_cuda.hpp"
#include <chrono>

namespace fs = std::filesystem;
namespace {
struct SyntheticEngram {
    Matrix wkv{25600,6144,true};
    std::vector<uint16_t> q = std::vector<uint16_t>(residual_size,0x3f80), k = q;
    Layer1EngramWeights view() const { return {wkv.view(),q,k}; }
};
std::vector<int32_t> case_tokens(const GgufModel& f, int c) {
    return tensor<int32_t>(f,"case."+std::to_string(c)+".tokens",24,c==0?3:1);
}
void synthetic(const fs::path& dir) {
    constexpr int count=130, prefill=129;
    const auto g=geometry(); const SyntheticWeights weights; const auto w=weights.view(1);
    const SyntheticEngram ew; const auto map=engram_fixture::synthetic_map();
    std::vector<int32_t> tokens(count); for (int t=0;t<count;++t) tokens[size_t(t)]=(t*37)%129280;
    tokens[3]=1; tokens.back()=129279;
    std::vector<uint32_t> rows(size_t(count)*24); Layer1EngramHash hash(map); hash.forward(tokens,rows,0);
    auto ids=rows; std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
    const auto path=dir/"rows.gguf";
    engram_fixture::sparse(path,engram_fixture::table_meta(true,ids),engram_fixture::table_tensors(),ids);
    EngramIoOptions options; options.allow_partial=true;
    Stream stream,other; ZeroSource source(g,1);
    rejects("default attention constructor retains layer-0 guard",[&] { CudaAttention wrong(w.attention); });
    rejects("complete layer rejects wrong attention layer before I/O",[&] {
        auto wrong=w; wrong.attention.layer=0; CudaLayer1Decoder invalid(g,wrong,ew.view(),map,"absent",source,6,options);
    });
    rejects("layer-14 Engram weights rejected before layer-1 table I/O",[&] {
        auto wrong=ew.view();wrong.layer=14;CudaLayer1Decoder invalid(g,w,wrong,map,"absent",source,6,options);
    });
    rejects("complete layer rejects compressed layer-1 geometry before I/O",[&] {
        auto wrong=g; wrong.compress_ratios[1]=2; CudaLayer1Decoder invalid(wrong,w,ew.view(),map,"absent",source,6,options);
    });
    core::DeviceArena io(size_t(count)*(5*residual_size+2*hidden+8)*4+65536);
    auto alloc=[&](size_t n) { return static_cast<float*>(io.alloc(n*4)); };
    auto* x=alloc(size_t(count)*residual_size); auto* y=alloc(size_t(count)*residual_size);
    auto* incoming=alloc(size_t(count)*4); auto* next=alloc(size_t(count)*4);
    auto* er=alloc(size_t(count)*residual_size); auto* lookup=alloc(6144);
    auto* norm=alloc(size_t(count)*hidden); auto* ay=alloc(size_t(count)*hidden);
    std::vector<float> input(size_t(count)*residual_size), pre(size_t(count)*4);
    for (size_t i=0;i<input.size();++i) input[i]=bf(.07f*std::sin(float(i%997)*.03f)+.02f*float((i/hidden)%4));
    for (size_t i=0;i<pre.size();++i) pre[i]=.1f+float((i*3+1)%7)*.15f;
    upload(x,input,stream.value); upload(incoming,pre,stream.value);
    std::vector<float> expected, expected_kv;
    {
        CudaLayer1Engram projection(ew.view());
        for (int t=0;t<count;++t) {
            const auto raw=engram_fixture::row_oracle(std::span(rows).subspan(size_t(t)*24,24));
            std::vector<float> values(raw.size()); std::transform(raw.begin(),raw.end(),values.begin(),kernels::f32_from_bf16);
            upload(lookup,values,stream.value); projection.forward(lookup,x+size_t(t)*residual_size,er+size_t(t)*residual_size,stream.value);
        }
        const auto engram_residual=download(er,size_t(count)*residual_size,stream.value);
        check(!same(engram_residual,input),"nonzero Engram residual is exercised");
        upload(norm,collapse_norm(engram_residual,pre),stream.value);
        CudaAttention attention(w.attention,0,1);
        attention.forward(norm,ay,prefill,0,stream.value);
        attention.forward(norm+size_t(prefill)*hidden,ay+size_t(prefill)*hidden,1,prefill,stream.value);
        const auto middle=join(engram_residual,download(ay,size_t(count)*hidden,stream.value),weights.attn_mhc.post(count),weights.attn_mhc.comb(count));
        std::vector<float> attn_pre(size_t(count)*4); const auto coefficients=weights.attn_mhc.pre();
        for (int t=0;t<count;++t) std::copy(coefficients.begin(),coefficients.end(),attn_pre.begin()+t*4);
        upload(norm,collapse_norm(middle,attn_pre),stream.value);
        CudaSharedExpert shared(g,w.shared);
        for (int t=0;t<count;++t) shared.forward(norm+size_t(t)*hidden,ay+size_t(t)*hidden,stream.value);
        expected=join(middle,download(ay,size_t(count)*hidden,stream.value),weights.ffn_mhc.post(count),weights.ffn_mhc.comb(count));
        expected_kv=cache(attention.kv_cache(),stream.value);
    }
    CudaLayer1Decoder decoder(g,w,ew.view(),map,path.string(),source,6,options);
    const size_t bytes=decoder.device_bytes();
    const auto one=std::span(tokens).first(1); std::vector<Route> routes(count);
    upload(y,std::vector<float>(residual_size,123.f),stream.value);
    upload(next,std::vector<float>(4,321.f),stream.value);
    rejects("empty prefill",[&] { decoder.forward({},x,incoming,y,next,0,stream.value); });
    rejects("invalid initial position",[&] { decoder.forward(one,x,incoming,y,next,1,stream.value); });
    rejects("null stream",[&] { decoder.forward(one,x,incoming,y,next,0,nullptr); });
    rejects("null residual",[&] { decoder.forward(one,nullptr,incoming,y,next,0,stream.value); });
    rejects("null incoming",[&] { decoder.forward(one,x,nullptr,y,next,0,stream.value); });
    rejects("null output",[&] { decoder.forward(one,x,incoming,nullptr,next,0,stream.value); });
    rejects("null next coefficients",[&] { decoder.forward(one,x,incoming,y,nullptr,0,stream.value); });
    rejects("misaligned residual",[&] { decoder.forward(one,reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(x)+1),incoming,y,next,0,stream.value); });
    rejects("overflowing address range",[&] { decoder.forward(one,reinterpret_cast<float*>(UINTPTR_MAX-3),incoming,y,next,0,stream.value); });
    rejects("mismatched routes",[&] { decoder.forward(one,x,incoming,y,next,0,stream.value,routes); });
    rejects("token/routes alias",[&] { decoder.forward(std::span(reinterpret_cast<const int32_t*>(routes.data()),1),x,incoming,y,next,0,stream.value,std::span(routes).first(1)); });
    rejects("token/device residual alias before token dereference",[&] { decoder.forward(std::span(reinterpret_cast<const int32_t*>(x),1),x,incoming,y,next,0,stream.value); });
    for (int a=0;a<4;++a) for (int b=a+1;b<4;++b) {
        float* p[]={x,incoming,y,next}; p[b]=p[a];
        rejects("device buffer alias "+std::to_string(a)+"/"+std::to_string(b),[&] { decoder.forward(one,p[0],p[1],p[2],p[3],0,stream.value); });
    }
    for (int a=0;a<4;++a) {
        float* p[]={x,incoming,y,next}; p[a]=reinterpret_cast<float*>(const_cast<uint16_t*>(decoder.kv_cache()));
        rejects("borrowed KV alias "+std::to_string(a),[&] { decoder.forward(one,p[0],p[1],p[2],p[3],0,stream.value); });
    }
    auto invalid=tokens; invalid.back()=129280;
    rejects("last invalid token rejected before the first row read",[&] { decoder.forward(invalid,x,incoming,y,next,0,stream.value); });
    invalid.back()=-1;
    rejects("negative original token",[&] { decoder.forward(invalid,x,incoming,y,next,0,stream.value); });
    check(decoder.next_position()==0 && source.calls==0 && decoder.engram_stats()[0].requests==0 && decoder.engram_stats()[1].requests==0,
          "argument failures preserve both histories and issue no expert/Engram I/O");
    check(download(y,residual_size,stream.value)==std::vector<float>(residual_size,123.f) && download(next,4,stream.value)==std::vector<float>(4,321.f),
          "argument failures preserve both caller outputs");
    std::vector<float> first;
    for (int pass=0;pass<2;++pass) {
        if (pass) {
            const auto before=decoder.engram_stats(); decoder.reset(stream.value);
            check(decoder.next_position()==0 && cache(decoder.kv_cache(),stream.value)==std::vector<float>(128*512,0.f),"reset clears KV and completed position");
            check(before[0].requests==decoder.engram_stats()[0].requests && source.calls==6,"reset retains row/expert accounting and cache");
        }
        decoder.forward(std::span(tokens).first(prefill),x,incoming,y,next,0,stream.value,std::span(routes).first(prefill));
        const auto before=cache(decoder.kv_cache(),stream.value); const auto requests=decoder.engram_stats()[0].requests;
        rejects("second prefill",[&] { decoder.forward(one,x,incoming,y,next,0,stream.value); });
        rejects("noncontiguous decode",[&] { decoder.forward(one,x,incoming,y,next,prefill+1,stream.value); });
        rejects("multi-token decode",[&] { decoder.forward(std::span(tokens).first(2),x,incoming,y,next,prefill,stream.value); });
        rejects("cross-stream decode",[&] { decoder.forward(one,x,incoming,y,next,prefill,other.value); });
        rejects("cross-stream reset",[&] { decoder.reset(other.value); });
        check(decoder.next_position()==prefill && decoder.engram_stats()[0].requests==requests && same(cache(decoder.kv_cache(),stream.value),before),"rejected continuations preserve KV/hash position and I/O");
        decoder.forward(std::span(tokens).last(1),x+size_t(prefill)*residual_size,incoming+prefill*4,y+size_t(prefill)*residual_size,next+prefill*4,prefill,stream.value,std::span(routes).last(1));
        const auto actual=download(y,size_t(count)*residual_size,stream.value);
        compare(actual,expected,"nonzero Engram + incoming mHC + wrapped attention + shared FFN composition");
        compare(cache(decoder.kv_cache(),stream.value),expected_kv,"129-token prefill then decode keeps correct 128-slot ring");
        std::vector<float> expected_pre(size_t(count)*4); const auto cp=weights.ffn_mhc.pre();
        for (int t=0;t<count;++t) std::copy(cp.begin(),cp.end(),expected_pre.begin()+t*4);
        compare(download(next,expected_pre.size(),stream.value),expected_pre,"returns FFN coefficients for next layer, not attention coefficients",true);
        bool original=true;
        for (const auto& route:routes) for (int k=0;k<6;++k) original &= route.ids[size_t(k)]==k && std::abs(route.weights[size_t(k)]-.249999985f)<1e-6f;
        check(original,"original top-six route IDs and normalized weights");
        check(decoder.next_position()==count && decoder.device_bytes()==bytes,"whole call advances position with fixed device allocation");
        if (!pass) first=actual; else check(same(first,actual),"reset clears Engram token history and replays bit-exactly");
    }
    const auto stats=decoder.stats(); const auto rs=decoder.engram_stats();
    check(source.calls==6 && stats.misses==6 && stats.hits==2*count*6-6,"warm expert cache survives reset");
    check(rs[0].requests==2*count*24 && rs[1].requests==2*count*24 && rs[0].cache_hits>0 && rs[1].cache_hits>0,"Engram uses both bounded row caches throughout reset/replay");
    {
        ZeroSource missing(g,1); missing.missing=true;
        CudaLayer1Decoder broken(g,w,ew.view(),map,path.string(),missing,2,options);
        rejects("expert I/O failure after Engram/attention mutation",[&] { broken.forward(one,x,incoming,y,next,0,stream.value); },true);
        check(broken.next_position()==0 && broken.engram_stats()[0].requests==24,"failed call does not publish a completed position");
        rejects("execution failure poisons complete layer",[&] { broken.forward(one,x,incoming,y,next,0,stream.value); },true);
        rejects("reset cannot revive failed layer",[&] { broken.reset(stream.value); },true);
    }
    {
        const auto truncated=dir/"truncated.gguf";
        engram_fixture::sparse(truncated,engram_fixture::table_meta(true,ids),engram_fixture::table_tensors(),ids);
        ZeroSource unopened(g,1); CudaLayer1Decoder broken(g,w,ew.view(),map,truncated.string(),unopened,2,options);
        fs::resize_file(truncated,4096); // Deliberate I/O failure on an owned fixture, after mappings have closed.
        rejects("row I/O failure aborts before attention and expert I/O",[&] { broken.forward(one,x,incoming,y,next,0,stream.value); },true);
        check(broken.next_position()==0 && unopened.calls==0,"failed row read publishes no sequence progress");
        rejects("row failure poisons complete layer",[&] { broken.forward(one,x,incoming,y,next,0,stream.value); },true);
        rejects("reset cannot hide row I/O failure",[&] { broken.reset(stream.value); },true);
    }
    std::printf("synthetic_layer1_device_bytes=%zu row_requests=%llu expert_misses=%llu\n",bytes,(unsigned long long)rs[0].requests,(unsigned long long)stats.misses);
}

// Native boundary is measured separately for strict same-input mHC validation.
// The original CPU graph still gates each Engram/attention residual and final output.
std::vector<std::vector<float>> native_boundaries(const Geometry& g,const DecoderWeights& w,const Layer1EngramWeights& ew,
    const std::vector<uint32_t>& map,const char* table_path,const GgufModel& f,cudaStream_t stream) {
    Layer1EngramHash hash(map); EngramIoOptions options; options.allow_partial=true;
    Layer1EngramTable table(table_path,options); CudaLayer1Engram engram(ew);
    CudaMhc mhc(g,w.attention_mhc); CudaAttention attention(w.attention,0,1);
    core::DeviceArena io(2*1024*1024);
    auto alloc=[&](size_t n) { return static_cast<float*>(io.alloc(n*4)); };
    auto *x=alloc(3*residual_size), *er=alloc(3*residual_size), *pre=alloc(12), *lookup=alloc(6144);
    auto *norm=alloc(3*hidden), *ay=alloc(3*hidden), *middle=alloc(3*residual_size), *dummy=alloc(residual_size), *zero=alloc(hidden);
    ck(cudaMemsetAsync(zero,0,hidden*4,stream)); std::vector<std::vector<float>> result;
    for (int c=0;c<3;++c) {
        const int count=c==0?3:1,pos=c==0?0:c+2; const auto prefix="case."+std::to_string(c)+".";
        const auto tokens=case_tokens(f,c);
        upload(x,tensor<float>(f,prefix+"input",0,size_t(count)*residual_size),stream);
        upload(pre,tensor<float>(f,prefix+"incoming",0,size_t(count)*4),stream);
        for (int t=0;t<count;++t) {
            std::array<uint32_t,24> rows{}; std::array<uint16_t,6144> raw{};
            hash.forward(std::span(tokens).subspan(size_t(t),1),rows,pos+t); table.gather(rows,raw);
            std::vector<float> values(raw.size()); std::transform(raw.begin(),raw.end(),values.begin(),kernels::f32_from_bf16);
            upload(lookup,values,stream); engram.forward(lookup,x+size_t(t)*residual_size,er+size_t(t)*residual_size,stream);
            const auto state=mhc.prepare(er+size_t(t)*residual_size,pre+size_t(t)*4,stream);
            ck(cudaMemcpyAsync(norm+size_t(t)*hidden,state.normalized,hidden*4,cudaMemcpyDeviceToDevice,stream));
            mhc.finish(zero,dummy,stream);
        }
        compare(download(er,size_t(count)*residual_size,stream),tensor<float>(f,prefix+"after_engram",0,size_t(count)*residual_size),prefix+"native Engram vs pinned official boundary");
        attention.forward(norm,ay,count,pos,stream);
        for (int t=0;t<count;++t) {
            (void)mhc.prepare(er+size_t(t)*residual_size,pre+size_t(t)*4,stream);
            mhc.finish(ay+size_t(t)*hidden,middle+size_t(t)*residual_size,stream);
        }
        result.push_back(download(middle,size_t(count)*residual_size,stream));
        const auto reference=tensor<float>(f,prefix+"after_attention",0,size_t(count)*residual_size);
        compare(result.back(),reference,prefix+"native attention residual vs original Block");
        compare(cpu_pre(reference,w.ffn_mhc),tensor<float>(f,prefix+"next_pre",0,size_t(count)*4),prefix+"independent CPU coefficients match original Block on same input",true);
    }
    return result;
}
void real(const char* fixture_path,const char* routed_path,const char* projection_path,const char* reference_path,const char* table_path) {
    const auto g=geometry(); const auto f=GgufModel::open(fixture_path),routed=GgufModel::open(routed_path),projection=GgufModel::open(projection_path);
    const auto* arch=f.meta().get("general.architecture");
    if (!arch || arch->s!="deepseek41-layer1-decoder-fixture") throw std::runtime_error("Expected prepared layer-1 decoder fixture");
    const auto reference=GgufModel::open(reference_path); const auto map=read_engram_token_map(reference.shard(0));
    const auto ew=read_layer1_engram_weights(projection);
    const auto router=tensor<uint16_t>(routed,"router.weight",30,hidden*384); const auto bias=tensor<float>(routed,"router.bias",0,384);
    const DecoderWeights weights{read_mhc_weights(f,g,1,MhcBranch::attention),read_mhc_weights(f,g,1,MhcBranch::ffn),
        read_attention_weights(f,g,1),read_shared_expert(f,g,1),router,bias};
    const ExpertIndex index(routed,g,{1}); std::array<bool,384> populated{};
    for (int c=0;c<3;++c) for (float id:tensor<float>(f,"case."+std::to_string(c)+".route_ids",0,c==0?18:6)) {
        if (id!=std::floor(id) || id<0 || id>=384) throw std::runtime_error("Invalid original expert ID in fixture");
        populated[size_t(id)]=true;
    }
    Stream stream,other; core::DeviceArena io(1024*1024);
    auto alloc=[&](size_t n) { return static_cast<float*>(io.alloc(n*4)); };
    auto *x=alloc(3*residual_size), *y=alloc(3*residual_size), *pre=alloc(12), *next=alloc(12), *norm=alloc(hidden), *moe_output=alloc(hidden);
    const auto boundaries=native_boundaries(g,weights,ew,map,table_path,f,stream.value);
    std::vector<std::vector<float>> expected; std::vector<std::vector<Route>> expected_routes;
    {
        CoveredSource source(g,index,populated,1); CudaMoE moe(g,1,router,bias,source,6,weights.shared);
        for (int c=0;c<3;++c) {
            const int count=c==0?3:1; const auto prefix="case."+std::to_string(c)+".";
            const auto normalized=tensor<float>(f,prefix+"ffn_input",0,size_t(count)*hidden);
            const auto ids=tensor<float>(f,prefix+"route_ids",0,size_t(count)*6),mixing=tensor<float>(f,prefix+"route_weights",0,size_t(count)*6);
            std::vector<float> outputs; std::vector<Route> routes;
            for (int t=0;t<count;++t) {
                ck(cudaMemcpyAsync(norm,normalized.data()+size_t(t)*hidden,hidden*4,cudaMemcpyHostToDevice,stream.value));
                const auto route=moe.forward(norm,moe_output,stream.value); const auto value=download(moe_output,hidden,stream.value);
                outputs.insert(outputs.end(),value.begin(),value.end()); bool correct=true;
                for (size_t k=0;k<6;++k) correct &= route.ids[k]==int32_t(ids[size_t(t)*6+k]) && std::abs(route.weights[k]-mixing[size_t(t)*6+k])<=2e-6f;
                check(correct,prefix+"native quantized MoE baseline retains original routes token="+std::to_string(t)); routes.push_back(route);
            }
            expected.push_back(join(tensor<float>(f,prefix+"after_attention",0,size_t(count)*residual_size),outputs,
                tensor<float>(f,prefix+"post",0,size_t(count)*4),tensor<float>(f,prefix+"comb",0,size_t(count)*16)));
            expected_routes.push_back(routes);
        }
    }
    std::vector<std::vector<float>> first_outputs,first_pre;
    for (int slots:{6,2}) {
        EngramIoOptions options; options.allow_partial=true; options.cache_rows=slots==6?4096:0;
        CoveredSource source(g,index,populated,1); CudaLayer1Decoder decoder(g,weights,ew,map,table_path,source,slots,options);
        const size_t bytes=decoder.device_bytes();
        for (int pass=0;pass<2;++pass) {
            if (pass) decoder.reset(stream.value);
            for (int c=0;c<3;++c) {
                const int count=c==0?3:1,pos=c==0?0:c+2; const auto prefix="case."+std::to_string(c)+".";
                const auto label="real layer1 slots="+std::to_string(slots)+" pass="+std::to_string(pass)+" case="+std::to_string(c);
                const auto tokens=case_tokens(f,c); std::vector<Route> routes(static_cast<size_t>(count));
                upload(x,tensor<float>(f,prefix+"input",0,size_t(count)*residual_size),stream.value);
                upload(pre,tensor<float>(f,prefix+"incoming",0,size_t(count)*4),stream.value);
                decoder.forward(tokens,x,pre,y,next,pos,stream.value,routes);
                const auto actual=download(y,size_t(count)*residual_size,stream.value),coefficients=download(next,size_t(count)*4,stream.value);
                compare(actual,expected[size_t(c)],label+" original Engram/Block + native Q8_1 MoE output");
                compare(coefficients,cpu_pre(boundaries[size_t(c)],weights.ffn_mhc),label+" strict same-input next-layer coefficients",true);
                compare_propagated(coefficients,tensor<float>(f,prefix+"next_pre",0,size_t(count)*4),boundaries[size_t(c)],
                    tensor<float>(f,prefix+"after_attention",0,size_t(count)*residual_size),weights.ffn_mhc,label+" original coefficient propagation bound");
                compare(cache(decoder.kv_cache(),stream.value),tensor<float>(f,prefix+"cache",0,128*512),label+" KV state");
                bool correct=true;
                for (int t=0;t<count;++t) for (size_t k=0;k<6;++k)
                    correct &= routes[size_t(t)].ids[k]==expected_routes[size_t(c)][size_t(t)].ids[k] &&
                        std::abs(routes[size_t(t)].weights[k]-expected_routes[size_t(c)][size_t(t)].weights[k])<=1e-4f;
                check(correct,label+" original expert IDs and weights");
                check(decoder.next_position()==pos+count && decoder.device_bytes()==bytes,label+" completed position and fixed allocation");
                if (slots==6 && !pass) { first_outputs.push_back(actual); first_pre.push_back(coefficients); }
                else {
                    check(same(actual,first_outputs[size_t(c)]),label+" output bit-exact across reset/row cache/expert eviction");
                    check(same(coefficients,first_pre[size_t(c)]),label+" coefficients bit-exact across reset/row cache/expert eviction");
                }
            }
        }
        const auto stats=decoder.stats(); const auto rs=decoder.engram_stats();
        check(stats.hits+stats.misses==60 && stats.expert_h2d_bytes==stats.misses*nvfp4_expert_bytes(g) && stats.router_d2h_bytes==10*384*4 &&
              uint64_t(source.reads())==stats.misses && stats.evictions>0,"original expert-cache accounting slots="+std::to_string(slots));
        if (slots==2) check(stats.misses==60 && stats.evictions==58,"two slots force reload of every original expert");
        check(rs[0].requests==240 && rs[1].requests==240 && (slots==6 ? rs[0].cache_hits>=120 && rs[1].cache_hits>=120 : !rs[0].cache_hits && !rs[1].cache_hits),
              "row cache retained across reset or explicitly disabled slots="+std::to_string(slots));
        rejects("real layer rejects cross-stream reset",[&] { decoder.reset(other.value); });
        std::printf("real_layer1_slots=%d device_bytes=%zu row_cache=%llu row_reads=%llu row_hits=%llu row_bytes=%llu expert_h2d_bytes=%llu hits=%llu misses=%llu evictions=%llu\n",
            slots,bytes,(unsigned long long)options.cache_rows,(unsigned long long)(rs[0].reads+rs[1].reads),(unsigned long long)(rs[0].cache_hits+rs[1].cache_hits),
            (unsigned long long)(rs[0].bytes+rs[1].bytes),(unsigned long long)stats.expert_h2d_bytes,(unsigned long long)stats.hits,(unsigned long long)stats.misses,(unsigned long long)stats.evictions);
    }
}
} // namespace
int main(int argc,char** argv) {
    if (!(argc==1 || (argc==7 && std::string(argv[1])=="--real"))) {
        std::fprintf(stderr,"usage: deepseek41_layer1_decoder_cuda_test [--real DECODER.gguf ROUTED.gguf PROJECTION.gguf LOOKUP_REFERENCE.gguf TABLE.gguf]\n"); return 2;
    }
    int count=0; const auto status=cudaGetDeviceCount(&count);
    if (status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !count)) return 77;
    const auto dir=fs::temp_directory_path()/("strata-layer1-decoder-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try { ck(status); ck(cudaSetDevice(0)); if (argc==1) synthetic(dir); else real(argv[2],argv[3],argv[4],argv[5],argv[6]); }
    catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); ++failures; }
    fs::remove_all(dir); std::printf("%d checks, %d failures\n",checks,failures); return failures?1:0;
}
