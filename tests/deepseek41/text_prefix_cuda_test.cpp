#include "prefix_fixture.hpp"
#include "engram_fixture.hpp"
#include "text_io_reference.hpp"
#include "strata/deepseek41/text_io_cuda.hpp"
#include "strata/deepseek41/generation_cuda.hpp"
#include "strata/deepseek41/memory_plan.hpp"
#include "strata/core/conversation_memory.hpp"
#include <chrono>

namespace fs = std::filesystem;
namespace {
using prefix_fixture::Case;
using prefix_fixture::Result;
struct ZeroLayers : core::ExpertSource {
    std::vector<uint8_t> bytes;
    std::array<uint64_t,40> loads{};
    int fail_layer=-1;
    explicit ZeroLayers(const Geometry& g) : bytes(nvfp4_expert_bytes(g),0) {
        const std::array<float,4> scales{1.f,1.f,1.f,0.f};
        std::memcpy(bytes.data()+bytes.size()-nvfp4_scale_tail_bytes,scales.data(),nvfp4_scale_tail_bytes);
    }
    const uint8_t* blob(int64_t l,int64_t e) override {
        if (l<0 || l>=40 || e<0 || e>=384) throw std::runtime_error("Bad original layer/expert");
        ++loads[size_t(l)]; return l==fail_layer?nullptr:bytes.data();
    }
};
struct Synthetic {
    SyntheticWeights base;
    Matrix engram{25600,6144,true};
    std::vector<uint16_t> qk=std::vector<uint16_t>(residual_size,0x3f80);
    std::array<ConstantMhc,40> attention{ConstantMhc(-.7f),ConstantMhc(-.2f),ConstantMhc(.4f),ConstantMhc(.9f),ConstantMhc(-.5f),ConstantMhc(.6f),ConstantMhc(-.3f),ConstantMhc(.3f),ConstantMhc(-.8f),ConstantMhc(.1f),ConstantMhc(-.1f),ConstantMhc(.7f),ConstantMhc(-.9f),ConstantMhc(.2f),ConstantMhc(-.1f),ConstantMhc(.5f),ConstantMhc(-.4f),ConstantMhc(.8f),ConstantMhc(-.6f),ConstantMhc(.3f),ConstantMhc(.4f),ConstantMhc(-.5f),ConstantMhc(.1f),ConstantMhc(.7f),ConstantMhc(-.2f),ConstantMhc(.6f),ConstantMhc(-.4f),ConstantMhc(.8f),ConstantMhc(.2f),ConstantMhc(-.7f),ConstantMhc(.5f),ConstantMhc(-.1f),ConstantMhc(.6f),ConstantMhc(-.4f),ConstantMhc(.3f),ConstantMhc(-.8f),ConstantMhc(.45f),ConstantMhc(-.25f),ConstantMhc(.9f),ConstantMhc(-.6f)};
    std::array<ConstantMhc,40> ffn{ConstantMhc(-1.4f),ConstantMhc(-.6f),ConstantMhc(.2f),ConstantMhc(1.1f),ConstantMhc(-1.1f),ConstantMhc(.5f),ConstantMhc(-.4f),ConstantMhc(.8f),ConstantMhc(-.2f),ConstantMhc(.9f),ConstantMhc(-1.3f),ConstantMhc(.4f),ConstantMhc(-.7f),ConstantMhc(1.2f),ConstantMhc(.3f),ConstantMhc(-.8f),ConstantMhc(.6f),ConstantMhc(-1.2f),ConstantMhc(.7f),ConstantMhc(-.5f),ConstantMhc(-.9f),ConstantMhc(.8f),ConstantMhc(-.3f),ConstantMhc(.2f),ConstantMhc(.6f),ConstantMhc(-.7f),ConstantMhc(.9f),ConstantMhc(-.2f),ConstantMhc(-.4f),ConstantMhc(1.1f),ConstantMhc(-.8f),ConstantMhc(.3f),ConstantMhc(-.5f),ConstantMhc(.7f),ConstantMhc(-1.1f),ConstantMhc(.4f),ConstantMhc(-.35f),ConstantMhc(.8f),ConstantMhc(-.65f),ConstantMhc(1.2f)};
    std::array<std::vector<float>,40> bias;
    Synthetic() {
        for (size_t l=0;l<40;++l) {
            bias[l].resize(384);
            for (int e=0;e<384;++e) bias[l][size_t(e)]=-float((e+384-int(l)*7)%384);
            // Token-dependent coefficients, so shifting them across token indices is observable.
            ffn[l].scale[0]=.3f;
            for (int row=0;row<4;++row) ffn[l].fn[size_t(row)*residual_size+size_t(row)*hidden+17*l]=.5f;
        }
    }
    TextPrefixWeights view(int count) const {
        TextPrefixWeights w;w.layers.resize(size_t(count));
        for (int l=0;l<count;++l) { auto& v=w.layers[size_t(l)];v=base.view(l);v.attention_mhc=attention[size_t(l)].view();v.ffn_mhc=ffn[size_t(l)].view();v.bias=bias[size_t(l)]; }
        w.engram={engram.view(),qk,qk};w.compressor=base.compressor();w.key=base.index_key();w.query=base.query();
        if(count>=14){w.compressor8=base.compressor(8);w.key8=base.index_key(8);w.query8=base.query(8);}
        if(count>=15){w.engram14={engram.view(),qk,qk,14};w.compressor14=base.compressor(14);w.key14=base.index_key(14);w.query14=base.query(14);}
        if(count>=24){w.compressor20={base.compkv,base.compnorm};w.key20={base.key,base.keynorm};w.query20={base.index_query.view(),base.head_weights};}
        if(count>=28)w.query24={base.index_query.view(),base.head_weights,24};
        if(count>=32)w.query28={base.index_query.view(),base.head_weights,28};
        if(count>=36)w.query32={base.index_query.view(),base.head_weights,32};
        if(count==40)w.query36={base.index_query.view(),base.head_weights,36};
        return w;
    }
};
bool same_routes(const std::vector<Route>& a,const std::vector<Route>& b) {
    if (a.size()!=b.size()) return false;
    for (size_t t=0;t<a.size();++t) for (size_t k=0;k<6;++k)
        if (a[t].ids[k]!=b[t].ids[k] || a[t].weights[k]!=b[t].weights[k]) return false;
    return true;
}
std::vector<std::vector<float>> snapshot(const CudaTextPrefix& p,cudaStream_t s) {
    std::vector<std::vector<float>> result;
    for (int l=0;l<p.layer_count();++l) result.push_back(cache(p.kv_cache(l),s));
    result.push_back(cache(p.compressed_kv_cache(),s,size_t(p.capacity())*512));
    result.push_back(cache(p.index_key_cache(),s,size_t(p.capacity())*128));
    if(p.layer_count()>=14){result.push_back(cache(p.compressed_kv_cache(8),s,size_t(p.capacity())*512));result.push_back(cache(p.index_key_cache(8),s,size_t(p.capacity())*128));}
    if(p.layer_count()>=15){result.push_back(cache(p.compressed_kv_cache(14),s,size_t(p.capacity())*512));result.push_back(cache(p.index_key_cache(14),s,size_t(p.capacity())*128));}
    if(p.layer_count()>=24){result.push_back(cache(p.compressed_kv_cache(20),s,size_t(p.capacity(20))*512));result.push_back(cache(p.index_key_cache(20),s,size_t(p.capacity(20))*128));}
    return result;
}
void verify_sequence(CudaTextPrefix& prefix,const std::vector<Case>& cases,const std::vector<Result>& expected,cudaStream_t s,const std::string& label) {
    size_t largest=0;for(const auto& c:cases)largest=std::max(largest,c.tokens.size());
    core::DeviceArena io(largest*(2*residual_size+8)*4+4096);
    auto alloc=[&](size_t n) { return static_cast<float*>(io.alloc(n*4)); };
    auto *x=alloc(largest*residual_size),*p=alloc(largest*4),*y=alloc(largest*residual_size),*next=alloc(largest*4);
    const size_t bytes=prefix.device_bytes(),workspace=prefix.workspace_bytes();int pos=0;
    for (size_t c=0;c<cases.size();++c) {
        const auto& input=cases[c];const size_t n=input.tokens.size(); const auto name=label+" case="+std::to_string(c);
        upload(x,input.residual,s);upload(p,input.incoming,s);std::vector<Route> routes(n*size_t(prefix.layer_count()));
        prefix.forward(input.tokens,x,p,y,next,pos,s,routes);
        const auto actual=download(y,n*residual_size,s),coeff=download(next,n*4,s);
        const bool finite_bf16=std::all_of(actual.begin(),actual.end(),[](float v){return std::isfinite(v) && v==bf(v);});
        const bool valid_coeff=std::all_of(coeff.begin(),coeff.end(),[](float v){return std::isfinite(v) && v>=0.f && v<=1.000002f;});
        check(finite_bf16 && same(actual,expected[c].residual.back()),name+" finite BF16 residual handoffs match separate-buffer layer calls bit-exactly");
        check(valid_coeff && same(coeff,expected[c].coefficients.back()),name+" finite same-token delayed coefficients match bit-exactly");
        check(same_routes(routes,expected[c].routes),name+" every layer retains original top-six IDs, weights and token order");
        bool equal=true;for(int l=0;l<prefix.layer_count();++l)equal &= same(cache(prefix.kv_cache(l),s),expected[c].kv[size_t(l)]);
        check(equal,name+" all layer window histories match direct calls");
        check(same(cache(prefix.compressed_kv_cache(),s,size_t(prefix.capacity())*512),expected[c].compressed) &&
              same(cache(prefix.index_key_cache(),s,size_t(prefix.capacity())*128),expected[c].keys),name+" layer-2 shared KV/index state preserved through all consumers");
        if(prefix.layer_count()>=14) check(same(cache(prefix.compressed_kv_cache(8),s,size_t(prefix.capacity())*512),expected[c].compressed8) &&
            same(cache(prefix.index_key_cache(8),s,size_t(prefix.capacity())*128),expected[c].keys8) && prefix.compressed_groups(8)==(pos+int(n))/2,
            name+" layer-8 shared KV/index state preserved through all consumers");
        if(prefix.layer_count()>=15) check(same(cache(prefix.compressed_kv_cache(14),s,size_t(prefix.capacity())*512),expected[c].compressed14) &&
            same(cache(prefix.index_key_cache(14),s,size_t(prefix.capacity())*128),expected[c].keys14) && prefix.compressed_groups(14)==(pos+int(n))/2,
            name+" independent layer-14 compressed KV/index state matches explicit calls");
        if(prefix.layer_count()>=24) check(same(cache(prefix.compressed_kv_cache(20),s,size_t(prefix.capacity(20))*512),expected[c].compressed20) &&
            same(cache(prefix.index_key_cache(20),s,size_t(prefix.capacity(20))*128),expected[c].keys20) && prefix.compressed_groups(20)==pos+int(n),
            name+(prefix.layer_count()>=28?" source20 ratio-1 state survives layers 21.."+std::to_string(prefix.layer_count()-1):" source20 ratio-1 state survives consumers 21..23"));
        pos+=int(n);
        check(prefix.next_position()==pos && prefix.compressed_groups()==pos/2 && prefix.device_bytes()==bytes && prefix.workspace_bytes()==workspace,name+" completed prefix position and fixed allocation");
        check(same(download(x,n*residual_size,s),input.residual) && same(download(p,n*4,s),input.incoming),name+" caller input and incoming coefficients unchanged");
    }
}
std::vector<Case> synthetic_cases() {
    std::vector<Case> result;int pos=0;
    for (int n:{3,1,1}) {
        Case c;c.tokens.resize(size_t(n));c.residual.resize(size_t(n)*residual_size);c.incoming.resize(size_t(n)*4);
        for (int t=0;t<n;++t) c.tokens[size_t(t)]=std::array<int32_t,5>{0,455,1805,1,20852}[size_t(pos+t)];
        for (size_t i=0;i<c.residual.size();++i)c.residual[i]=bf(.08f*std::sin(float((i+size_t(pos)*701)%997)*.07f)+.03f*float((i/hidden)%4));
        for (size_t i=0;i<c.incoming.size();++i)c.incoming[i]=.1f+float((i+size_t(pos)*3)%7)*.15f;
        pos+=n;result.push_back(std::move(c));
    }
    return result;
}
// Original embedding/head may be supplied, but the backbone here deliberately
// stays synthetic. Full real-token generation and its routed coverage are later.
void text_io_integration(const fs::path& dir,const char* real_boundary) {
    auto g=geometry();
    for(int l=14;l<20;++l){g.compress_ratios[size_t(l)]=2;g.sources[size_t(l)]={14,14,-1};}
    for(int l=20;l<24;++l){g.compress_ratios[size_t(l)]=1;g.sources[size_t(l)]={20,20,-1};}
    for(int l=24;l<40;++l){g.compress_ratios[size_t(l)]=1;g.sources[size_t(l)]={20,l/4*4,l%4==0?20:-1};}
    const Synthetic weights;const auto w=weights.view(40);TextIoWeights boundary;
    std::unique_ptr<GgufModel> original;
    std::vector<uint16_t> embedding,norm,head;
    if(real_boundary){original=std::make_unique<GgufModel>(GgufModel::open(real_boundary));boundary=read_text_io_weights(*original);}
    else {
        constexpr int vocab=20853;
        embedding.resize(size_t(vocab)*hidden);head.resize(embedding.size());norm.resize(hidden);
        for(size_t i=0;i<embedding.size();++i){
            embedding[i]=kernels::bf16_from_f32(float(int((i*17+3)%67)-33)/128.f);
            head[i]=kernels::bf16_from_f32(float(int((i*11+7)%43)-21)/256.f);
        }
        for(int d=0;d<hidden;++d)norm[size_t(d)]=kernels::bf16_from_f32(.5f+float(d%128)/256.f);
        boundary={hidden,vocab,embedding,norm,head};
    }
    auto cases=synthetic_cases();std::vector<int32_t> tokens;
    for(auto& c:cases){
        c.residual=text_io_reference::embedding(boundary,c.tokens);c.incoming.assign(c.tokens.size()*4,0.f);
        for(size_t t=0;t<c.tokens.size();++t)c.incoming[t*4]=1.f;
        tokens.insert(tokens.end(),c.tokens.begin(),c.tokens.end());
    }
    const auto map=engram_fixture::synthetic_map();Layer1EngramHash hash(map);
    std::vector<uint32_t> rows(tokens.size()*24);hash.forward(tokens,rows,0);
    std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());const auto table=dir/"rows.gguf";
    engram_fixture::sparse(table,engram_fixture::table_meta(true,rows),engram_fixture::table_tensors(),rows);
    auto map14=map;for(auto& id:map14)id=uint32_t((id+17)%engram_compressed_tokens);map14[2]=2;
    EngramHash hash14(14,map14);std::vector<uint32_t> rows14(tokens.size()*24);hash14.forward(tokens,rows14,0);
    std::sort(rows14.begin(),rows14.end());rows14.erase(std::unique(rows14.begin(),rows14.end()),rows14.end());
    const auto table14=(dir/"rows14.gguf").string();
    engram_fixture::sparse(table14,engram_fixture::table_meta(true,rows14,14),engram_fixture::table_tensors(14),rows14,false,false,14);
    EngramIoOptions options;options.allow_partial=true;Stream stream;
    ZeroLayers source(g),reference_source(g);
    const auto expected=prefix_fixture::direct(g,w,map,table.string(),reference_source,cases,2,stream.value,map14,table14);
    std::vector<std::vector<float>> expected_logits;
    for(const auto& r:expected)expected_logits.push_back(text_io_reference::head(boundary,r.residual.back(),r.coefficients.back()));
    CudaTextPrefix prefix(g,w,map,table.string(),source,2,3,2,options,0,map14,table14);
    CudaTextIo io(boundary,3);
    core::DeviceArena buffers(3*(2*residual_size+8)*sizeof(float)+size_t(boundary.vocab_size)*sizeof(float)+2048);
    auto alloc=[&](size_t n){return static_cast<float*>(buffers.alloc(n*sizeof(float)));};
    auto* x=alloc(3*residual_size);auto* pre=alloc(12);auto* y=alloc(3*residual_size);auto* next=alloc(12);auto* logits=alloc(size_t(boundary.vocab_size));
    std::vector<std::vector<float>> first_logits;
    const size_t bytes=prefix.device_bytes()+io.device_bytes();
    for(int pass=0;pass<2;++pass){
        int position=0;if(pass)prefix.reset(stream.value);
        for(size_t c=0;c<cases.size();++c){
            const auto& input=cases[c];const size_t n=input.tokens.size();const auto label="text_io_prefix pass="+std::to_string(pass)+" case="+std::to_string(c);
            io.embed(input.tokens,x,pre,stream.value);
            check(same(download(x,n*residual_size,stream.value),input.residual),label+" token embedding reaches all four residual streams");
            check(same(download(pre,n*4,stream.value),input.incoming),label+" fresh one-hot coefficients for every prefill/decode token");
            std::vector<Route> routes(n*40);prefix.forward(input.tokens,x,pre,y,next,position,stream.value,routes);
            check(same(download(y,n*residual_size,stream.value),expected[c].residual.back()),label+" all40 layers match separate-buffer composition");
            check(same(download(next,n*4,stream.value),expected[c].coefficients.back()) && same_routes(routes,expected[c].routes),label+" final same-token coefficients and every routed ID match");
            io.head(y,next,int(n),logits,stream.value);const auto actual=download(logits,size_t(boundary.vocab_size),stream.value);
            double error=0.,norm2=0.,peak=0.,maximum=0.;bool finite=true;
            for(size_t v=0;v<actual.size();++v){
                const double want=expected_logits[c][v],delta=double(actual[v])-want;
                error+=delta*delta;norm2+=want*want;peak=std::max(peak,std::abs(want));maximum=std::max(maximum,std::abs(delta));finite &= std::isfinite(actual[v]);
            }
            const double relative=std::sqrt(error/std::max(norm2,1e-30));
            std::printf("  %s logits_relative_l2=%.9g max_abs=%.9g peak=%.9g\n",label.c_str(),relative,maximum,peak);
            check(finite && relative<=2e-4 && maximum<=2e-4*(1.+peak),label+" final logits match independent CPU collapse/norm/FP64 dots");
            check(same(download(y,n*residual_size,stream.value),expected[c].residual.back()) && same(download(next,n*4,stream.value),expected[c].coefficients.back()),label+" head leaves final residual and coefficients unchanged");
            if(pass)check(same(actual,first_logits[c]),label+" reset/replay keeps logits bit-identical");else first_logits.push_back(actual);
            position+=int(n);check(prefix.next_position()==position && prefix.device_bytes()+io.device_bytes()==bytes,label+" position and allocations remain consistent");
        }
    }
    std::printf("text_io_prefix40_original_boundary=%d synthetic_backbone=1 prefix_device_bytes=%zu io_device_bytes=%zu caller_device_bytes=%zu host_staging_bytes=%zu vocab_size=%d\n",real_boundary?1:0,prefix.device_bytes(),io.device_bytes(),buffers.capacity(),io.host_staging_bytes(),io.vocab_size());
}

void synthetic(const fs::path& dir,int count) {
    auto g=geometry();if(count>=15){for(int l=14;l<std::min(count,20);++l){g.compress_ratios[size_t(l)]=2;g.sources[size_t(l)]={14,14,-1};}}if(count>=24){for(int l=20;l<24;++l){g.compress_ratios[size_t(l)]=1;g.sources[size_t(l)]={20,20,-1};}}if(count>=28){for(int l=24;l<count;++l){g.compress_ratios[size_t(l)]=1;g.sources[size_t(l)]={20,l/4*4,l%4==0?20:-1};}}const Synthetic weights;const auto w=weights.view(count);const auto map=engram_fixture::synthetic_map();
    const auto cases=synthetic_cases();std::vector<int32_t> tokens;for(const auto& c:cases)tokens.insert(tokens.end(),c.tokens.begin(),c.tokens.end());
    Layer1EngramHash hash(map);std::vector<uint32_t> rows(tokens.size()*24);hash.forward(tokens,rows,0);
    std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());const auto table=dir/"rows.gguf";
    engram_fixture::sparse(table,engram_fixture::table_meta(true,rows),engram_fixture::table_tensors(),rows);
    std::vector<uint32_t> map14;std::string table14;
    if(count>=15) {
        map14=map;for(auto& id:map14)id=uint32_t((id+17)%engram_compressed_tokens);
        map14[2]=2; // Preserve the fixed original padding-token mapping.
        EngramHash hash14(14,map14);std::vector<uint32_t> rows14(tokens.size()*24);hash14.forward(tokens,rows14,0);
        std::sort(rows14.begin(),rows14.end());rows14.erase(std::unique(rows14.begin(),rows14.end()),rows14.end());
        table14=(dir/"rows14.gguf").string();
        engram_fixture::sparse(table14,engram_fixture::table_meta(true,rows14,14),engram_fixture::table_tensors(14),rows14,false,false,14);
    }
    EngramIoOptions options;options.allow_partial=true;Stream stream,other;
    ZeroLayers source(g),reference(g);
    const auto expected=prefix_fixture::direct(g,w,map,table.string(),reference,cases,2,stream.value,map14,table14);
    bool varied=false;for(const auto& r:expected)for(size_t l=1;l<size_t(count);++l)varied |= !same(r.coefficients[l],r.coefficients[l-1]);
    check(varied && expected[0].coefficients[1][0]!=expected[0].coefficients[1][4],"test distinguishes layers and same-token coefficient indices");
    rejects("prefix rejects layer-14 Engram in layer 1 before allocation",[&] { auto bad=w;bad.engram.layer=14;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14); });
    rejects("wrong layer ordering rejected before table I/O",[&] { auto bad=w;std::swap(bad.layers[0],bad.layers[1]);CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14); });
    rejects("wrong shared source rejected before table I/O",[&] { auto bad=g;bad.sources[3].kv=3;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14); });
    rejects("unsupported prefix length before table I/O",[&] { auto bad=w;bad.layers.resize(5);CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14); });
    for(int l=3;l<count;++l) {
        rejects("consumer index source rejected layer="+std::to_string(l),[&] { auto bad=g;bad.sources[size_t(l)].index=(l==8 || l==14 || l==20 || l==24 || l==28 || l==32 || l==36)?2:l;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14); });
        rejects("consumer ratio rejected layer="+std::to_string(l),[&] { auto bad=g;bad.compress_ratios[size_t(l)]=l<20?1:2;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14); });
        rejects("consumer weight identity rejected layer="+std::to_string(l),[&] { auto bad=w;bad.layers[size_t(l)].attention.layer=2;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14); });
    }
    if(count>=14) {
        rejects("mixed source-8 compressor rejected before table I/O",[&]{auto bad=w;bad.compressor8=w.compressor;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("mixed source-8 key rejected before table I/O",[&]{auto bad=w;bad.key8=w.key;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("mixed source-8 query rejected before table I/O",[&]{auto bad=w;bad.query8=w.query;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("source-8 must own KV before table I/O",[&]{auto bad=g;bad.sources[8].kv=2;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        for(int l=9;l<14;++l) rejects("second group cannot borrow source 2 layer="+std::to_string(l),[&]{auto bad=g;bad.sources[size_t(l)]={2,2,-1};CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
    }
    if(count>=15) {
        rejects("layer14 requires its explicit token map and table",[&]{CudaTextPrefix p(g,w,map,"absent",source,6,3,2,options);});
        rejects("layer14 rejects missing table path before layer1 I/O",[&]{CudaTextPrefix p(g,w,map,"absent",source,6,3,2,options,0,map14,{});});
        rejects("layer14 rejects wrong token-map length before layer1 I/O",[&]{CudaTextPrefix p(g,w,map,"absent",source,6,3,2,options,0,std::span(map14).first(12),table14);});
        rejects("layer1 Engram cannot substitute for layer14",[&]{auto bad=w;bad.engram14=w.engram;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        for(int kind=0;kind<3;++kind) rejects("mixed source14 identity rejected before table I/O "+std::to_string(kind),[&]{
            auto bad=w;if(kind==0)bad.compressor14=w.compressor8;else if(kind==1)bad.key14=w.key8;else bad.query14=w.query8;
            CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);
        });
        rejects("source14 cannot borrow source8",[&]{auto bad=g;bad.sources[14]={8,8,-1};CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
    }
    if(count>=20) for(int l=15;l<20;++l) {
        for(int owner:{2,8}) rejects("third group cannot borrow earlier source layer="+std::to_string(l),[&]{auto bad=g;bad.sources[size_t(l)]={owner,owner,-1};CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("third group cannot use candidates",[&]{auto bad=g;bad.sources[size_t(l)].candidates=20;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
    }
    if(count>=24) for(int l=20;l<24;++l) {
        rejects("fourth group requires original source20",[&]{auto bad=g;bad.sources[size_t(l)]={14,14,-1};CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("group20..23 does not consume candidates",[&]{auto bad=g;bad.sources[size_t(l)].candidates=20;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
    }
    if(count>=28) {
        rejects("source24 requires original query weights before I/O",[&]{auto bad=w;bad.query24.layer=28;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("incomplete 29-element index group remains rejected",[&]{auto bad=w;bad.layers.resize(29);CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        for(int l=24;l<count;++l) {
            rejects("new group borrows KV20",[&]{auto bad=g;bad.sources[size_t(l)].kv=24;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
            rejects("new group uses its original index owner",[&]{auto bad=g;bad.sources[size_t(l)].index=20;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
            rejects("only index owners consume candidates20",[&]{auto bad=g;bad.sources[size_t(l)].candidates=l%4==0?-1:20;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        }
        rejects("index source28 requires explicit query weights",[&]{CudaSharedDecoder d(g,weights.base.view(28),source,6,0,28);});
    }
    if(count>=32) {
        rejects("source28 cannot borrow source24 query identity",[&]{auto bad=w;bad.query28.layer=24;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("source28 consumers cannot borrow source24 indices",[&]{auto bad=g;bad.sources[29].index=24;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("incomplete33-element group remains unsupported",[&]{auto bad=w;bad.layers.resize(33);CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("index source32 requires explicit query weights",[&]{CudaSharedDecoder d(g,weights.base.view(32),source,6,0,32);});
    }
    if(count>=36) {
        for(int other:{24,28})rejects("source32 cannot borrow earlier query identity",[&]{auto bad=w;bad.query32.layer=other;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        for(int l=33;l<36;++l)rejects("source32 consumers cannot borrow source28 indices",[&]{auto bad=g;bad.sources[size_t(l)].index=28;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("incomplete37-element group remains unsupported",[&]{auto bad=w;bad.layers.resize(37);CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("index source36 requires explicit query weights",[&]{CudaSharedDecoder d(g,weights.base.view(36),source,6,0,36);});
    }
    if(count==40) {
        for(int other:{24,28,32})rejects("source36 cannot borrow earlier query identity",[&]{auto bad=w;bad.query36.layer=other;CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        for(int l=37;l<40;++l)for(int other:{24,28,32})rejects("source36 consumers cannot borrow earlier indices",[&]{auto bad=g;bad.sources[size_t(l)].index=other;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
        for(int length:{41,43})rejects("MTP layers cannot extend backbone prefix",[&]{auto bad=w;bad.layers.resize(size_t(length));CudaTextPrefix p(g,bad,map,"absent",source,6,3,2,options,0,map14,table14);});
        rejects("MTP layer40 cannot use backbone shared decoder",[&]{CudaSharedDecoder d(g,weights.base.view(40),source,6,0,40);});
        rejects("backbone requires384/top6 rather than draft128/top3",[&]{auto bad=g;bad.experts=128;bad.top_k=3;CudaTextPrefix p(bad,w,map,"absent",source,6,3,2,options,0,map14,table14);});
    }
    rejects("source14 cannot be a shared consumer",[&]{CudaSharedDecoder d(g,weights.base.view(14),source,6,0,14);});
    rejects("layer20 cannot be a shared consumer",[&]{CudaSharedDecoder d(g,weights.base.view(20),source,6,0,20);});
    rejects("implicit consumer stays at layer 3",[&] { CudaSharedDecoder d(g,weights.base.view(4),source,6); });
    rejects("layer 8 requires a new source group",[&] { CudaSharedDecoder d(g,weights.base.view(8),source,6,0,8); });
    rejects("empty prefill capacity",[&] { CudaTextPrefix p(g,w,map,"absent",source,6,0,2,options,0,map14,table14); });
    rejects("empty group capacity",[&] { CudaTextPrefix p(g,w,map,"absent",source,6,3,0,options,0,map14,table14); });
    core::DeviceArena io(2*1024*1024);auto alloc=[&](size_t n){return static_cast<float*>(io.alloc(n*4));};
    auto *x=alloc(6*residual_size),*p=alloc(24),*y=alloc(6*residual_size),*next=alloc(24);
    const auto one=std::span(tokens).first(1);
    size_t completed_device_bytes=0,completed_workspace_bytes=0;
    { // Release the successful full prefix before constructing fault-injection instances.
    CudaTextPrefix prefix(g,w,map,table.string(),source,6,6,2,options,0,map14,table14);
    upload(x,cases[0].residual,stream.value);upload(p,cases[0].incoming,stream.value);
    upload(y,std::vector<float>(3*residual_size,123.f),stream.value);upload(next,std::vector<float>(12,321.f),stream.value);
    std::vector<Route> routes(size_t(3*count));std::vector<int32_t> big(7,0);
    rejects("empty prefill",[&]{prefix.forward({},x,p,y,next,0,stream.value);});
    rejects("oversized prefill",[&]{prefix.forward(big,x,p,y,next,0,stream.value);});
    rejects("later-layer capacity checked before layer 0 executes",[&]{prefix.forward(std::span(big).first(6),x,p,y,next,0,stream.value);});
    rejects("invalid initial position",[&]{prefix.forward(one,x,p,y,next,1,stream.value);});
    rejects("null stream",[&]{prefix.forward(one,x,p,y,next,0,nullptr);});
    rejects("wrong route count",[&]{prefix.forward(one,x,p,y,next,0,stream.value,routes);});
    rejects("host route/token alias",[&]{prefix.forward({reinterpret_cast<int32_t*>(routes.data()),1},x,p,y,next,0,stream.value,std::span(routes).first(size_t(count)));});
    rejects("misaligned input",[&]{prefix.forward(one,reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(x)+1),p,y,next,0,stream.value);});
    rejects("overflowing input range",[&]{prefix.forward(one,reinterpret_cast<float*>(UINTPTR_MAX-3),p,y,next,0,stream.value);});
    auto invalid=cases[0].tokens;invalid.back()=129280;
    rejects("late invalid token checked before layer 0 executes",[&]{prefix.forward(invalid,x,p,y,next,0,stream.value);});
    invalid.back()=-1;rejects("negative token",[&]{prefix.forward(invalid,x,p,y,next,0,stream.value);});
    for(int i=0;i<4;++i){float* a[]={x,p,y,next};a[i]=nullptr;rejects("null caller buffer "+std::to_string(i),[&]{prefix.forward(one,a[0],a[1],a[2],a[3],0,stream.value);});}
    for(int i=0;i<4;++i)for(int j=i+1;j<4;++j){float* a[]={x,p,y,next};a[j]=a[i];rejects("caller alias "+std::to_string(i)+"/"+std::to_string(j),[&]{prefix.forward(one,a[0],a[1],a[2],a[3],0,stream.value);});}
    for(int l=0;l<count+2;++l){const auto* c=l<count?prefix.kv_cache(l):l==count?prefix.compressed_kv_cache():prefix.index_key_cache();
        rejects("owned cache alias "+std::to_string(l),[&]{prefix.forward(one,x,p,reinterpret_cast<float*>(const_cast<uint16_t*>(c)),next,0,stream.value);});}
    if(count>=14) {
        check(prefix.compressed_kv_cache()!=prefix.compressed_kv_cache(8) && prefix.index_key_cache()!=prefix.index_key_cache(8),"two source groups own distinct buffers");
        for(const auto* c:{prefix.compressed_kv_cache(8),prefix.index_key_cache(8)})
            rejects("source-8 cache alias checked before layer 0",[&]{prefix.forward(one,x,p,reinterpret_cast<float*>(const_cast<uint16_t*>(c)),next,0,stream.value);});
    } else rejects("inactive source 8 cache rejected",[&]{(void)prefix.compressed_kv_cache(8);});
    for(int owner:{-1,3,count>=24?24:20}) {
        rejects("invalid compressed source ID",[&]{(void)prefix.compressed_kv_cache(owner);});
        rejects("invalid index source ID",[&]{(void)prefix.index_key_cache(owner);});
        rejects("invalid group-count source ID",[&]{(void)prefix.compressed_groups(owner);});
    }
    if(count>=15) {
        check(prefix.compressed_kv_cache(14)!=prefix.compressed_kv_cache(8) && prefix.compressed_kv_cache(14)!=prefix.compressed_kv_cache(),"third source owns independent compressed storage");
        for(const auto* c:{prefix.compressed_kv_cache(14),prefix.index_key_cache(14)}) {
            auto* alias=reinterpret_cast<float*>(const_cast<uint16_t*>(c));
            rejects("source-14 output alias checked before layer 0",[&]{prefix.forward(one,x,p,alias,next,0,stream.value);});
            rejects("source-14 token alias checked before host read",[&]{prefix.forward({reinterpret_cast<int32_t*>(alias),1},x,p,y,next,0,stream.value);});
            rejects("source-14 route alias checked before layer 0",[&]{prefix.forward(one,x,p,y,next,0,stream.value,{reinterpret_cast<Route*>(alias),size_t(count)});});
        }
        check(prefix.engram_stats(14)[0].requests==0 && prefix.engram_stats(14)[1].requests==0,"preflight failures do not touch second Engram table");
    } else {
        rejects("inactive source14 cache rejected",[&]{(void)prefix.compressed_kv_cache(14);});
        rejects("inactive Engram14 stats rejected",[&]{(void)prefix.engram_stats(14);});
    }
    if(count>=24) {
        check(prefix.capacity()==2 && prefix.capacity(20)==5,"ratio-1 capacity retains final incomplete ratio-2 token");
        for(const auto* c:{prefix.compressed_kv_cache(20),prefix.index_key_cache(20),
                           prefix.compressed_kv_cache(20)+size_t(prefix.capacity(20)-1)*512,
                           prefix.index_key_cache(20)+size_t(prefix.capacity(20)-1)*128}) {
            auto* alias=reinterpret_cast<float*>(const_cast<uint16_t*>(c));
            for(int i=0;i<4;++i){float* a[]={x,p,y,next};a[i]=alias;rejects("source20 alias before layer0",[&]{prefix.forward(one,a[0],a[1],a[2],a[3],0,stream.value);});}
            rejects("source20 token alias before host access",[&]{prefix.forward({reinterpret_cast<int32_t*>(alias),1},x,p,y,next,0,stream.value);});
            rejects("source20 route alias before layer0",[&]{prefix.forward(one,x,p,y,next,0,stream.value,{reinterpret_cast<Route*>(alias),size_t(count)});});
        }
        check(prefix.compressed_kv_cache(20)!=prefix.compressed_kv_cache(14) && prefix.index_key_cache(20)!=prefix.index_key_cache(14),"ratio-1 source owns separate caches");
    }
    rejects("non-Engram layer rejected",[&]{(void)prefix.engram_stats(8);});
    rejects("invalid layer cache request",[&]{(void)prefix.kv_cache(count);});
    const auto history=snapshot(prefix,stream.value);
    check(prefix.next_position()==0 && source.loads==std::array<uint64_t,40>{} && prefix.engram_stats()[0].requests==0,"all invalid calls preserve whole prefix and issue no I/O");
    check(download(y,3*residual_size,stream.value)==std::vector<float>(3*residual_size,123.f) && download(next,12,stream.value)==std::vector<float>(12,321.f),"invalid calls preserve both outputs");
    for(int pass=0;pass<2;++pass){
        if(pass){const auto old=prefix.engram_stats();prefix.reset(stream.value);check(snapshot(prefix,stream.value)==history && prefix.next_position()==0,"one reset clears all layer histories and shared state");check(old[0].requests==prefix.engram_stats()[0].requests,"reset retains row cache accounting");}
        verify_sequence(prefix,cases,expected,stream.value,"synthetic pass="+std::to_string(pass));
        const auto before=snapshot(prefix,stream.value);const auto loads=source.loads;
        rejects("group capacity exhausted before any early layer advances",[&]{prefix.forward(one,x,p,y,next,5,stream.value);});
        rejects("second prefill",[&]{prefix.forward(one,x,p,y,next,0,stream.value);});
        rejects("cross-stream reset",[&]{prefix.reset(other.value);});
        rejects("cross-stream forward",[&]{prefix.forward(one,x,p,y,next,5,other.value);});
        rejects("multi-token decode",[&]{prefix.forward(std::span(tokens).first(2),x,p,y,next,5,stream.value);});
        check(prefix.next_position()==5 && before==snapshot(prefix,stream.value) && loads==source.loads,"rejections after prefill/decode preserve all histories and caches");
    }
    for(size_t l=0;l<size_t(count);++l)check(prefix.stats()[l].misses==6 && prefix.stats()[l].hits==54 && source.loads[l]==6,"independent expert cache retained across reset layer="+std::to_string(l));
    if(count>=15) {
        const auto es=prefix.engram_stats(14);
        check(es[0].requests==480 && es[1].requests==480 && es[0].cache_hits>0 && es[1].cache_hits>0,"both layer14 hash passes replay with independent bounded row caches");
    }
    completed_device_bytes=prefix.device_bytes();completed_workspace_bytes=prefix.workspace_bytes();
    }
    // Failure at every layer: earlier layers may have executed, but the prefix
    // must never publish progress or allow reset to disguise partial execution.
    for(int layer=0;layer<count;++layer){
        // Earlier gates cover every old layer. Exercise both sides of the new
        // group boundary plus each new decoder without duplicating all 20 cases.
        if(count==24 && layer>0 && layer<19)continue;
        if(count==28 && layer>0 && layer<23)continue;
        if(count==32 && layer>0 && layer<27)continue;
        if(count==36 && layer>0 && layer<31)continue;
        if(count==40 && layer>0 && layer<35)continue;
        ZeroLayers missing(g);missing.fail_layer=layer;
        CudaTextPrefix broken(g,w,map,table.string(),missing,2,3,2,options,0,map14,table14);
        rejects("injected source failure layer="+std::to_string(layer),[&]{broken.forward(cases[0].tokens,x,p,y,next,0,stream.value);},true);
        check(broken.next_position()==0 && missing.loads[size_t(layer)]==1,"failed prefix publishes no completed call");
        bool untouched=true;for(int later=layer+1;later<count;++later)untouched &= !missing.loads[size_t(later)];
        check(untouched,"execution stops before later layers");
        rejects("partial prefix cannot continue",[&]{broken.forward(one,x,p,y,next,0,stream.value);},true);
        rejects("partial prefix cannot reset",[&]{broken.reset(stream.value);},true);
    }
    if(count>=15) {
        // Force table I/O to fail after every earlier decoder has executed.
        ZeroLayers row_failure(g);EngramIoOptions no_cache=options;no_cache.cache_rows=0;
        CudaTextPrefix broken(g,w,map,table.string(),row_failure,2,3,2,no_cache,0,map14,table14);
        fs::resize_file(table14,4096);
        rejects("layer14 table failure poisons whole prefix",[&]{broken.forward(cases[0].tokens,x,p,y,next,0,stream.value);},true);
        check(broken.next_position()==0 && row_failure.loads[13]>0 && row_failure.loads[14]==0,"row failure occurs after actual layer13 but before layer14 expert read");
        rejects("row failure cannot continue prefix",[&]{broken.forward(one,x,p,y,next,0,stream.value);},true);
        rejects("row failure cannot be hidden by reset",[&]{broken.reset(stream.value);},true);
    }
    std::printf("synthetic_prefix_device_bytes=%zu workspace_bytes=%zu\n",completed_device_bytes,completed_workspace_bytes);
}
void real(const fs::path& dir,const fs::path& extra,bool probe,int count,bool prepare=false) {
    prefix_fixture::RealWeights f(dir,count);prefix_fixture::Source reference(f,extra,prepare);Stream stream;
    std::vector<Case> cases;
    for(int c=0;c<3;++c){const size_t n=c?1:3;const auto prefix="case."+std::to_string(c)+".";
        cases.push_back({tensor<int32_t>(f.decoder[0],prefix+"tokens",24,n),tensor<float>(f.input,prefix+"input",0,n*residual_size),tensor<float>(f.input,prefix+"incoming",0,n*4)});}
    const auto expected=prefix_fixture::direct(f.g,f.weights,f.map,f.table,reference,cases,3,stream.value,f.map14,f.table14);
    if(probe){
        std::printf("COMPLETE_ORIGINAL_ROUTES [");for(size_t l=0;l<size_t(count);++l){std::printf("%s[",l?",":"");bool first=true;for(int id:reference.selected[l]){std::printf("%s%d",first?"":",",id);first=false;}std::printf("]");}std::printf("]\n");return;
    }
    // Layer 0 is exactly the previously validated fixture input. Later layers
    // receive actual outputs and coefficients, including the native Q8_1 MoE.
    for(size_t c=0;c<cases.size();++c)for(size_t l=1;l<size_t(count);++l)
        check(!same(expected[c].residual[l],expected[c].residual[l-1]) && !same(expected[c].coefficients[l],expected[c].coefficients[l-1]),"real nontrivial layer handoff case="+std::to_string(c)+" layer="+std::to_string(l));
    for(int slots:{6,2}){
        prefix_fixture::Source source(f,extra);EngramIoOptions options;options.allow_partial=true;options.cache_rows=slots==6?4096:0;
        const int prefill=slots==6?3:7;
        CudaTextPrefix prefix(f.g,f.weights,f.map,f.table,source,slots,prefill,3,options,0,f.map14,f.table14);
        for(int pass=0;pass<2;++pass){if(pass)prefix.reset(stream.value);verify_sequence(prefix,cases,expected,stream.value,"real slots="+std::to_string(slots)+" pass="+std::to_string(pass));}
        const auto stats=prefix.stats();for(size_t l=0;l<size_t(count);++l){const auto& s=stats[l];
            check(s.hits+s.misses==60 && s.misses==source.loads[l] && s.expert_h2d_bytes==s.misses*nvfp4_expert_bytes(f.g) && s.router_d2h_bytes==10*384*4 && s.evictions>0,"real per-layer cache accounting layer="+std::to_string(l));
            if(slots==2)check(s.misses>=42 && s.evictions==s.misses-2,"two slots force repeated eviction without changing chain output");
            std::printf("real_prefix_slots=%d layer=%zu hits=%llu misses=%llu evictions=%llu expert_h2d_bytes=%llu\n",slots,l,(unsigned long long)s.hits,(unsigned long long)s.misses,(unsigned long long)s.evictions,(unsigned long long)s.expert_h2d_bytes);
        }
        const auto es=prefix.engram_stats();check(es[0].requests==240 && es[1].requests==240 && (slots==6?es[0].cache_hits>=120 && es[1].cache_hits>=120:!es[0].cache_hits && !es[1].cache_hits),"Engram token history/cache participates in whole-prefix reset");
        if(count>=15) {
            const auto es14=prefix.engram_stats(14);
            check(es14[0].requests==480 && es14[1].requests==480 && (slots==6?es14[0].cache_hits>=240 && es14[1].cache_hits>=240:!es14[0].cache_hits && !es14[1].cache_hits),"independent Engram14 histories/cache participate in whole-prefix reset");
            std::printf("real_prefix_slots=%d engram_layer=14 row_reads=%llu row_hits=%llu row_requests=%llu row_bytes=%llu\n",slots,(unsigned long long)(es14[0].reads+es14[1].reads),(unsigned long long)(es14[0].cache_hits+es14[1].cache_hits),(unsigned long long)(es14[0].requests+es14[1].requests),(unsigned long long)(es14[0].bytes+es14[1].bytes));
        }
        std::printf("real_prefix_slots=%d device_bytes=%zu workspace_bytes=%zu prefill_capacity=%d row_reads=%llu row_hits=%llu\n",slots,prefix.device_bytes(),prefix.workspace_bytes(),prefix.prefill_capacity(),(unsigned long long)(es[0].reads+es[1].reads),(unsigned long long)(es[0].cache_hits+es[1].cache_hits));
    }
}
// The first real token-to-logits control uses bounded original rows/experts.
// Backbone parity is against explicit native calls, not an official full-model oracle.
void original_text_io(const fs::path& dir,const fs::path& extra,const char* boundary_path,
                      bool probe,bool prepare,const char* capture_dir) {
    const fs::path capture=capture_dir?fs::path(capture_dir):fs::path();
    if(capture_dir && !fs::create_directory(capture)) throw std::runtime_error("Capture directory must be new");
    prefix_fixture::RealWeights f(dir,40);
    const auto model=GgufModel::open(boundary_path);const auto boundary=read_text_io_weights(model);
    std::vector<Case> cases(2);cases[0].tokens={0,455};cases[1].tokens={1805};
    for(auto& c:cases){
        c.residual=text_io_reference::embedding(boundary,c.tokens);c.incoming.assign(c.tokens.size()*4,0.f);
        for(size_t t=0;t<c.tokens.size();++t)c.incoming[t*4]=1.f;
    }
    Stream stream;prefix_fixture::Source reference(f,extra,prepare);
    constexpr int groups=2;
    const auto expected=prefix_fixture::direct(f.g,f.weights,f.map,f.table,reference,cases,groups,stream.value,f.map14,f.table14);
    auto print_routes=[&]{
        std::printf("COMPLETE_ORIGINAL_ROUTES [");
        for(size_t l=0;l<40;++l){std::printf("%s[",l?",":"");bool first=true;for(int id:reference.selected[l]){std::printf("%s%d",first?"":",",id);first=false;}std::printf("]");}
        std::printf("]\n");std::fflush(stdout);
    };
    if(probe){print_routes();return;}
    std::vector<std::vector<float>> expected_logits;
    std::vector<std::vector<std::vector<float>>> histories;
    for(size_t c=0;c<cases.size();++c){
        const auto& r=expected[c];expected_logits.push_back(text_io_reference::head(boundary,r.residual.back(),r.coefficients.back()));
        histories.push_back(r.kv);
        for(const auto* values:{&r.compressed,&r.keys,&r.compressed8,&r.keys8,&r.compressed14,&r.keys14,&r.compressed20,&r.keys20})histories.back().push_back(*values);
        for(size_t l=1;l<40;++l)check(!same(r.residual[l],r.residual[l-1]) && !same(r.coefficients[l],r.coefficients[l-1]),
            "original_text_io nontrivial handoff case="+std::to_string(c)+" layer="+std::to_string(l));
    }
    auto save=[&](const std::string& name,const auto& values){
        std::ofstream out(capture/name,std::ios::binary);out.exceptions(std::ios::badbit|std::ios::failbit);
        out.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(values[0])));
    };
    std::vector<std::vector<float>> first_logits;
    for(int slots:{6,2}){
        prefix_fixture::Source source(f,extra);EngramIoOptions options;options.allow_partial=true;options.cache_rows=slots==6?4096:0;
        CudaTextPrefix prefix(f.g,f.weights,f.map,f.table,source,slots,slots==6?2:4,groups,options,0,f.map14,f.table14);
        CudaTextIo io(boundary,2);
        core::DeviceArena buffers(2*(2*residual_size+8)*sizeof(float)+size_t(boundary.vocab_size)*sizeof(float)+2048);
        auto alloc=[&](size_t n){return static_cast<float*>(buffers.alloc(n*sizeof(float)));};
        auto *x=alloc(2*residual_size),*pre=alloc(8),*y=alloc(2*residual_size),*next=alloc(8),*logits=alloc(size_t(boundary.vocab_size));
        const size_t bytes=prefix.device_bytes()+io.device_bytes(),workspace=prefix.workspace_bytes();
        for(int pass=0;pass<2;++pass){
            int position=0;if(pass)prefix.reset(stream.value);
            for(size_t c=0;c<cases.size();++c){
                const auto& input=cases[c];const size_t n=input.tokens.size();
                const auto label="original_text_io slots="+std::to_string(slots)+" pass="+std::to_string(pass)+" case="+std::to_string(c);
                io.embed(input.tokens,x,pre,stream.value);
                check(same(download(x,n*residual_size,stream.value),input.residual) && same(download(pre,n*4,stream.value),input.incoming),label+" original embeddings and one-hot coefficients");
                std::vector<Route> routes(n*40);prefix.forward(input.tokens,x,pre,y,next,position,stream.value,routes);
                const auto residual=download(y,n*residual_size,stream.value),coefficients=download(next,n*4,stream.value);
                check(std::all_of(residual.begin(),residual.end(),[](float v){return std::isfinite(v) && v==bf(v);}) && same(residual,expected[c].residual.back()),label+" final BF16 residual matches explicit original-weight composition");
                check(std::all_of(coefficients.begin(),coefficients.end(),[](float v){return std::isfinite(v) && v>=0.f && v<=1.000002f;}) && same(coefficients,expected[c].coefficients.back()),label+" final same-token coefficients match");
                check(same_routes(routes,expected[c].routes),label+" all40 original top-six routes and weights match");
                const auto state=snapshot(prefix,stream.value);bool state_match=state.size()==histories[c].size();
                if(state_match)for(size_t i=0;i<state.size();++i)state_match &= same(state[i],histories[c][i]);
                check(state_match,label+" all40 windows and all shared KV/index histories match");
                io.head(y,next,int(n),logits,stream.value);const auto actual=download(logits,size_t(boundary.vocab_size),stream.value);
                double square=0.,norm2=0.,peak=0.,maximum=0.;bool finite=true;
                for(size_t v=0;v<actual.size();++v){const double want=expected_logits[c][v],delta=double(actual[v])-want;
                    square+=delta*delta;norm2+=want*want;peak=std::max(peak,std::abs(want));maximum=std::max(maximum,std::abs(delta));finite &= std::isfinite(actual[v]);}
                const double relative=std::sqrt(square/std::max(norm2,1e-30));
                const auto best=std::max_element(actual.begin(),actual.end())-actual.begin();
                std::printf("  %s logits_relative_l2=%.9g max_abs=%.9g peak=%.9g argmax=%td\n",label.c_str(),relative,maximum,peak,best);
                check(finite && relative<=2e-4 && maximum<=2e-4*(1.+peak),label+" all129280 logits match CPU collapse/norm/FP64 dots");
                check(best==std::max_element(expected_logits[c].begin(),expected_logits[c].end())-expected_logits[c].begin(),label+" maximum-logit token agrees");
                check(same(download(y,n*residual_size,stream.value),residual) && same(download(next,n*4,stream.value),coefficients) && same(download(x,n*residual_size,stream.value),input.residual) && same(download(pre,n*4,stream.value),input.incoming),label+" boundary calls preserve input and final state");
                if(slots==6 && pass==0)first_logits.push_back(actual);else check(same(actual,first_logits[c]),label+" reset and cache size keep logits bit-identical");
                position+=int(n);bool groups_match=true;for(int l:{2,8,14})groups_match &= prefix.compressed_groups(l)==position/2;
                check(prefix.next_position()==position && groups_match && prefix.compressed_groups(20)==position && prefix.device_bytes()+io.device_bytes()==bytes && prefix.workspace_bytes()==workspace,label+" positions and fixed allocations agree");
                if(capture_dir && slots==6 && pass==0){const auto name="case"+std::to_string(c)+".";
                    save(name+"tokens.i32",input.tokens);save(name+"embedding.f32",input.residual);save(name+"initial.f32",input.incoming);
                    save(name+"residual.f32",residual);save(name+"coefficients.f32",coefficients);save(name+"logits.f32",actual);}
            }
        }
        const auto stats=prefix.stats();
        for(size_t l=0;l<40;++l){const auto& st=stats[l];
            check(source.selected[l]==reference.selected[l] && st.hits+st.misses==36 && st.misses==source.loads[l] && st.expert_h2d_bytes==st.misses*nvfp4_expert_bytes(f.g) && st.router_d2h_bytes==6*384*4,
                "original_text_io cache/coverage accounting slots="+std::to_string(slots)+" layer="+std::to_string(l));
            if(slots==2)check(st.evictions==st.misses-2 && st.evictions>0,"original_text_io two-slot eviction layer="+std::to_string(l));
        }
        for(int l:{1,14}){const auto es=prefix.engram_stats(l);const uint64_t requests=l==1?144:288;
            check(es[0].requests==requests && es[1].requests==requests && (slots==6?es[0].cache_hits>=requests/2 && es[1].cache_hits>=requests/2:!es[0].cache_hits && !es[1].cache_hits),"original_text_io Engram history/cache reset layer="+std::to_string(l));}
        std::printf("original_text_io slots=%d original_backbone=1 prefix_device_bytes=%zu io_device_bytes=%zu caller_device_bytes=%zu host_staging_bytes=%zu vocab_size=%d\n",slots,prefix.device_bytes(),io.device_bytes(),buffers.capacity(),io.host_staging_bytes(),io.vocab_size());
    }
    print_routes();
}

#include "generation_test.hpp"

} // namespace
int main(int argc,char** argv){
    int count=4;
    if(argc>=3 && std::string(argv[1])=="--layers") {
        const std::string value=argv[2];
        if(value!="4" && value!="8" && value!="14" && value!="15" && value!="20" && value!="24" && value!="28" && value!="32" && value!="36" && value!="40") return 2;
        count=std::stoi(value);argc-=2;argv+=2;
    }
    const bool generation_synth=argc==2 && std::string(argv[1])=="--generation" && count==4;
    const bool generation_real=count==40 && ((argc==6 && std::string(argv[1])=="--generation-prepare") || (argc==7 && std::string(argv[1])=="--generation-real"));
    const bool budgeted=count==40 && argc>=11 && (std::string(argv[1])=="--generate-budgeted" || std::string(argv[1])=="--memory-plan");
    const bool generate=budgeted || (count==40 && argc>=8 && std::string(argv[1])=="--generate");
    const bool text_io=(argc==2 || argc==3) && std::string(argv[1])=="--text-io" && count==40;
    const bool original_io=count==40 && ((argc==5 && (std::string(argv[1])=="--text-io-real" || std::string(argv[1])=="--text-io-probe" || std::string(argv[1])=="--text-io-prepare")) || (argc==6 && std::string(argv[1])=="--text-io-real"));
    if(!(generation_synth || generation_real || generate || text_io || original_io || argc==1 || (argc==4 && (std::string(argv[1])=="--real" || std::string(argv[1])=="--probe" || std::string(argv[1])=="--prepare")))){std::fprintf(stderr,"usage: deepseek41_text_prefix_cuda_test [--layers 4|8|14|15|20|24|28|32|36|40] [--real|--probe|--prepare FIXTURE_DIR SUPPLEMENTAL_DIR] | --layers 40 --text-io [BOUNDARY_FIXTURE] | --layers 40 --text-io-real|--text-io-probe|--text-io-prepare FIXTURE_DIR SUPPLEMENTAL_DIR BOUNDARY_FIXTURE [CAPTURE_DIR (real only)] | --generation | --layers 40 --generation-prepare|--generation-real FIXTURE_DIR SUPPLEMENTAL_DIR BOUNDARY_FIXTURE ENGRAM_DIR [CAPTURE_DIR (real only)] | --layers 40 --generate FIXTURE_DIR SUPPLEMENTAL_DIR BOUNDARY_FIXTURE ENGRAM_DIR MAX_NEW_TOKENS TOKEN_ID... | --layers 40 --generate-budgeted|--memory-plan [--cpu-expert-cache-mib N] [--expert-gguf PATH] FIXTURE_DIR SUPPLEMENTAL_DIR BOUNDARY_FIXTURE ENGRAM_DIR GPU_MIB HOST_MIB EXPERT_SLOTS MAX_NEW_TOKENS TOKEN_ID...\n");return 2;}
    int n=0;const auto status=cudaGetDeviceCount(&n);if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !n))return 77;
    const auto dir=fs::temp_directory_path()/("strata-text-prefix-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(dir);int exit_code=0;
    try{ck(status);ck(cudaSetDevice(0));if(generation_synth)generation_synthetic(dir);else if(generation_real || generate){
        const bool prepare=generation_real && std::string(argv[1])=="--generation-prepare";
        const bool validate=generation_real && !prepare;std::vector<int32_t> tokens{0,455};int budget=2,slots=2;uint64_t cpu_cache_budget=0;std::string expert_gguf;GenerationMemoryLimits limits;
        if(generate){auto integer=[](const char* raw){size_t end=0;const int value=std::stoi(raw,&end);if(end!=std::strlen(raw))throw std::invalid_argument("Invalid integer argument");return value;};
            while(budgeted && (std::string(argv[2])=="--cpu-expert-cache-mib" || std::string(argv[2])=="--expert-gguf")){
                if(argc<13)throw std::invalid_argument("Missing budgeted generation arguments");
                if(std::string(argv[2])=="--expert-gguf"){expert_gguf=argv[3];if(expert_gguf.empty())throw std::invalid_argument("Empty expert GGUF path");}
                else{const int mib=integer(argv[3]);if(mib<0)throw std::invalid_argument("CPU expert cache allowance cannot be negative");cpu_cache_budget=uint64_t(mib)<<20;}
                for(int i=2;i<argc-2;++i)argv[i]=argv[i+2];
                argc-=2;
            }
            if(budgeted){const int gpu=integer(argv[6]),host=integer(argv[7]);if(gpu<=0 || host<=0)throw std::invalid_argument("Memory limits must be positive MiB");limits.device_bytes=uint64_t(gpu)<<20;limits.host_bytes=uint64_t(host)<<20;slots=integer(argv[8]);}
            budget=integer(argv[budgeted?9:6]);tokens.clear();for(int i=budgeted?10:7;i<argc;++i)tokens.push_back(integer(argv[i]));}
        generation_original(argv[2],argv[3],argv[4],argv[5],tokens,budget,prepare,validate,validate?argv[6]:nullptr,budgeted?&limits:nullptr,slots,budgeted && std::string(argv[1])=="--memory-plan",cpu_cache_budget,expert_gguf.empty()?nullptr:expert_gguf.c_str());
    }else if(original_io)original_text_io(argv[2],argv[3],argv[4],std::string(argv[1])!="--text-io-real",std::string(argv[1])=="--text-io-prepare",argc==6?argv[5]:nullptr);else if(text_io)text_io_integration(dir,argc==3?argv[2]:nullptr);else if(argc==1)synthetic(dir,count);else real(argv[2],argv[3],std::string(argv[1])!="--real",count,std::string(argv[1])=="--prepare");}
    catch(const prefix_fixture::MissingExperts& e){if((argc==4 && std::string(argv[1])=="--probe") || (original_io && std::string(argv[1])=="--text-io-probe")){
        std::printf("MISSING_EXPERTS {\"layer\":%d,\"experts\":[",e.layer);for(size_t i=0;i<e.ids.size();++i)std::printf("%s%d",i?",":"",e.ids[i]);std::printf("]}\n");exit_code=78;
    }else{std::fprintf(stderr,"Unpopulated original experts in layer %d\n",e.layer);++failures;}}
    catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    fs::remove_all(dir);std::printf("%d checks, %d failures\n",checks,failures);return exit_code?exit_code:failures?1:0;
}
