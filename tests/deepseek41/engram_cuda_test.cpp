#include "strata/deepseek41/engram_cuda.hpp"
#include "strata/deepseek41/engram.hpp"
#include "strata/deepseek41/shared_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr size_t dim=5120, residual_size=4*dim, lookup_size=6144, kv_size=5*dim;
int checks=0,failures=0;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(bool ok,const std::string& label) { ++checks; failures+=!ok; std::printf("%s %s\n",ok?"ok  ":"FAIL",label.c_str()); }
void rejects(const std::string& label,const std::function<void()>& f) {
    try { f(); check(false,label); } catch(const std::exception&) { check(true,label); }
}
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
bool same(std::span<const float> a,std::span<const float> b) { return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size_bytes())==0; }
struct Stream {
    cudaStream_t s{};
    Stream(){ ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); }
    ~Stream(){ cudaStreamDestroy(s); }
};
std::vector<float> copy(const float* p,size_t n,cudaStream_t s) {
    std::vector<float> out(n); ck(cudaMemcpyAsync(out.data(),p,n*4,cudaMemcpyDeviceToHost,s)); ck(cudaStreamSynchronize(s)); return out;
}
void upload(float* p,std::span<const float> v,cudaStream_t s) { ck(cudaMemcpyAsync(p,v.data(),v.size_bytes(),cudaMemcpyHostToDevice,s)); }
void compare(std::span<const float> actual,std::span<const float> expected,const std::string& name,int kind=0) {
    if(actual.size()!=expected.size()) throw std::runtime_error("comparison shape mismatch");
    double e2=0,n2=0,maximum=0,peak=0; bool finite=true,pointwise=true;
    for(size_t i=0;i<actual.size();++i) {
        const double d=double(actual[i])-expected[i]; e2+=d*d; n2+=double(expected[i])*expected[i];
        maximum=std::max(maximum,std::abs(d)); peak=std::max(peak,std::abs(double(expected[i])));
        finite&=std::isfinite(actual[i])&&std::isfinite(expected[i]);
        pointwise&=std::abs(d)<=2e-6+2e-5*std::abs(double(expected[i]));
    }
    const double rel=std::sqrt(e2/std::max(n2,1e-30));
    std::printf("metric %s relL2=%.9g max_abs=%.9g peak=%.9g\n",name.c_str(),rel,maximum,peak);
    check(finite&&(kind==2?pointwise:kind==1?maximum<=2e-4:rel<=.003&&maximum<=1e-5+.02*peak),name);
}
struct Expected { std::vector<float> gate,output; };
Expected gate_reference(std::span<const float> x,std::span<const float> kv,const EngramWeights& w,bool live) {
    Expected r; r.gate.resize(4); r.output.resize(residual_size);
    for(size_t c=0;c<4;++c) {
        // Independent scalar FP64 sums of the specified F32 products. F32 graph
        // operations follow the reductions; normalization is per stream.
        double hh=0,kk=0,dot=0;
        for(size_t d=0;d<dim;++d) {
            const size_t i=c*dim+d; const float h=bf(x[i]),k=kv[i];
            const float product=kernels::f32_from_bf16(w.q[i])*kernels::f32_from_bf16(w.k[i]);
            hh+=float(h*h); kk+=float(k*k); dot+=float(float(h*product)*k);
        }
        const float rstd=(1.f/std::sqrt(float(hh/dim)+1e-20f))*(1.f/std::sqrt(float(kk/dim)+1e-20f));
        const float score=(float(dot)*rstd)*float(1/std::sqrt(double(dim)));
        r.gate[c]=live?1.f/(1.f+std::exp(-std::copysign(std::sqrt(std::max(std::abs(score),1e-6f)),score))):0.f;
        for(size_t d=0;d<dim;++d) r.output[c*dim+d]=bf(bf(x[c*dim+d])+float(r.gate[c]*kv[4*dim+d]));
    }
    return r;
}
struct Buffers {
    core::DeviceArena arena{lookup_size*4+residual_size*8+lookup_size*2};
    float* lookup=static_cast<float*>(arena.alloc(lookup_size*4));
    float* x=static_cast<float*>(arena.alloc(residual_size*4));
    float* y=static_cast<float*>(arena.alloc(residual_size*4));
    uint16_t* quantized=static_cast<uint16_t*>(arena.alloc(lookup_size*2));
};
void arguments(CudaEngram& block,Buffers& b,EngramProjectionView view,cudaStream_t s) {
    const auto before=copy(b.y,residual_size,s); Stream other;
    rejects("default stream",[&]{block.forward(b.lookup,b.x,b.y,nullptr);});
    rejects("switch stream",[&]{block.forward(b.lookup,b.x,b.y,other.s);});
    rejects("null input",[&]{block.forward(nullptr,b.x,b.y,s);});
    rejects("misaligned output",[&]{block.forward(b.lookup,b.x,reinterpret_cast<float*>(reinterpret_cast<char*>(b.y)+1),s);});
    rejects("overflowing pointer range",[&]{block.forward(reinterpret_cast<float*>(UINTPTR_MAX-3),b.x,b.y,s);});
    rejects("lookup/residual overlap",[&]{block.forward(b.x,b.x,b.y,s);});
    rejects("lookup/output overlap",[&]{block.forward(b.y,b.x,b.y,s);});
    rejects("residual/output overlap",[&]{block.forward(b.lookup,b.x,b.x,s);});
    rejects("lookup into owned arena",[&]{block.forward(view.kv,b.x,b.y,s);});
    rejects("residual into owned arena",[&]{block.forward(b.lookup,view.kv,b.y,s);});
    rejects("output into owned arena",[&]{block.forward(b.lookup,b.x,const_cast<float*>(view.kv),s);});
    check(same(copy(b.y,residual_size,s),before),"argument errors preserve caller output");
    block.forward(b.lookup,b.x,b.y,s);
    check(same(copy(b.y,residual_size,s),before),"argument errors preserve healthy execution and replay");
}
void synthetic(int layer) {
    std::printf("synthetic_engram_layer=%d\n",layer);
    std::vector<uint8_t> values(kv_size*lookup_size,0),scales((kv_size/32)*(lookup_size/32));
    std::vector<uint16_t> q(residual_size),k(residual_size);
    const std::array<float,4> qw=layer==1?std::array<float,4>{.25f,-.5f,2.f,4.f}:std::array<float,4>{-.75f,1.5f,-.25f,.5f};
    for(size_t i=0;i<residual_size;++i){q[i]=kernels::bf16_from_f32(qw[i/dim]);k[i]=kernels::bf16_from_f32(i%2?.5f:1.f);}
    for(size_t r=0;r<kv_size/32;++r)for(size_t c=0;c<lookup_size/32;++c)scales[r*(lookup_size/32)+c]=uint8_t(119+(r+c)%5);
    for(size_t r=0;r<kv_size;++r){values[r*lookup_size+r%lookup_size]=0x38;values[r*lookup_size+(r*17+31)%lookup_size]=r%2?0xb0:0x30;}
    EngramWeights w{{values,scales,int(kv_size),int(lookup_size)},q,k,layer};
    auto unsupported=w;unsupported.layer=2;
    rejects("unsupported original layer before allocation",[&]{CudaEngram invalid(unsupported);});
    auto bad=w;bad.wkv.rows=20480;
    rejects("wrong projection dimensions before allocation",[&]{CudaEngram invalid(bad);});
    bad=w;bad.wkv.scales={}; rejects("missing projection scales",[&]{CudaEngram invalid(bad);});
    bad=w;bad.q=std::span(q).first(1);rejects("wrong normalization dimensions",[&]{CudaEngram invalid(bad);});
    auto saved=q.back();q.back()=0x7fc0;rejects("nonfinite q/k product",[&]{CudaEngram invalid(w);});q.back()=saved;
    values.back()=0x7f;rejects("invalid FP8 in final startup chunk",[&]{CudaEngram invalid(w);});values.back()=0;
    CudaEngram block(w);Stream stream;Buffers b;
    check(block.layer()==layer,"CUDA projection retains original weight identity");
    check(block.device_bytes()==314769424,"fixed 300 MiB projection plus bounded product and scratch");
    std::vector<float> lookup(lookup_size),x(residual_size);
    for(int test=0;test<5;++test){
        for(size_t i=0;i<lookup.size();++i)lookup[i]=test==3?0.f:float(int(i%7)-3)*.25f;
        std::vector<float> expected_kv(kv_size);
        for(size_t r=0;r<kv_size;++r){
            const size_t c0=r%lookup_size,c1=(r*17+31)%lookup_size;
            const float a=std::ldexp(lookup[c0],int((r/32+c0/32)%5)-8);
            const float v=std::ldexp((r%2?-.5f:.5f)*lookup[c1],int((r/32+c1/32)%5)-8);
            expected_kv[r]=bf(a+v);
        }
        for(size_t i=0;i<x.size();++i)x[i]=test==1?0.f:test==2?bf(expected_kv[i]*(i/dim==2?-8.f:2.f)):bf(std::sin(float(i)*.13f)*std::ldexp(1.f,int(i/dim)*3-3));
        const bool live=test!=4; const auto expected=gate_reference(x,expected_kv,w,live);
        upload(b.lookup,lookup,stream.s);upload(b.x,x,stream.s);
        const auto view=block.forward(b.lookup,b.x,b.y,stream.s,live);
        const auto kv=copy(view.kv,kv_size,stream.s),gate=copy(view.gate,4,stream.s),out=copy(b.y,residual_size,stream.s);
        check(same(kv,expected_kv),"synthetic projection exact case="+std::to_string(test));
        compare(gate,expected.gate,"synthetic-gate-"+std::to_string(test),2);
        compare(out,expected.output,"synthetic-output-"+std::to_string(test));
        check(std::all_of(out.begin(),out.end(),[](float v){return v==bf(v);}),"output restores BF16 boundary");
        if(test==1||test==3)check(std::all_of(gate.begin(),gate.end(),[](float g){return std::abs(g-.50025f)<2e-7f;}),"zero score uses copysign with sqrt floor, not sign(0)");
        if(!live)check(same(out,x)&&std::all_of(gate.begin(),gate.end(),[](float g){return g==0;}),"dead token suppresses all four updates");
        if(test==0)arguments(block,b,view,stream.s);
    }
    if(layer==14) {
        // Two resident projections with distinct q weights and explicit streams.
        auto q1=q;for(auto& v:q1)v=kernels::bf16_from_f32(-kernels::f32_from_bf16(v));
        const Layer1EngramWeights w1{w.wkv,q1,k};CudaLayer1Engram first(w1);Stream s1;Buffers b1;
        check(first.layer()==1 && first.device_bytes()==block.device_bytes(),"legacy aggregate defaults to layer 1 with identical allocation budget");
        const auto view14=block.forward(b.lookup,b.x,b.y,stream.s);
        const auto out14=copy(b.y,residual_size,stream.s),kv14=copy(view14.kv,kv_size,stream.s),gate14=copy(view14.gate,4,stream.s);
        upload(b1.lookup,lookup,s1.s);upload(b1.x,x,s1.s);
        const auto view1=first.forward(b1.lookup,b1.x,b1.y,s1.s);
        const auto out1=copy(b1.y,residual_size,s1.s),gate1=copy(view1.gate,4,s1.s);
        const auto expected1=gate_reference(x,kv14,w1,true);
        compare(gate1,expected1.gate,"independent-layer1-gate",2);
        compare(out1,expected1.output,"independent-layer1-output");
        check(view1.kv!=view14.kv && view1.gate!=view14.gate && !same(gate1,gate14) && !same(out1,out14),"resident layers own distinct weights, scratch and results");
        check(same(copy(view14.kv,kv_size,stream.s),kv14) && same(copy(view14.gate,4,stream.s),gate14),"layer-1 forward preserves layer-14 projection and gate views");
        block.forward(b.lookup,b.x,b.y,stream.s);
        check(same(copy(b.y,residual_size,stream.s),out14),"interleaved layer-14 replay retains its weights and stream");
    }
}
template<class T> std::vector<T> tensor(const GgufFile& f,const char* name,uint32_t type,size_t n){
    const auto* t=f.find(name);
    if(!t||t->type!=type||tensor_payload_bytes(*t)!=n*sizeof(T)||t->offset>f.file_size()-f.data_start()||n*sizeof(T)>f.file_size()-f.data_start()-t->offset)
        throw std::runtime_error(std::string("Malformed fixture tensor ")+name);
    std::vector<T> v(n);std::memcpy(v.data(),f.tensor_data(*t),v.size()*sizeof(T));return v;
}
void real(const char* projection,const char* reference,const char* table_path,int layer){
    const auto model=GgufModel::open(projection);const auto weights=read_engram_weights(model,layer);const auto& f=model.meta();
    GgufFile lookup_file(reference);const auto map=read_engram_token_map(lookup_file,layer);
    const auto tokens=tensor<int32_t>(f,"tokens",24,9);const auto live=tensor<uint8_t>(f,"live",24,9);const auto rows=tensor<uint32_t>(f,"rows",24,9*24);
    const auto lookup=tensor<float>(f,"input.lookup",0,9*lookup_size),input=tensor<float>(f,"input.residual",0,9*residual_size);
    const auto expected_kv=tensor<float>(f,"expected.kv",0,9*kv_size),expected_gate=tensor<float>(f,"expected.gate",0,9*4),expected_out=tensor<float>(f,"expected.output",0,9*residual_size);
    const auto expected_quant=tensor<uint16_t>(f,"expected.quantized",30,9*lookup_size);
    CudaEngram block(weights);Stream stream;Buffers b;std::vector<float> first_output;bool oracle_bits=true;
    check(weights.layer==layer && block.layer()==layer,"original real layer identity retained through CUDA construction");
    std::printf("engram_layer=%d engram_device_bytes=%zu\n",layer,block.device_bytes());
    for(uint64_t capacity:{uint64_t(4096),uint64_t(0)}){
        EngramTable table(layer,table_path,{8,capacity,true});EngramHash hash(layer,map);
        for(int pass=0;pass<2;++pass){
            hash.reset();std::vector<uint32_t> actual_rows(9*24);std::vector<float> all_output;
            const size_t prefill=pass?9:3;
            for(size_t pos=0;pos<9;){
                const size_t n=pos?1:prefill;hash.forward(std::span(tokens).subspan(pos,n),std::span(actual_rows).subspan(pos*24,n*24),int(pos),std::span(live).subspan(pos,n));
                for(size_t t=pos;t<pos+n;++t){
                    std::array<uint16_t,lookup_size> bits{};table.gather(std::span(actual_rows).subspan(t*24,24),bits);
                    std::vector<float> restored(lookup_size);for(size_t i=0;i<lookup_size;++i)restored[i]=kernels::f32_from_bf16(bits[i]);
                    check(same(restored,std::span(lookup).subspan(t*lookup_size,lookup_size)),"actual Strata row lookup matches original embed");
                    const auto x=std::span(input).subspan(t*residual_size,residual_size);
                    upload(b.lookup,restored,stream.s);upload(b.x,x,stream.s);
                    quantize_fp8_bf16(b.lookup,b.quantized,int(lookup_size),stream.s);
                    std::array<uint16_t,lookup_size> quant{};ck(cudaMemcpyAsync(quant.data(),b.quantized,lookup_size*2,cudaMemcpyDeviceToHost,stream.s));ck(cudaStreamSynchronize(stream.s));
                    check(std::equal(quant.begin(),quant.end(),expected_quant.begin()+ptrdiff_t(t*lookup_size)),"real FP8 activation codes/scales restore exactly");
                    const auto view=block.forward(b.lookup,b.x,b.y,stream.s,live[t]!=0);
                    const auto kv=copy(view.kv,kv_size,stream.s),gate=copy(view.gate,4,stream.s),out=copy(b.y,residual_size,stream.s);
                    const auto label=std::to_string(capacity)+"-"+std::to_string(pass)+"-"+std::to_string(t);
                    compare(kv,std::span(expected_kv).subspan(t*kv_size,kv_size),"real-kv-"+label);
                    compare(gate,std::span(expected_gate).subspan(t*4,4),"real-gate-"+label,1);
                    compare(out,std::span(expected_out).subspan(t*residual_size,residual_size),"real-output-"+label);
                    oracle_bits&=same(out,std::span(expected_out).subspan(t*residual_size,residual_size));
                    const auto same_input=gate_reference(x,kv,weights,live[t]!=0);
                    compare(gate,same_input.gate,"same-kv-gate-"+label,2);
                    compare(out,same_input.output,"same-kv-output-"+label);
                    if(!live[t])check(same(out,x),"masked real residual passes through exactly");
                    all_output.insert(all_output.end(),out.begin(),out.end());
                }
                pos+=n;
            }
            check(actual_rows==rows,"original hash IDs through prefill/decode/reset");
            if(first_output.empty())first_output=all_output;else check(same(first_output,all_output),"cache eviction, chunking and reset preserve full Engram result bits");
        }
    }
    std::printf("real_output_reference_bitwise=%d\n",int(oracle_bits));
}
}
int main(int argc,char** argv){
    int layer=0,arg=1;
    if(argc>1&&std::string(argv[1])=="--layer"){
        if(argc<3||(std::string(argv[2])!="1"&&std::string(argv[2])!="14"))return 2;
        layer=std::string(argv[2])=="1"?1:14;arg=3;
    }
    if(!(argc==arg||(argc==arg+4&&std::string(argv[arg])=="--real"))){std::fprintf(stderr,"usage: deepseek41_engram_cuda_test [--layer 1|14] [--real PROJECTION.gguf LOOKUP_REFERENCE.gguf TABLE.gguf]\n");return 2;}
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    try{
        ck(cudaSetDevice(0));
        if(argc==arg){for(int original:{1,14})if(!layer||layer==original)synthetic(original);}
        else real(argv[arg+1],argv[arg+2],argv[arg+3],layer?layer:1);
    }
    catch(const std::exception& e){std::fprintf(stderr,"ERROR %s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
