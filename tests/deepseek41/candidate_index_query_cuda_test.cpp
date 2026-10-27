#include "strata/deepseek41/candidate_index_query_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>
using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int hidden=5120,qr_dim=1280,dim=128,heads=32;
int checks=0,failures=0,index_layer=24;
void check(bool yes,const std::string& label){++checks;failures+=!yes;std::printf("%s %s\n",yes?"ok  ":"FAIL",label.c_str());}
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void rejects(const std::string& label,const std::function<void()>& f){try{f();check(false,label);}catch(const std::invalid_argument&){check(true,label);}}
struct Stream{cudaStream_t value{};Stream(){ck(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(value);}};
template<class T>void upload(T* p,const std::vector<T>& x,cudaStream_t s){ck(cudaMemcpyAsync(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice,s));ck(cudaStreamSynchronize(s));}
template<class T>std::vector<T> download(const T* p,size_t n,cudaStream_t s){std::vector<T> out(n);ck(cudaMemcpyAsync(out.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return out;}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, size_t count) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != (sizeof(T) == 2 ? 30u : 0u) || tensor_payload_bytes(*t) != count * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Malformed index query fixture tensor " + name);
    const auto* data = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*t)); return {data, data + count};
}
int number(const GgufModel& model, const std::string& name, int maximum) {
    const auto* m = model.meta().get(name);
    if (!m || m->type != MetaType::U32 || m->u > uint64_t(maximum)) throw std::runtime_error("Malformed fixture number " + name);
    return int(m->u);
}
template<class T> void exact_values(const std::vector<T>& actual,const std::vector<T>& expected,const std::string& label){
    if(actual.size()!=expected.size())throw std::runtime_error("Reference shape mismatch");
    size_t different=0;
    for(size_t i=0;i<actual.size();++i)if(actual[i]!=expected[i]){
        if(different<8)std::printf("    mismatch index=%zu actual=%.9g expected=%.9g\n",i,double(actual[i]),double(expected[i]));
        ++different;
    }
    std::printf("  %s different=%zu/%zu\n",label.c_str(),different,actual.size());check(different==0,label);
}
float bf(float x){return kernels::f32_from_bf16(kernels::bf16_from_f32(x));}
std::vector<int32_t> select(const std::vector<float>& scores,int visible,int offset){
    std::vector<int32_t> ids(scores.size());std::iota(ids.begin(),ids.end(),0);
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return scores[size_t(a)]>scores[size_t(b)];});
    ids.resize(std::min(size_t(512),ids.size()));std::sort(ids.begin(),ids.end());
    for(auto& id:ids)id=id<visible && std::isfinite(scores[size_t(id)])?id+offset:-1;
    return ids;
}
Source20Batch prefill(int n){return {0,n,0,n,n};}
std::vector<int32_t> blocks(int width,int visible){
    const int available=(visible+7)/8,selected=std::min(2048,available);
    std::vector<int32_t> result(size_t(std::min(2048,(width+7)/8)),-1);
    // Discard earliest blocks at the cap, deliberately excluding the strongest
    // synthetic candidate source scores. The newest visible block remains selected.
    std::iota(result.begin(),result.begin()+selected,available-selected);return result;
}
static_assert(!std::is_constructible_v<CudaCandidateIndexQuery,Source20IndexQueryWeights,int>);
static_assert(!std::is_constructible_v<CudaSource20IndexQuery,CandidateIndexQueryWeights,int>);
void synthetic(){
    constexpr int cap=16401,guard=64;
    std::vector<uint8_t> codes(4096*qr_dim,0),scales(128*40,127);
    for(int h=0;h<heads;++h)codes[size_t(h)*dim*qr_dim]=0x38;
    std::vector<uint16_t> ww(heads*hidden,0);ww[0]=0x3f80;
    CandidateIndexQueryWeights w{{codes,scales,4096,qr_dim},ww,index_layer};
    for(int n:{0,-1,1048577})rejects("invalid capacity",[&]{CudaCandidateIndexQuery bad(w,n);});
    rejects("MTP layer40 cannot be a backbone index source",[&]{auto q=w;q.layer=40;CudaCandidateIndexQuery bad(q,1);});
    rejects("source20 weights cannot masquerade as candidate source",[&]{auto q=w;q.layer=20;CudaCandidateIndexQuery bad(q,1);});
    rejects("missing query scales",[&]{auto q=w;q.query.scales={};CudaCandidateIndexQuery bad(q,1);});
    rejects("missing head weights",[&]{auto q=w;q.head_weights={};CudaCandidateIndexQuery bad(q,1);});
    CudaCandidateIndexQuery index(w,cap);Stream stream,other;
    core::DeviceArena arena((hidden+qr_dim+2*guard)*4+size_t(cap*dim+2*guard)*2+2304*4+2048);
    auto* x=static_cast<float*>(arena.alloc(hidden*4));auto* qr=static_cast<float*>(arena.alloc(qr_dim*4));
    auto* key_storage=static_cast<uint16_t*>(arena.alloc(size_t(cap*dim+2*guard)*2));auto* keys=key_storage+guard;
    auto* mask_storage=static_cast<int32_t*>(arena.alloc((2048+2*guard)*4));auto* mask=mask_storage+guard;
    upload(key_storage,std::vector<uint16_t>(cap*dim+2*guard,0x5a5a),stream.value);
    upload(mask_storage,std::vector<int32_t>(2048+2*guard,-77),stream.value);
    std::vector<uint16_t> history(size_t(cap)*dim,0);
    for(int i=0;i<cap;++i)history[size_t(i)*dim]=kernels::bf16_from_f32(i<24?4096.f:float(i%257-17));
    upload(keys,history,stream.value);std::vector<float> xx(hidden,0),qq(qr_dim,0);qq[0]=1;upload(qr,qq,stream.value);
    const auto bytes=index.device_bytes(),workspace=index.sort_workspace_bytes();
    const std::vector<std::pair<int,int>> cases{{1,1},{9,1},{9,8},{9,9},{513,512},{513,513},{16384,16384},{16385,16385},{16393,9},{16393,16385},{16393,16393},{cap,cap}};
    int pruned=0;
    for(const auto& [width,visible]:cases){
        for(float sign: width==9?std::vector<float>{1,-1,0}:std::vector<float>{1}){
            xx[0]=64*sign;upload(x,xx,stream.value);const auto row=blocks(width,visible);upload(mask,row,stream.value);
            const auto r=index.forward(x,qr,keys,prefill(width),visible-1,{mask,int(row.size())},128,stream.value);
            std::vector<float> expected(size_t(width),-INFINITY);
            for(int i=0;i<visible;++i)if(std::find(row.begin(),row.end(),i/8)!=row.end())expected[size_t(i)]=bf(std::max(0.f,kernels::f32_from_bf16(history[size_t(i)*dim]))*sign);
            const auto ids=download(r.indices,size_t(r.count),stream.value);
            check(download(r.scores,size_t(width),stream.value)==expected,"analytical masked scores width="+std::to_string(width)+" visible="+std::to_string(visible));
            check(r.count==std::min(512,width) && ids==select(expected,visible,128),"independent canonical top512 and causal padding");
            check(download(mask,row.size(),stream.value)==row,"source candidate row unchanged");
            if(visible>16384){check(std::find(ids.begin(),ids.end(),128)==ids.end(),"higher-scoring excluded block never enters selection");++pruned;}
        }
    }
    check(pruned==4,"four cap-crossing queries exercised actual pruning");
    const auto call=prefill(cap);const auto row=blocks(cap,cap);upload(mask,row,stream.value);
    const auto saved=index.forward(x,qr,keys,call,cap-1,{mask,int(row.size())},0,stream.value);
    const auto ids=download(saved.indices,size_t(saved.count),stream.value);const auto scores=download(saved.scores,cap,stream.value);
    const auto queries=download(index.query(),4096,stream.value);const auto weights=download(index.head_weights(),32,stream.value);
    auto run=[&](Source20CandidateSelection candidates){index.forward(x,qr,keys,call,cap-1,candidates,0,stream.value);};
    rejects("null candidates",[&]{run({nullptr,2048});});rejects("wrong candidate stride",[&]{run({mask,2047});});
    rejects("unaligned candidates",[&]{run({reinterpret_cast<int32_t*>(reinterpret_cast<char*>(mask)+1),2048});});
    rejects("candidate address overflow",[&]{run({reinterpret_cast<int32_t*>(UINTPTR_MAX-3),2048});});
    rejects("own output cannot be candidates",[&]{run({saved.indices,2048});});
    for(int bad_case=0;bad_case<6;++bad_case){auto bad=row;
        if(bad_case==0)bad[0]=-2;
        if(bad_case==1)bad[1]=bad[0];
        if(bad_case==2)std::swap(bad[0],bad[1]);
        if(bad_case==3)bad.back()=-1;
        if(bad_case==4)bad.back()=cap;
        if(bad_case==5)bad[7]=-1;
        upload(mask,bad,stream.value);rejects("malformed source row rejected before result mutation",[&]{run({mask,2048});});
    }
    upload(mask,row,stream.value);
    rejects("key count exceeds capacity",[&]{index.forward(x,qr,keys,prefill(cap+1),0,{mask,2048},0,stream.value);});
    rejects("bad row",[&]{index.forward(x,qr,keys,call,-1,{mask,2048},0,stream.value);});
    rejects("offset overflow",[&]{index.forward(x,qr,keys,call,cap-1,{mask,2048},INT_MAX,stream.value);});
    rejects("null input",[&]{index.forward(nullptr,qr,keys,call,cap-1,{mask,2048},0,stream.value);});
    rejects("null keys",[&]{index.forward(x,qr,nullptr,call,cap-1,{mask,2048},0,stream.value);});
    rejects("foreign stream",[&]{index.forward(x,qr,keys,call,cap-1,{mask,2048},0,other.value);});
    check(ids==download(saved.indices,ids.size(),stream.value) && scores==download(saved.scores,scores.size(),stream.value) && queries==download(index.query(),queries.size(),stream.value) && weights==download(index.head_weights(),weights.size(),stream.value),"all rejected calls preserve results and diagnostics");
    check(history==download(keys,history.size(),stream.value) && xx==download(x,xx.size(),stream.value) && qq==download(qr,qq.size(),stream.value),"all external inputs unchanged");
    check(download(key_storage,guard,stream.value)==std::vector<uint16_t>(guard,0x5a5a) && download(keys+history.size(),guard,stream.value)==std::vector<uint16_t>(guard,0x5a5a) && download(mask_storage,guard,stream.value)==std::vector<int32_t>(guard,-77) && download(mask+2048,guard,stream.value)==std::vector<int32_t>(guard,-77),"allocation guards intact");
    check(index.device_bytes()==bytes && index.sort_workspace_bytes()==workspace,"fixed startup allocations");
    std::printf("candidate_query_capacity=%d device_bytes=%zu sort_workspace_bytes=%zu\n",cap,bytes,workspace);
}
// Exact BF16 inputs put each dot one FP32 ULP to either side of a BF16
// midpoint after cancellation. This is independent of the checkpoint fixture.
void head_projection_rounding(){
    std::vector<uint8_t> codes(4096*qr_dim,0),scales(128*40,127);
    std::vector<uint16_t> ww(heads*hidden,0);std::vector<float> expected(heads);
    for(int h=0;h<heads;++h){
        const float sign=(h&1)?-1.f:1.f,side=(h&2)?1.f:-1.f;
        const float terms[]{128.f,1.f,3.f/256.f,side*0x1p-23f,-128.f};
        for(int i=0;i<5;++i)ww[size_t(h)*hidden+size_t(i)*32]=kernels::bf16_from_f32(sign*terms[i]);
        expected[size_t(h)]=sign*((h&2)?1.015625f:1.0078125f)/64.f;
    }
    CudaCandidateIndexQuery index({{codes,scales,4096,qr_dim},ww,index_layer},1);Stream stream;
    core::DeviceArena arena((hidden+qr_dim)*4+dim*2+1024);
    auto* x=static_cast<float*>(arena.alloc(hidden*4));auto* qr=static_cast<float*>(arena.alloc(qr_dim*4));
    auto* keys=static_cast<uint16_t*>(arena.alloc(dim*2));auto* mask=static_cast<int32_t*>(arena.alloc(4));
    upload(x,std::vector<float>(hidden,1.f),stream.value);upload(qr,std::vector<float>(qr_dim,0.f),stream.value);
    upload(keys,std::vector<uint16_t>(dim,0),stream.value);upload(mask,std::vector<int32_t>{0},stream.value);
    index.forward(x,qr,keys,prefill(1),0,{mask,1},0,stream.value);
    exact_values(download(index.head_weights(),heads,stream.value),expected,
                 "signed cancellation on both sides of BF16 head-weight midpoint");
}
std::string string_value(const GgufModel& model, const std::string& name) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::STRING) throw std::runtime_error("Missing fixture string " + name);
    return value->s;
}
void real(const char* path){
    const auto model=GgufModel::open(path);
    if(string_value(model,"general.architecture")!="deepseek41-candidate-index-query-fixture" ||
       string_value(model,"strata.deepseek41.revision")!="1bae0e1088993b9b82de08f59de5be5a34e42170" ||
       string_value(model,"strata.deepseek41.official_source_sha256")!="e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d" ||
       string_value(model,"strata.deepseek41.official_kernel_sha256")!="1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455" ||
       string_value(model,"fixture.linear_oracle")!="fp64" ||
       string_value(model,"fixture.source20_sha256")!="961541ba7c7f02926018acddd067f4815076838f23cbb6a6eecdc56e16077a1c")
        throw std::runtime_error("Wrong candidate index reference identity");
    Geometry g;g.layers=40;g.hidden=hidden;g.compress_ratios.resize(40);g.sources.resize(40);
    g.compress_ratios[20]=g.compress_ratios[size_t(index_layer)]=1;g.sources[20]={20,20,-1};g.sources[size_t(index_layer)]={20,index_layer,20};
    constexpr int capacity=16393,max_tokens=17,query_size=heads*dim;
    CudaCandidateIndexQuery index(read_candidate_index_query_weights(model,g,index_layer),capacity);Stream stream;
    core::DeviceArena arena(size_t(max_tokens)*(hidden+qr_dim+2048)*4+size_t(capacity)*dim*2+2048);
    auto* x=static_cast<float*>(arena.alloc(size_t(max_tokens)*hidden*4));
    auto* qr=static_cast<float*>(arena.alloc(size_t(max_tokens)*qr_dim*4));
    auto* keys=static_cast<uint16_t*>(arena.alloc(size_t(capacity)*dim*2));
    auto* candidates=static_cast<int32_t*>(arena.alloc(size_t(max_tokens)*2048*4));
    const auto bytes=index.device_bytes(),workspace=index.sort_workspace_bytes();
    const int cases=number(model,"fixture.cases",11);if(cases!=11)throw std::runtime_error("Missing reference cases");
    int rows=0,pruned=0;
    for(int i=0;i<cases;++i){
        const auto prefix="case."+std::to_string(i)+".";
        const int start=number(model,prefix+"start",capacity-1),tokens=number(model,prefix+"tokens",max_tokens);
        const int offset=number(model,prefix+"offset",128),count=start+tokens;
        if(tokens<1 || count>capacity)throw std::runtime_error("Invalid reference plan");
        const Source20Batch batch(start,tokens,start,count,tokens);
        const int stride=std::min(512,count),candidate_stride=batch.candidate_stride();
        const auto input=tensor<float>(model,prefix+"x",size_t(tokens)*hidden);
        const auto lowrank=tensor<float>(model,prefix+"qr",size_t(tokens)*qr_dim);
        const auto history=tensor<uint16_t>(model,prefix+"keys",size_t(count)*dim);
        const auto block_floats=tensor<float>(model,prefix+"candidates",size_t(tokens)*candidate_stride);
        std::vector<int32_t> blocks;blocks.reserve(block_floats.size());
        for(float v:block_floats){
            if(!std::isfinite(v) || v < -1 || v >= (count+7)/8 || v!=std::floor(v))throw std::runtime_error("Invalid reference candidate");
            blocks.push_back(int32_t(v));
        }
        const auto queries=tensor<uint16_t>(model,prefix+"query",size_t(tokens)*query_size);
        const auto weights=tensor<float>(model,prefix+"weights",size_t(tokens)*heads);
        const auto scores=tensor<float>(model,prefix+"scores",size_t(tokens)*count);
        const auto reference_ids=tensor<float>(model,prefix+"indices",size_t(tokens)*stride);
        upload(x,input,stream.value);upload(qr,lowrank,stream.value);upload(keys,history,stream.value);upload(candidates,blocks,stream.value);
        for(int row=0;row<tokens;++row){
            const auto label=prefix+"row."+std::to_string(row)+".";
            const auto result=index.forward(x+size_t(row)*hidden,qr+size_t(row)*qr_dim,keys,batch,row,
                                           {candidates+size_t(row)*candidate_stride,candidate_stride},offset,stream.value);
            exact_values(download(index.query(),query_size,stream.value),std::vector<uint16_t>(queries.begin()+size_t(row)*query_size,queries.begin()+size_t(row+1)*query_size),label+"FP4 query bits");
            exact_values(download(index.head_weights(),heads,stream.value),std::vector<float>(weights.begin()+row*heads,weights.begin()+(row+1)*heads),label+"head weights");
            const std::vector<float> expected(scores.begin()+size_t(row)*count,scores.begin()+size_t(row+1)*count);
            exact_values(download(result.scores,count,stream.value),expected,label+"candidate-masked causal scores");
            const auto actual=download(result.indices,size_t(result.count),stream.value);
            check(result.count==stride && actual==select(expected,batch.visible_keys(row),offset),label+"independent canonical top512");
            std::vector<float> actual_scores,torch_scores;bool membership=true;
            const auto first=blocks.begin()+size_t(row)*candidate_stride,last=first+candidate_stride;
            for(int id:actual)if(id!=-1){
                const int position=id-offset;
                if(position<0 || position>=batch.visible_keys(row) || std::find(first,last,position/8)==last)membership=false;
                else actual_scores.push_back(expected[size_t(position)]);
            }
            for(int j=0;j<stride;++j){
                const float v=reference_ids[size_t(row)*stride+j];
                if(v==-1)continue;
                if(!std::isfinite(v) || v<offset || v>=count+offset || v!=std::floor(v))throw std::runtime_error("Invalid reference index");
                torch_scores.push_back(expected[size_t(int(v)-offset)]);
            }
            std::sort(actual_scores.begin(),actual_scores.end());std::sort(torch_scores.begin(),torch_scores.end());
            check(membership && actual_scores==torch_scores,label+"Torch top512 score membership including ties");++rows;
        }
        check(input==download(x,input.size(),stream.value) && lowrank==download(qr,lowrank.size(),stream.value) &&
              history==download(keys,history.size(),stream.value) && blocks==download(candidates,blocks.size(),stream.value),prefix+"all borrowed inputs preserved");
        pruned+=count>16384;
    }
    check(rows==27 && rows==number(model,"fixture.query_rows",27) && pruned==2,"27 independent reference rows including two actual pruning cases");
    check(index.device_bytes()==bytes && index.sort_workspace_bytes()==workspace,"fixed allocations across all original-weight reference queries");
    std::printf("official_candidate_query_rows=%d pruning_cases=%d device_bytes=%zu sort_workspace_bytes=%zu\n",rows,pruned,bytes,workspace);
}

}
int main(int argc,char** argv){
    if(argc>=3 && std::string(argv[1])=="--layer"){
        const std::string value=argv[2];if(value!="24" && value!="28" && value!="32" && value!="36")return 2;
        index_layer=std::stoi(value);argc-=2;argv+=2;
    }
    if(!(argc==1 || (argc==3 && std::string(argv[1])=="--real")))return 2;
    int n=0;const auto status=cudaGetDeviceCount(&n);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess&&!n))return 77;
    try{ck(status);ck(cudaSetDevice(0));if(argc==1){synthetic();head_projection_rounding();}else real(argv[2]);}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
