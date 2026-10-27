#include "strata/deepseek41/source20_kv_cuda.hpp"
#include "strata/deepseek41/source20_index_key_cuda.hpp"
#include "strata/deepseek41/source20_compressor_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int dim = 128, latent_dim = 512, hidden = 5120, total = 17, guard = 64;
int checks = 0, failures = 0;
size_t compared = 0, mismatched = 0;
void check(bool ok, const std::string& name) {
    ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void ck(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void rejects(const std::string& name, const std::function<void()>& fn) {
    try { fn(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(value); }
};
template<class T> void upload(T* pointer, const std::vector<T>& values, cudaStream_t stream) {
    ck(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
}
template<class T> std::vector<T> download(const T* pointer, size_t count, cudaStream_t stream) {
    if (count == 0) return {};
    std::vector<T> values(count);
    ck(cudaMemcpyAsync(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return values;
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
void exact(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected, const std::string& name) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    size_t different = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
        if (different < 8) std::printf("  mismatch index=%zu actual=%04x expected=%04x\n", i, unsigned(actual[i]), unsigned(expected[i]));
        ++different;
    }
    compared += actual.size(); mismatched += different;
    std::printf("  %s differing=%zu/%zu\n", name.c_str(), different, actual.size());
    check(different == 0, name);
}
static_assert(!std::is_convertible_v<Source20Batch, CompressedBatch>);
static_assert(!std::is_constructible_v<CudaIndexKeyCache, Source20IndexKeyWeights, int>);
static_assert(!std::is_constructible_v<CudaSource20IndexKeyCache, IndexKeyWeights, int>);
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
float nearest(float x) {
    const float grid[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    int best = 0;
    for (int i = 1; i < 8; ++i)
        if (std::abs(x-grid[i]) < std::abs(x-grid[best]) ||
            (std::abs(x-grid[i]) == std::abs(x-grid[best]) && i%2 == 0)) best=i;
    return grid[best];
}
// Independent scalar E8M0/FP4 reference. Synthetic tails are zero; the pinned
// official CPU RoPE checks nonzero tails, odd/even positions and long contexts.
std::vector<uint16_t> reference(std::vector<float> values, int start = 0) {
    for (size_t row=0;row<values.size()/dim;++row)
        for (int pair=0;pair<32;++pair) {
            const float raw=1.f/float(std::pow(160000.,double(pair)/32.));
            const float smooth=1.f-std::clamp(float(pair-15)/10.f,0.f,1.f);
            const float frequency=(raw/16.f)*(1.f-smooth)+raw*smooth;
            const float angle=float(start+int(row))*frequency;
            const float c=std::cos(angle),s=std::sin(angle);
            const size_t at=row*dim+64+pair*2;
            const float a=bf(values[at]),b=bf(values[at+1]);
            values[at]=bf(a*c-b*s);values[at+1]=bf(a*s+b*c);
        }
    std::vector<uint16_t> result(values.size());
    for (size_t offset=0;offset<values.size();offset+=32) {
        float maximum=6.f*std::ldexp(1.f,-126);
        for (size_t j=0;j<32;++j) maximum=std::max(maximum,std::abs(bf(values[offset+j])));
        int exponent=0;const float fraction=std::frexp(maximum*(1.f/6.f),&exponent);
        const float scale=std::ldexp(1.f,exponent-int(fraction==.5f));
        for (size_t j=0;j<32;++j) {
            const float x=bf(values[offset+j]);
            result[offset+j]=kernels::bf16_from_f32(std::copysign(nearest(std::min(std::abs(x)/scale,6.f)),x)*scale);
        }
    }
    return result;
}
struct Buffers {
    Stream stream;
    core::DeviceArena arena{size_t(total*hidden+2*guard)*4+size_t(total*latent_dim+2*guard)*4+
                            size_t(total*dim+2*guard)*6};
    float* x_storage=static_cast<float*>(arena.alloc(size_t(total*hidden+2*guard)*4));
    float* latent_storage=static_cast<float*>(arena.alloc(size_t(total*latent_dim+2*guard)*4));
    float* key_storage=static_cast<float*>(arena.alloc(size_t(total*dim+2*guard)*4));
    uint16_t* output_storage=static_cast<uint16_t*>(arena.alloc(size_t(total*dim+2*guard)*2));
    float* x=x_storage+guard;
    float* latent=latent_storage+guard;
    float* key=key_storage+guard;
    uint16_t* output=output_storage+guard;
    void clear() {
        upload(x_storage,std::vector<float>(total*hidden+2*guard,-333.f),stream.value);
        clear_latent();
        upload(key_storage,std::vector<float>(total*dim+2*guard,-123.f),stream.value);
        upload(output_storage,std::vector<uint16_t>(total*dim+2*guard,0x5a5a),stream.value);
    }
    void clear_latent() { upload(latent_storage,std::vector<float>(total*latent_dim+2*guard,-123.f),stream.value); }
    void guards(const float* storage,int rows,int width) {
        const size_t unused=size_t(total-rows)*width+guard;
        check(same(download(storage,guard,stream.value),std::vector<float>(guard,-123.f)) &&
              same(download(storage+guard+rows*width,unused,stream.value),std::vector<float>(unused,-123.f)),
              "input guards and unused tail preserved");
    }
    void output_guards(int rows) {
        const size_t unused=size_t(total-rows)*dim+guard;
        check(same(download(output_storage,guard,stream.value),std::vector<uint16_t>(guard,0x5a5a)) &&
              same(download(output+rows*dim,unused,stream.value),std::vector<uint16_t>(unused,0x5a5a)),
              "output guards and unused tail preserved");
    }
};
Source20Batch batch(int start,int count,int capacity=total) { return Source20Batch(start,count,start,capacity,capacity); }
void synthetic() {
    Buffers b;Stream other;
    const float values[]={0.f,-0.f,.25f,-.25f,.75f,-.75f,1.25f,-1.25f,1.75f,-1.75f,2.5f,-2.5f,3.5f,-3.5f,5.f,6.f};
    const float scales[]={1.f,1.0078125f,.9921875f,std::ldexp(1.f,-126),std::ldexp(1.f,-130),std::ldexp(1.f,117)};
    std::vector<float> input(total*dim,0.f);
    for (int row=0;row<total;++row)
        for (int column=0;column<64;++column) input[size_t(row)*dim+column]=values[column%16]*scales[(row+column/32)%6];
    for (const auto& item : {std::pair{0,17},std::pair{1,3},std::pair{7,3},std::pair{65535,3},std::pair{1048573,3},std::pair{1048575,1}}) {
        b.clear();const std::vector<float> part(input.begin(),input.begin()+item.second*dim);
        upload(b.key,part,b.stream.value);
        postprocess_source20_index_keys(b.key,b.output,item.second,item.first,b.stream.value);
        exact(download(b.output,part.size(),b.stream.value),reference(part,item.first),"scalar E8M0/FP4 position "+std::to_string(item.first));
        check(same(part,download(b.key,part.size(),b.stream.value)),"postprocess input preserved");
        b.guards(b.key_storage,item.second,dim);b.output_guards(item.second);
    }
    const auto saved=download(b.output_storage,total*dim+2*guard,b.stream.value);
    for (const auto& item : {std::pair{-1,1},std::pair{1048576,1},std::pair{1048575,2},std::pair{0,0},std::pair{0,-1},std::pair{0,1048577},std::pair{0,std::numeric_limits<int>::max()}})
        rejects("invalid position/count refused",[&]{postprocess_source20_index_keys(b.key,b.output,item.second,item.first,b.stream.value);});
    rejects("null postprocess input",[&]{postprocess_source20_index_keys(nullptr,b.output,1,0,b.stream.value);});
    rejects("null postprocess output",[&]{postprocess_source20_index_keys(b.key,nullptr,1,0,b.stream.value);});
    rejects("null postprocess stream",[&]{postprocess_source20_index_keys(b.key,b.output,1,0,nullptr);});
    rejects("unaligned postprocess input",[&]{postprocess_source20_index_keys(reinterpret_cast<float*>(reinterpret_cast<char*>(b.key)+1),b.output,1,0,b.stream.value);});
    rejects("unaligned postprocess output",[&]{postprocess_source20_index_keys(b.key,reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(b.output)+1),1,0,b.stream.value);});
    rejects("overlapping postprocess buffers",[&]{postprocess_source20_index_keys(b.key,reinterpret_cast<uint16_t*>(b.key),1,0,b.stream.value);});
    rejects("postprocess input address overflow",[&]{postprocess_source20_index_keys(reinterpret_cast<float*>(UINTPTR_MAX-3),b.output,1,0,b.stream.value);});
    rejects("postprocess output address overflow",[&]{postprocess_source20_index_keys(b.key,reinterpret_cast<uint16_t*>(UINTPTR_MAX-3),1,0,b.stream.value);});
    check(same(saved,download(b.output_storage,total*dim+2*guard,b.stream.value)),"all raw argument failures leave output untouched");
    std::vector<uint16_t> projection(dim*latent_dim,0),norm(dim,0);
    for (int i=0;i<dim;++i) {projection[size_t(i)*latent_dim+i]=0x3f80;if(i<64) norm[i]=0x3f80;}
    Source20IndexKeyWeights weights{projection,norm};
    for (int capacity : {0,-1,1048577,std::numeric_limits<int>::max()})
        rejects("invalid key capacity",[&]{CudaSource20IndexKeyCache bad(weights,capacity);});
    rejects("missing projection",[&]{CudaSource20IndexKeyCache bad({{},norm},total);});
    rejects("missing norm",[&]{CudaSource20IndexKeyCache bad({projection,{}},total);});
    rejects("wrong projection size",[&]{CudaSource20IndexKeyCache bad({std::span<const uint16_t>(projection).first(1),norm},total);});
    rejects("wrong norm size",[&]{CudaSource20IndexKeyCache bad({projection,std::span<const uint16_t>(norm).first(1)},total);});
    for (uint16_t nonfinite : {uint16_t(0x7f80),uint16_t(0x7fc0)}) {
        auto bad_projection=projection;bad_projection[0]=nonfinite;
        rejects("nonfinite projection",[&]{CudaSource20IndexKeyCache bad({bad_projection,norm},total);});
        auto bad_norm=norm;bad_norm[0]=nonfinite;
        rejects("nonfinite norm",[&]{CudaSource20IndexKeyCache bad({projection,bad_norm},total);});
    }
    CudaSource20IndexKeyCache cache(weights,total);
    const auto* allocation=cache.data();const size_t bytes=133120+total*256;
    check(cache.size()==0 && cache.capacity()==total && cache.device_bytes()==bytes && !cache.last_normalized(),"bounded allocation and empty diagnostic");
    check(download(cache.data(),total*dim,b.stream.value)==std::vector<uint16_t>(total*dim,0),"startup cache is zero");
    std::vector<float> latent(total*latent_dim),normalized(total*dim,0.f);
    for (int row=0;row<total;++row) {
        for (int i=0;i<latent_dim;++i) latent[size_t(row)*latent_dim+i]=(row+i)%3 ? -1.0001f:1.0001f;
        for (int i=0;i<64;++i) normalized[size_t(row)*dim+i]=bf(latent[size_t(row)*latent_dim+i]);
        // Zero gamma retains negative zero for negative projected values.
        for (int i=64;i<dim;++i) normalized[size_t(row)*dim+i]=std::copysign(0.f,latent[size_t(row)*latent_dim+i]);
    }
    b.clear();upload(b.latent,latent,b.stream.value);const auto expected=reference(normalized);
    rejects("oversized append cannot bind a stream",[&]{cache.append(b.latent,batch(0,18,18),other.value);});
    cache.append(b.latent,batch(0,7),b.stream.value);
    exact(download(cache.data(),7*dim,b.stream.value),{expected.begin(),expected.begin()+7*dim},"seven-token prefill index keys");
    const auto state=download(cache.data(),total*dim,b.stream.value);
    rejects("gap",[&]{cache.append(b.latent,batch(8,1),b.stream.value);});
    rejects("duplicate",[&]{cache.append(b.latent,batch(6,1),b.stream.value);});
    rejects("null latent",[&]{cache.append(nullptr,batch(7,1),b.stream.value);});
    rejects("unaligned latent",[&]{cache.append(reinterpret_cast<float*>(reinterpret_cast<char*>(b.latent)+1),batch(7,1),b.stream.value);});
    rejects("latent address overflow",[&]{cache.append(reinterpret_cast<float*>(UINTPTR_MAX-3),batch(7,1),b.stream.value);});
    rejects("cache storage cannot be latent",[&]{cache.append(reinterpret_cast<const float*>(cache.data()),batch(7,1),b.stream.value);});
    rejects("scratch storage cannot be latent",[&]{cache.append(cache.last_normalized(),batch(7,1),b.stream.value);});
    rejects("null append stream",[&]{cache.append(b.latent,batch(7,1),nullptr);});
    rejects("stream switch",[&]{cache.append(b.latent,batch(7,1),other.value);});
    rejects("reset stream switch",[&]{cache.reset(other.value);});
    check(cache.size()==7 && same(state,download(cache.data(),total*dim,b.stream.value)),"failed append leaves full storage unchanged");
    for(int row=7;row<total;++row) cache.append(b.latent+row*latent_dim,batch(row,1),b.stream.value);
    exact(download(cache.data(),total*dim,b.stream.value),expected,"single-token decode fills bounded cache");
    rejects("full cache cannot evict",[&]{cache.append(b.latent,batch(17,1,18),b.stream.value);});
    check(cache.size()==total && same(expected,download(cache.data(),total*dim,b.stream.value)),"full rejection preserves cache");
    cache.reset(b.stream.value);
    check(cache.size()==0 && !cache.last_normalized() && cache.data()==allocation && cache.device_bytes()==bytes &&
          download(cache.data(),total*dim,b.stream.value)==std::vector<uint16_t>(total*dim,0),"reset retains allocation and clears all slots");
    rejects("stale descriptor after reset",[&]{cache.append(b.latent,batch(16,1),b.stream.value);});
    for(int row=0;row<total;++row) cache.append(b.latent+row*latent_dim,batch(row,1),b.stream.value);
    exact(download(cache.data(),total*dim,b.stream.value),expected,"single-token replay parity");
    cache.reset(b.stream.value);cache.append(b.latent,batch(0,total),b.stream.value);
    exact(download(cache.data(),total*dim,b.stream.value),expected,"whole-prefill replay parity");
    check(same(latent,download(b.latent,latent.size(),b.stream.value)),"all key appends preserve original latent");
    b.guards(b.latent_storage,total,latent_dim);
    cache.reset(b.stream.value);upload(b.latent,std::vector<float>(latent_dim,0.f),b.stream.value);
    cache.append(b.latent,batch(0,1),b.stream.value);
    exact(download(cache.data(),dim,b.stream.value),reference(std::vector<float>(dim,0.f)),"zero input after reset");
    check(download(cache.data()+dim,(total-1)*dim,b.stream.value)==std::vector<uint16_t>((total-1)*dim,0),"unused slots remain zero");
}
int number(const GgufModel& model, const std::string& name, int maximum) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::U32 || value->u > uint64_t(maximum))
        throw std::runtime_error("Malformed fixture integer " + name);
    return int(value->u);
}
std::string string_value(const GgufModel& model, const std::string& name) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::STRING) throw std::runtime_error("Missing fixture string " + name);
    return value->s;
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, int rows, int width) {
    size_t shard = 0; const auto* value = model.find(name, &shard);
    if (!value || value->type != (sizeof(T) == 2 ? 30u : 0u) ||
        value->shape != std::vector<uint64_t>{uint64_t(width), uint64_t(rows), 1} || !model.in_bounds(*value, shard))
        throw std::runtime_error("Malformed reference tensor " + name);
    const auto* data = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*value));
    return {data, data + size_t(rows) * width};
}
size_t norm_values=0,norm_different=0;
double max_norm_rel=0,max_norm_abs=0;
void normalized_check(const std::vector<float>& actual,const std::vector<float>& expected,const std::string& name) {
    if(actual.size()!=expected.size()) throw std::runtime_error("Normalized reference shape mismatch");
    double error=0,energy=0,peak=0,maximum=0;bool finite=true;size_t different=0;
    for(size_t i=0;i<actual.size();++i) {
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        const double delta=double(actual[i])-expected[i];error+=delta*delta;energy+=double(expected[i])*expected[i];
        peak=std::max(peak,std::abs(double(expected[i])));maximum=std::max(maximum,std::abs(delta));
        different+=actual[i]!=expected[i];
    }
    const double relative=std::sqrt(error/std::max(energy,1e-30));
    norm_values+=actual.size();norm_different+=different;max_norm_rel=std::max(max_norm_rel,relative);max_norm_abs=std::max(max_norm_abs,maximum);
    // Same declared pre-quantization normalized tolerance as the compressor gate.
    // Final BF16 FP4 cache comparisons below still require exact bytes.
    check(finite && relative<=.003 && maximum<=1e-5+.02*peak,name+" normalized key tolerance");
}
void real(const char* fixture_path,const char* compressor_path,const char* kv_path) {
    const auto model=GgufModel::open(fixture_path),weights=GgufModel::open(compressor_path),kv_ref=GgufModel::open(kv_path);
    if(string_value(model,"general.architecture")!="deepseek41-source20-index-key-fixture" ||
       string_value(model,"strata.deepseek41.revision")!="1bae0e1088993b9b82de08f59de5be5a34e42170" ||
       string_value(model,"strata.deepseek41.official_source_sha256")!="e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d" ||
       string_value(model,"strata.deepseek41.official_kernel_sha256")!="1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455" ||
       string_value(weights,"strata.deepseek41.revision")!=string_value(model,"strata.deepseek41.revision") ||
       string_value(kv_ref,"strata.deepseek41.revision")!=string_value(model,"strata.deepseek41.revision") ||
       string_value(weights,"strata.deepseek41.source_hashes")!=string_value(model,"strata.deepseek41.compressor_source_hashes") ||
       string_value(kv_ref,"strata.deepseek41.source_hashes")!=string_value(model,"strata.deepseek41.compressor_source_hashes"))
        throw std::runtime_error("Wrong source-20 reference/checkpoint identity");
    Buffers b;const int cases=number(model,"fixture.cases",32);
    if(cases!=19) throw std::runtime_error("Expected 19 raw postprocess cases");
    for(int index=0;index<cases;++index) {
        const std::string prefix="case."+std::to_string(index)+".";
        const int position=number(model,prefix+"position",1048575),tokens=number(model,prefix+"tokens",4);
        if(tokens<1) throw std::runtime_error("Empty reference case");
        const auto input=tensor<float>(model,prefix+"input",tokens,dim);
        b.clear();upload(b.key,input,b.stream.value);
        postprocess_source20_index_keys(b.key,b.output,tokens,position,b.stream.value);
        exact(download(b.output,input.size(),b.stream.value),tensor<uint16_t>(model,prefix+"output",tokens,dim),prefix+"official RoPE / independent E8M0 FP4");
        check(same(input,download(b.key,input.size(),b.stream.value)),prefix+"normalized key preserved");
        b.guards(b.key_storage,tokens,dim);b.output_guards(tokens);
    }
    Geometry g;g.layers=40;g.hidden=hidden;g.compress_ratios.resize(40);g.compress_ratios[20]=1;
    g.sources.resize(40);g.sources[20]={20,20,-1};
    CudaSource20Compressor compressor(read_source20_compressor_weights(weights,g),total,total);
    const auto key_weights=read_source20_index_key_weights(model,g);
    CudaSource20IndexKeyCache keys(key_weights,total),oracle_keys(key_weights,total);
    CudaSource20KvCache kv(total);
    const int plans=number(model,"fixture.plans",16);
    if(plans!=9 || number(kv_ref,"fixture.plans",16)!=plans) throw std::runtime_error("Expected nine matching sequence plans");
    int calls=0;std::vector<uint16_t> baseline;
    for(int plan=0;plan<plans;++plan) {
        const std::string prefix="plan."+std::to_string(plan)+".";
        const int tokens=number(model,prefix+"tokens",total),first=number(model,prefix+"prefill",total);
        if(tokens<1 || first<1 || first>tokens || number(kv_ref,prefix+"tokens",total)!=tokens || number(kv_ref,prefix+"prefill",total)!=first)
            throw std::runtime_error("Invalid or mismatched sequence plan");
        const auto input=tensor<float>(model,prefix+"input",tokens,hidden);
        check(same(input,tensor<float>(kv_ref,prefix+"input",tokens,hidden)),prefix+"same original input as KV oracle");
        b.clear();upload(b.x,input,b.stream.value);
        compressor.reset(b.stream.value);keys.reset(b.stream.value);oracle_keys.reset(b.stream.value);kv.reset(b.stream.value);
        for(int start=0;start<tokens;) {
            const int count=start==0 ? first:1;
            const std::string call=prefix+"at."+std::to_string(start)+".";
            // Validate both consumers before advancing the producer; no joint transaction yet.
            if(count>keys.capacity()-keys.size() || count>kv.capacity()-kv.size()) throw std::runtime_error("Test cache capacity exhausted");
            b.clear_latent();
            const auto emitted=compressor.forward(b.x+size_t(start)*hidden,b.latent,count,start,b.stream.value);
            const auto before=download(b.latent,size_t(count)*latent_dim,b.stream.value);
            if(plan%2==0) kv.append(b.latent,emitted,b.stream.value);
            keys.append(b.latent,emitted,b.stream.value);
            if(plan%2!=0) kv.append(b.latent,emitted,b.stream.value);
            check(keys.size()==start+count && kv.size()==start+count && compressor.next_position()==start+count,call+"all producer/consumer positions agree");
            const auto expected=tensor<uint16_t>(model,call+"cache",start+count,dim);
            exact(download(keys.data(),expected.size(),b.stream.value),expected,call+"native compressor/key pipeline");
            const auto expected_kv=tensor<uint16_t>(kv_ref,call+"cache",start+count,latent_dim);
            exact(download(kv.data(),expected_kv.size(),b.stream.value),expected_kv,call+"paired original KV cache");
            const auto normalized=tensor<float>(model,call+"normalized",count,dim);
            const std::vector<float> last(normalized.end()-dim,normalized.end());
            normalized_check(download(keys.last_normalized(),dim,b.stream.value),last,call+"native latent");
            check(same(before,download(b.latent,before.size(),b.stream.value)),call+"both consumers preserve pre-RoPE latent");
            b.guards(b.latent_storage,count,latent_dim);
            const auto original_latent=tensor<float>(model,call+"latent",count,latent_dim);
            check(same(original_latent,tensor<float>(kv_ref,call+"latent",count,latent_dim)),call+"official latent matches KV fixture");
            upload(b.latent,original_latent,b.stream.value);oracle_keys.append(b.latent,emitted,b.stream.value);
            exact(download(oracle_keys.data(),expected.size(),b.stream.value),expected,call+"isolated index-key pipeline with official latent");
            normalized_check(download(oracle_keys.last_normalized(),dim,b.stream.value),last,call+"official latent");
            check(same(original_latent,download(b.latent,original_latent.size(),b.stream.value)),call+"official latent preserved");
            const size_t unused=size_t(total-start-count)*dim;
            check(download(keys.data()+expected.size(),unused,b.stream.value)==std::vector<uint16_t>(unused,0) &&
                  download(oracle_keys.data()+expected.size(),unused,b.stream.value)==std::vector<uint16_t>(unused,0),call+"unused key slots remain zero");
            ++calls;start+=count;
        }
        const auto actual=download(keys.data(),size_t(tokens)*dim,b.stream.value);
        if(plan==0) baseline=actual;else if(plan<6) check(same(actual,baseline),prefix+"reset/prefill/decode partition byte parity");
        check(same(input,download(b.x,input.size(),b.stream.value)),prefix+"decoder input preserved");
        check(same(download(b.x_storage,guard,b.stream.value),std::vector<float>(guard,-333.f)) &&
              same(download(b.x+total*hidden,guard,b.stream.value),std::vector<float>(guard,-333.f)),prefix+"decoder input guards preserved");
        check(keys.device_bytes()==137472 && oracle_keys.device_bytes()==137472 && kv.device_bytes()==17408,prefix+"fixed allocations");
    }
    check(calls==58,"all 58 source-20 sequence calls consumed");
    std::printf("source20_key_capacity=%d key_bytes=%zu compressor_plus_both_caches_bytes=%zu\n",keys.capacity(),keys.device_bytes(),compressor.device_bytes()+keys.device_bytes()+kv.device_bytes());
}
} // namespace
int main(int argc,char** argv) {
    if(!(argc==1 || (argc==5 && std::string(argv[1])=="--real"))) return 2;
    int count=0;const auto status=cudaGetDeviceCount(&count);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && count==0)) return 77;
    try {ck(status);ck(cudaSetDevice(0));if(argc==5) real(argv[2],argv[3],argv[4]);else synthetic();}
    catch(const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());++failures;}
    std::printf("normalized_values=%zu different=%zu max_relative_l2=%.12g max_abs=%.12g\n",norm_values,norm_different,max_norm_rel,max_norm_abs);
    std::printf("compared_bf16_values=%zu different_bf16_values=%zu\n",compared,mismatched);
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
