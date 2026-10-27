#include "prefix_fixture.hpp"
#include <bit>

namespace fs = std::filesystem;
namespace {
std::vector<prefix_fixture::Case> inputs(const prefix_fixture::RealWeights& f) {
    std::vector<prefix_fixture::Case> cases;
    for (int c=0;c<3;++c) {
        const size_t n=c?1:3; const auto name="case."+std::to_string(c)+".";
        cases.push_back({tensor<int32_t>(f.decoder[0],name+"tokens",24,n),
            tensor<float>(f.input,name+"input",0,n*residual_size),tensor<float>(f.input,name+"incoming",0,n*4)});
    }
    return cases;
}
std::vector<std::vector<float>> source2_state(const CudaTextPrefix& p,cudaStream_t s) {
    return {cache(p.kv_cache(2),s),cache(p.compressed_kv_cache(),s,size_t(p.capacity())*512),
            cache(p.index_key_cache(),s,size_t(p.capacity())*128)};
}
void capture(const fs::path& dir,const fs::path& extras,const fs::path& output) {
    if (std::endian::native!=std::endian::little || fs::exists(output)) throw std::invalid_argument("Capture needs little endian and a fresh output path");
    prefix_fixture::RealWeights f(dir,8);prefix_fixture::Source source(f,extras);Stream stream;
    EngramIoOptions io;io.allow_partial=true;
    CudaTextPrefix prefix(f.g,f.weights,f.map,f.table,source,6,3,3,io);
    core::DeviceArena buffers(1024*1024);auto alloc=[&](size_t n){return static_cast<float*>(buffers.alloc(n*4));};
    auto *x=alloc(3*residual_size),*pre=alloc(12),*y=alloc(3*residual_size),*next=alloc(12);
    std::vector<float> residual,coefficients;std::vector<int32_t> tokens;int pos=0;
    for (const auto& c:inputs(f)) {
        const size_t n=c.tokens.size();upload(x,c.residual,stream.value);upload(pre,c.incoming,stream.value);
        prefix.forward(c.tokens,x,pre,y,next,pos,stream.value);
        const auto out=download(y,n*residual_size,stream.value),co=download(next,n*4,stream.value);
        check(std::all_of(out.begin(),out.end(),[](float v){return std::isfinite(v)&&v==bf(v);}),"actual layer-7 residual is finite BF16-valued");
        check(std::all_of(co.begin(),co.end(),[](float v){return std::isfinite(v)&&v>=0&&v<=1.000002f;}),"actual same-token layer-7 coefficients are finite and bounded");
        residual.insert(residual.end(),out.begin(),out.end());coefficients.insert(coefficients.end(),co.begin(),co.end());
        tokens.insert(tokens.end(),c.tokens.begin(),c.tokens.end());pos+=int(n);
    }
    if (failures || pos!=5) throw std::runtime_error("Invalid native handoff");
    std::ofstream out(output,std::ios::binary);out.exceptions(std::ios::badbit|std::ios::failbit);
    const uint32_t header[]{1,5};out.write("DS41PFX7",8);out.write(reinterpret_cast<const char*>(header),8);
    out.write(reinterpret_cast<const char*>(tokens.data()),std::streamsize(tokens.size()*4));
    out.write(reinterpret_cast<const char*>(residual.data()),std::streamsize(residual.size()*4));
    out.write(reinterpret_cast<const char*>(coefficients.data()),std::streamsize(coefficients.size()*4));out.close();
    std::printf("captured_prefix_layers=8 tokens=%d prefix_device_bytes=%zu handoff_bytes=%ju\n",pos,prefix.device_bytes(),uintmax_t(fs::file_size(output)));
}
struct Weights8 {
    Geometry g=geometry();GgufModel model,routed;
    std::vector<uint16_t> router;std::vector<float> bias;
    ExpertIndex index;std::array<bool,384> populated{};
    DecoderWeights weights;CompressorWeights comp;IndexKeyWeights key;IndexQueryWeights query;
    Weights8(const char* decoder,const char* experts)
        :model(GgufModel::open(decoder)),routed(GgufModel::open(experts)),
         router(tensor<uint16_t>(routed,"router.weight",30,hidden*384)),bias(tensor<float>(routed,"router.bias",0,384)),index(routed,g,{8}),
         weights{read_mhc_weights(model,g,8,MhcBranch::attention),read_mhc_weights(model,g,8,MhcBranch::ffn),read_attention_weights(model,g,8),read_shared_expert(model,g,8),router,bias},
         comp(read_compressor_weights(model,g,8)),key(read_index_key_weights(model,g,8)),query(read_index_query_weights(model,g,8)) {
        const auto* ids=routed.meta().get("strata.deepseek41.populated_ids");
        if (!ids || ids->type!=MetaType::ARRAY || ids->items.size()!=ids->count || ids->count<6 || ids->count>32)
            throw std::runtime_error("Missing bounded original source-8 expert coverage");
        for (const auto& id:ids->items) {if(id.u>=384)throw std::runtime_error("Invalid expert ID");populated[size_t(id.u)]=true;}
    }
};
struct Result {std::vector<float> output,coefficients,window,compressed,keys;std::vector<Route> routes;};
bool same_routes(const std::vector<Route>& a,const std::vector<Route>& b) {
    if(a.size()!=b.size())return false;
    for(size_t i=0;i<a.size();++i)if(a[i].ids!=b[i].ids || a[i].weights!=b[i].weights)return false;
    return true;
}
void chain(const fs::path& dir,const fs::path& extras,const char* decoder,const char* experts) {
    prefix_fixture::RealWeights f(dir,8);Weights8 w(decoder,experts);Stream stream;
    core::DeviceArena buffers(1024*1024);auto alloc=[&](size_t n){return static_cast<float*>(buffers.alloc(n*4));};
    auto *x=alloc(3*residual_size),*p=alloc(12),*middle=alloc(3*residual_size),*mp=alloc(12),*y=alloc(3*residual_size),*next=alloc(12);
    std::vector<Result> expected;const auto cases=inputs(f);
    // Separate source-8 calls on the captured fixture. The decoder's pinned CPU
    // oracle is checked by the source decoder test; this is the live handoff gate.
    {
        CoveredSource source(w.g,w.index,w.populated,8);
        CudaSourceDecoder owner(w.g,w.weights,w.comp,w.key,w.query,source,6,3,0,3,8);
        for(int c=0;c<3;++c) {
            const int n=c?1:3,pos=c?c+2:0;const auto name="case."+std::to_string(c)+".";
            upload(x,tensor<float>(w.model,name+"input",0,size_t(n)*residual_size),stream.value);
            upload(p,tensor<float>(w.model,name+"incoming",0,size_t(n)*4),stream.value);
            std::vector<Route> routes(static_cast<size_t>(n));owner.forward(x,p,y,next,n,pos,stream.value,routes);
            expected.push_back({download(y,size_t(n)*residual_size,stream.value),download(next,size_t(n)*4,stream.value),cache(owner.kv_cache(),stream.value),
                cache(owner.compressed_kv_cache(),stream.value,3*512),cache(owner.index_key_cache(),stream.value,3*128),std::move(routes)});
        }
    }
    for(int slots:{6,2}) {
        prefix_fixture::Source upstream(f,extras);CoveredSource source(w.g,w.index,w.populated,8);
        EngramIoOptions options;options.allow_partial=true;
        CudaTextPrefix prefix(f.g,f.weights,f.map,f.table,upstream,slots,3,3,options);
        CudaSourceDecoder owner(w.g,w.weights,w.comp,w.key,w.query,source,slots,3,0,3,8);
        const size_t bytes=prefix.device_bytes()+owner.device_bytes();
        for(int pass=0;pass<2;++pass) {
            if(pass) {
                const auto publication=owner.shared_batch();const auto old=cache(owner.compressed_kv_cache(),stream.value,3*512);
                prefix.reset(stream.value);
                check(publication.source_layer()==8 && same(old,cache(owner.compressed_kv_cache(),stream.value,3*512)),"resetting source-2 prefix preserves independent source-8 state");
                owner.reset(stream.value);rejects("source-8 reset expires its publication",[&]{(void)publication.source_layer();});
            }
            int pos=0;
            for(size_t c=0;c<cases.size();++c) {
                const auto& input=cases[c];const int n=int(input.tokens.size());const auto name="case."+std::to_string(c)+".";
                upload(x,input.residual,stream.value);upload(p,input.incoming,stream.value);
                prefix.forward(input.tokens,x,p,middle,mp,pos,stream.value);
                check(same(download(middle,size_t(n)*residual_size,stream.value),tensor<float>(w.model,name+"input",0,size_t(n)*residual_size)) &&
                      same(download(mp,size_t(n)*4,stream.value),tensor<float>(w.model,name+"incoming",0,size_t(n)*4)),"live layer-7 handoff matches the captured oracle input bit-exactly");
                const auto before=source2_state(prefix,stream.value);std::vector<Route> routes(static_cast<size_t>(n));
                owner.forward(middle,mp,y,next,n,pos,stream.value,routes);
                check(same(download(y,size_t(n)*residual_size,stream.value),expected[c].output) && same(download(next,size_t(n)*4,stream.value),expected[c].coefficients),"live source-8 residual and coefficients match separate calls bit-exactly");
                check(same_routes(routes,expected[c].routes),"live source-8 retains original routes and weights");
                check(same(cache(owner.kv_cache(),stream.value),expected[c].window) && same(cache(owner.compressed_kv_cache(),stream.value,3*512),expected[c].compressed) &&
                      same(cache(owner.index_key_cache(),stream.value,3*128),expected[c].keys),"source-8 owned histories match separate calls bit-exactly");
                check(before==source2_state(prefix,stream.value) && prefix.next_position()==pos+n,"source-8 execution preserves source-2 prefix state");
                check(owner.shared_batch().source_layer()==8 && owner.next_position()==pos+n && owner.compressed_groups()==(pos+n)/2 &&
                      prefix.device_bytes()+owner.device_bytes()==bytes,"separate source-8 identity, position and fixed allocation");
                pos+=n;
            }
        }
        const auto stats=owner.stats();check(stats.hits+stats.misses==60 && stats.misses==uint64_t(source.reads()) && stats.evictions>0,"live source-8 uses original-ID cache accounting");
        std::printf("live_chain_slots=%d prefix_layers=8 prefix_device_bytes=%zu source8_device_bytes=%zu combined_device_bytes=%zu\n",slots,prefix.device_bytes(),owner.device_bytes(),bytes);
    }
}
} // namespace
int main(int argc,char** argv) {
    if(!((argc==5 && std::string(argv[1])=="--capture") || (argc==6 && std::string(argv[1])=="--chain"))) {
        std::fprintf(stderr,"usage: source8_handoff --capture FIXTURES EXTRAS OUTPUT | --chain FIXTURES EXTRAS DECODER8 ROUTED8\n");return 2;
    }
    int n=0;const auto status=cudaGetDeviceCount(&n);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !n))return 77;
    try {ck(status);ck(cudaSetDevice(0));if(argc==5)capture(argv[2],argv[3],argv[4]);else chain(argv[2],argv[3],argv[4],argv[5]);}
    catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
