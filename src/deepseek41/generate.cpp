#include "strata/deepseek41/model_loader.hpp"
#include "strata/deepseek41/generation_cuda.hpp"
#include "strata/deepseek41/expert_source.hpp"
#include "strata/deepseek41/memory_plan.hpp"
#include "strata/core/conversation_memory.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

using namespace strata;
using namespace strata::deepseek41;
namespace {
void ck(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
struct Stream {cudaStream_t value{};Stream(){ck(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~Stream(){cudaStreamDestroy(value);}};
int integer(const char* raw){size_t end=0;const int n=std::stoi(raw,&end);if(end!=std::strlen(raw))throw std::invalid_argument("Invalid integer argument");return n;}
struct Args {
    std::string manifest;
    GenerationMemoryLimits limits;
    int slots=2,budget=2;uint64_t cache=0;bool plan=false;
    std::vector<int32_t> tokens;
};
Args parse(int argc,char** argv){
    Args a;std::set<std::string> seen;
    for(int i=1;i<argc;++i){const std::string key=argv[i];
        if(!seen.insert(key).second)throw std::invalid_argument("Repeated option: "+key);
        if(key=="--plan"){a.plan=true;continue;}
        if(key=="--tokens"){while(++i<argc)a.tokens.push_back(integer(argv[i]));break;}
        if(++i>=argc)throw std::invalid_argument("Missing value for "+key);
        if(key=="--manifest"){a.manifest=argv[i];continue;}
        const int n=integer(argv[i]);
        if(key=="--gpu-budget-mib" || key=="--host-budget-mib"){
            if(n<=0)throw std::invalid_argument("Memory limits must be positive MiB");
            (key=="--gpu-budget-mib"?a.limits.device_bytes:a.limits.host_bytes)=uint64_t(n)<<20;
        }else if(key=="--cpu-expert-cache-mib"){
            if(n<0)throw std::invalid_argument("CPU cache allowance must be nonnegative");
            a.cache=uint64_t(n)<<20;
        }else if(key=="--expert-slots")a.slots=n;
        else if(key=="--max-new-tokens")a.budget=n;
        else throw std::invalid_argument("Unknown option: "+key);
    }
    if(a.manifest.empty() || a.tokens.empty() || a.tokens.size()>5 || a.budget<0 || a.budget>5 ||
       (a.budget && a.tokens.size()+size_t(a.budget-1)>5))throw std::invalid_argument("Manifest generation requires 1..5 input IDs within context5");
    for(int32_t id:a.tokens)if(id<0 || id>=129280)throw std::invalid_argument("Token outside original vocabulary");
    return a;
}
void run(const Args& args){
    const auto manifest=ModelManifest::read(args.manifest);
    ck(cudaSetDevice(0));
    const auto* limits=&args.limits;const int slots=args.slots;const auto& prompt=args.tokens;
    const uint64_t cpu_cache_budget=args.cache;const bool plan_only=args.plan;
    GenerationMemoryPlan plan;core::DeviceInfo before;uint64_t observed_device=0,cap_bytes=0,held_bytes=0;
    std::unique_ptr<core::DeviceArena> test_reservation;
    if(limits){
        plan=short_generation_memory(slots,int(prompt.size()),cpu_cache_budget);
        // Test runner only: hold the difference from this larger physical card.
        // The native allocator then has at most the declared card size left.
        if(const char* raw=std::getenv("STRATA_DEEPSEEK41_TEST_GPU_CAP_MIB")){
            size_t end=0;const int mib=std::stoi(raw,&end);if(mib<=0 || end!=std::strlen(raw))throw std::invalid_argument("Bad test GPU cap");
            cap_bytes=uint64_t(mib)<<20;const auto initial=core::device_info();
            if(cap_bytes>initial.total_bytes)throw std::invalid_argument("Test cap exceeds physical GPU size");
            held_bytes=initial.total_bytes-cap_bytes;
            if(held_bytes)test_reservation=std::make_unique<core::DeviceArena>(held_bytes);
            std::printf("TEST_DEVICE_CAP {\"cap_bytes\":%llu,\"reserved_bytes\":%llu}\n",(unsigned long long)cap_bytes,(unsigned long long)held_bytes);
        }
        before=core::device_info();const auto host=core::sample_host_memory();
        const MemoryAvailable available{before.free_bytes,host?std::optional<uint64_t>(host->available):std::nullopt};
        const auto reason=memory_rejection(plan,*limits,available);
        std::printf("MEMORY_PLAN {\"profile\":\"1017-full40-context5\",\"expert_slots_per_layer\":%d,\"gpu_budget_bytes\":%llu,\"host_budget_bytes\":%llu,\"gpu_headroom_bytes\":%llu,\"host_headroom_bytes\":%llu,\"dense_device_bytes\":%llu,\"expert_device_bytes\":%llu,\"workspace_device_upper_bytes\":%llu,\"device_upper_bytes\":%llu,\"host_weights_upper_bytes\":%llu,\"host_source_upper_bytes\":%llu,\"host_workspace_upper_bytes\":%llu,\"host_upper_bytes\":%llu,\"cpu_expert_cache_budget_bytes\":%llu,\"host_expert_cache_bytes\":%llu,\"gpu_free_bytes\":%llu,\"host_available_bytes\":%llu,\"host_probe_known\":%s,\"admitted\":%s,\"reason\":\"%s\"}\n",slots,
            (unsigned long long)limits->device_bytes,(unsigned long long)limits->host_bytes,(unsigned long long)limits->device_headroom,(unsigned long long)limits->host_headroom,
            (unsigned long long)plan.dense_device,(unsigned long long)plan.expert_device,(unsigned long long)plan.workspace_device,(unsigned long long)plan.device_bytes(),
            (unsigned long long)plan.weights_host,(unsigned long long)plan.source_host,(unsigned long long)plan.workspace_host,(unsigned long long)plan.host_bytes(),(unsigned long long)cpu_cache_budget,(unsigned long long)plan.expert_cache_host,
            (unsigned long long)before.free_bytes,(unsigned long long)available.host.value_or(0),host?"true":"false",reason.empty()?"true":"false",reason.c_str());
        std::fflush(stdout);require_memory(plan,*limits,available);if(plan_only)return;
    }
    LoadedModel model(manifest);
    IndexedExpertSource source(std::move(model.experts));
    const uint64_t source_host=source.host_upper_bytes()+model.retained_expert_mapping_upper_bytes+manifest.metadata_upper_bytes();
    if(source_host>plan.source_host)throw std::runtime_error("Model expert source exceeds admitted staging/mapping bound");
    CachedExpertSource host_cache(model.geometry,source,cpu_cache_budget);
    Stream stream;EngramIoOptions options;options.allow_partial=!model.engram_complete[0] || !model.engram_complete[1];
    CudaTextPrefix prefix(model.geometry,model.weights,model.token_maps[0],model.tables[0],host_cache,slots,int(prompt.size()),2,options,0,model.token_maps[1],model.tables[1]);
    CudaTextIo io(model.text_io,int(prompt.size()));CudaGreedyGeneration generation(prefix,io);
    const size_t bytes=prefix.device_bytes()+io.device_bytes()+generation.device_bytes();
    auto verify=[&]{
        if(host_cache.allocated_bytes()!=plan.expert_cache_host || bytes>plan.device_bytes())throw std::runtime_error("Actual model allocations exceed admitted plan");
        const auto now=core::device_info();observed_device=std::max(observed_device,before.free_bytes>now.free_bytes?before.free_bytes-now.free_bytes:0);
        if(observed_device>limits->device_bytes || observed_device>plan.device_bytes()+limits->device_headroom ||
           (cap_bytes && now.total_bytes-now.free_bytes-held_bytes>cap_bytes))throw std::runtime_error("Observed GPU use exceeds admitted capacity");
        const auto host=core::sample_host_memory();
        if(!core::conversation_memory_admit(host?std::optional<uint64_t>(host->available):std::nullopt,0,limits->host_headroom))throw std::runtime_error("Host headroom exhausted");
    };
    verify();const auto result=generation.generate(prompt,args.budget,stream.value);verify();
    std::printf("GENERATION_RESULT {\"prompt_ids\":[");for(size_t i=0;i<prompt.size();++i)std::printf("%s%d",i?",":"",prompt[i]);
    std::printf("],\"generated_ids\":[");for(size_t i=0;i<result.tokens.size();++i)std::printf("%s%d",i?",":"",result.tokens[i]);
    std::printf("],\"stopped_on_eos\":%s,\"evaluated_tokens\":%d,\"context_capacity\":%d,\"prefix_device_bytes\":%zu,\"io_device_bytes\":%zu,\"generation_device_bytes\":%zu",result.stopped_on_eos?"true":"false",prefix.next_position(),generation.context_capacity(),prefix.device_bytes(),io.device_bytes(),generation.device_bytes());
    std::printf(",\"memory\":{\"gpu_budget_bytes\":%llu,\"host_budget_bytes\":%llu,\"device_upper_bytes\":%llu,\"host_upper_bytes\":%llu,\"observed_device_delta_bytes\":%llu,\"expert_slots_per_layer\":%d,\"simulated_card_bytes\":%llu}",(unsigned long long)limits->device_bytes,(unsigned long long)limits->host_bytes,(unsigned long long)plan.device_bytes(),(unsigned long long)plan.host_bytes(),(unsigned long long)observed_device,slots,(unsigned long long)cap_bytes);
    const auto cs=host_cache.stats();const auto fs=source.io_stats();
    std::printf(",\"host_expert_cache\":{\"allocated_bytes\":%zu,\"resident_payload_bytes\":%zu,\"hits\":%llu,\"misses\":%llu,\"evictions\":%llu,\"upstream_reads\":%lld}",host_cache.allocated_bytes(),host_cache.resident_bytes(),(unsigned long long)cs.hits,(unsigned long long)cs.misses,(unsigned long long)cs.evictions,(long long)source.reads());
    std::printf(",\"expert_file_io\":{\"staging_bytes\":%zu,\"reads\":%llu,\"requested_bytes\":%llu,\"completed_bytes\":%llu,\"converted\":%llu,\"failures\":%llu}",source.reader_staging_bytes(),(unsigned long long)fs.reads,(unsigned long long)fs.requested_bytes,(unsigned long long)fs.completed_bytes,(unsigned long long)fs.converted,(unsigned long long)fs.failures);
    std::printf(",\"model_manifest\":{\"version\":1,\"mapped_component_files\":%zu,\"experts_complete\":%s,\"engram1_complete\":%s,\"engram14_complete\":%s},\"indexed_expert_source\":{\"entries\":%zu,\"files\":%zu,\"metadata_upper_bytes\":%zu,\"staging_bytes\":%zu,\"retained_mapped_expert_upper_bytes\":%llu,\"host_upper_bytes\":%llu,\"logical_payload_bytes\":%llu,\"complete_expert_coverage\":%s}}\n",model.mapped_files(),model.experts_complete?"true":"false",model.engram_complete[0]?"true":"false",model.engram_complete[1]?"true":"false",source.index().entries(),source.index().files(),source.index().metadata_upper_bytes(),source.staging_bytes(),(unsigned long long)model.retained_expert_mapping_upper_bytes,(unsigned long long)source_host,(unsigned long long)fs.payload_bytes,model.experts_complete?"true":"false");
}
}
int main(int argc,char** argv){
    if(argc==2 && std::string(argv[1])=="--help"){
        std::puts("strata-deepseek41-generate --manifest MODEL.gguf [--plan] [--gpu-budget-mib N] [--host-budget-mib N] [--expert-slots N] [--cpu-expert-cache-mib N] [--max-new-tokens N] --tokens ID...");return 0;
    }
    try{run(parse(argc,argv));return 0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
