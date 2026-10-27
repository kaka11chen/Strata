#include "strata/deepseek41/memory_plan.hpp"
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
using namespace strata::deepseek41;
int main() {
    int checks=0;
    auto check=[&](bool ok,const char* label){++checks;if(!ok)throw std::runtime_error(label);};
    const auto p=short_generation_memory(2,2);GenerationMemoryLimits limits;
    const MemoryAvailable enough{limits.device_bytes,limits.host_bytes};
    check(memory_rejection(p,limits,enough).empty(),"20/64 GiB short profile admits two slots");
    check(!memory_rejection(short_generation_memory(6,2),limits,enough).empty(),"six slots cannot bypass the common GPU budget");
    const auto one=short_generation_memory(1,1),three=short_generation_memory(3,5);
    check(three.device_bytes()-p.device_bytes()==p.device_bytes()-one.device_bytes(),"one slot per layer consumes the same global cache increment");
    check(three.host_bytes()==one.host_bytes(),"GPU cache choice does not multiply weight staging");
    auto exact=limits;exact.device_bytes=p.device_bytes()+limits.device_headroom;exact.host_bytes=p.host_bytes()+limits.host_headroom;
    check(memory_rejection(p,exact,{exact.device_bytes,exact.host_bytes}).empty(),"exact boundary admitted");
    --exact.device_bytes;check(!memory_rejection(p,exact,enough).empty(),"one-byte GPU budget deficit rejected");
    ++exact.device_bytes;--exact.host_bytes;check(!memory_rejection(p,exact,enough).empty(),"one-byte host budget deficit rejected");
    check(!memory_rejection(p,limits,{p.device_bytes()+limits.device_headroom-1,limits.host_bytes}).empty(),"live VRAM can be tighter than the operator budget");
    check(!memory_rejection(p,limits,{limits.device_bytes,p.host_bytes()+limits.host_headroom-1}).empty(),"live cgroup room can be tighter than host budget");
    check(!memory_rejection(p,limits,{std::nullopt,limits.host_bytes}).empty(),"unknown GPU probe fails closed");
    check(!memory_rejection(p,limits,{limits.device_bytes,std::nullopt}).empty(),"unknown host probe fails closed");
    check(!memory_rejection(p,limits,{0,limits.host_bytes}).empty(),"known zero GPU availability fails");
    check(!memory_rejection(p,limits,{limits.device_bytes,0}).empty(),"known zero host availability fails");
    auto overflow=limits;overflow.device_headroom=std::numeric_limits<uint64_t>::max();
    check(!memory_rejection(p,overflow,enough).empty(),"headroom comparison does not wrap");
    for(auto pair:{std::pair{0,2},std::pair{385,2},std::pair{2,0},std::pair{2,6}}){
        try{(void)short_generation_memory(pair.first,pair.second);check(false,"unsupported profile accepted");}
        catch(const std::invalid_argument&){check(true,"unsupported profile rejected");}
    }
    const auto cached=short_generation_memory(2,2,512ull<<20);
    check(cached.device_bytes()==p.device_bytes() && cached.host_bytes()>p.host_bytes(),"CPU cache is charged only to host allowance");
    check(cached.expert_cache_host<=(512ull<<20) && cached.expert_cache_host>(480ull<<20),"host cache bound includes complete slots and metadata");
    auto host_limit=limits;host_limit.host_bytes=p.host_bytes()+limits.host_headroom;
    check(!memory_rejection(cached,host_limit,enough).empty(),"cache cannot bypass the common host limit");
    check(short_generation_memory(2,2,0).host_bytes()==p.host_bytes(),"zero CPU cache preserves the prior memory plan");
    bool rejected=false;try{require_memory(p,limits,{0,0});}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"require rejects before allocation");
    std::printf("device_upper=%llu host_upper=%llu\n%d checks, 0 failures\n",(unsigned long long)p.device_bytes(),(unsigned long long)p.host_bytes(),checks);
}
