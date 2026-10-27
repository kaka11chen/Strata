#include "strata/deepseek41/memory_plan.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/deepseek41/host_expert_cache.hpp"
#include "strata/core/conversation_memory.hpp"
#include <stdexcept>

namespace strata::deepseek41 {
GenerationMemoryPlan short_generation_memory(int slots, int prefill, uint64_t cpu_cache_budget) {
    if (slots < 1 || slots > 384 || prefill < 1 || prefill > 5)
        throw std::invalid_argument("1017 short-generation memory profile needs 1..384 expert slots and 1..5 prompt tokens");
    constexpr uint64_t h = 5120, width = 2304, vocab = 129280, layers = 40, mib = 1ull << 20;
    Geometry g; g.layers = 40; g.hidden = int(h); g.expert_width = int(width); g.experts = 384; g.top_k = 6;
    const uint64_t expert = nvfp4_expert_bytes(g), raw_expert = 3 * h * width / 32 * 17;
    // attention_cuda.cu: five BF16 matrices per original layer.
    const uint64_t attention = 2 * (1280 * h + 32768ull * 1280 + 512 * h + 8192ull * 4096 + h * 8192);
    const uint64_t shared = 3 * h * width * 2;
    const uint64_t mhc = 2 * (96 * h * 4 + 12 + 96 + h * 4);
    const uint64_t router = 384 * h * 2;
    const uint64_t engram = 2 * (25600ull * 6144 * 2 + 4 * h * 4);
    const uint64_t compressor = 3 * (2 * h * 512 * 4 + 512 * 4) + h * 512 * 2 + 512 * 4;
    const uint64_t keys = 4 * (512 * 128 * 2 + 128 * 4);
    const uint64_t queries = 8 * (32 * 128 * 1280 * 2 + 32 * h * 2);
    GenerationMemoryPlan p;
    p.dense_device = layers * (attention + shared + mhc + router + (1280 + 512 + 64) * 4) +
                     engram + compressor + keys + queries + vocab * h * 2 + h * 4;
    p.expert_device = layers * uint64_t(slots) * expert;
    // <=5 positions: 8 MiB/layer covers attention/compressor/index/KV, mHC,
    // routed/shared scratch, cache maps, alignment and CUB selection storage.
    // 2 MiB additionally covers prefix double buffers and generation/text IO.
    // cuBLAS/runtime allocations are separate configurable admission headroom.
    p.workspace_device = layers * 8 * mib + 2 * mib;
    // Charge all dense host inputs at the larger device representation, plus
    // the full CPU embedding and copied routers/maps. This is conservative:
    // original FP8 inputs take less space and many views share mappings.
    p.weights_host = p.dense_device + vocab * h * 2 + 256 * mib;
    // Guarded fixture Source: seven mapped ExpertIndices, each accepting <=64
    // populated experts, seven adapter staging buffers and one raw/native pair.
    // This bound is specific to that loader; a future source must supply its own
    // accounting before this profile can be used for arbitrary model packages.
    p.source_host = (7 * 64 + 1) * raw_expert + 8 * expert;
    // Serial FP8 expansion (largest matrix 80 MiB), row caches/read slabs,
    // token maps, residency tables and bounded request staging. No full Engram.
    p.workspace_host = 512 * mib;
    p.expert_cache_host = host_expert_cache_layout(g, cpu_cache_budget).allocated_bytes();
    return p;
}
std::string memory_rejection(const GenerationMemoryPlan& p, const GenerationMemoryLimits& l, const MemoryAvailable& a) {
    using core::conversation_memory_admit;
    if (!conversation_memory_admit(l.device_bytes, p.device_bytes(), l.device_headroom))
        return "GPU budget cannot hold weights, expert slots, workspace and CUDA headroom";
    if (!conversation_memory_admit(l.host_bytes, p.host_bytes(), l.host_headroom))
        return "Host budget cannot hold dense inputs, bounded weight staging/cache and headroom";
    if (!conversation_memory_admit(a.device, p.device_bytes(), l.device_headroom))
        return a.device ? "Insufficient live free GPU memory" : "GPU memory availability is unknown";
    if (!conversation_memory_admit(a.host, p.host_bytes(), l.host_headroom))
        return a.host ? "Insufficient live host/cgroup memory" : "Host/cgroup memory availability is unknown";
    return {};
}
void require_memory(const GenerationMemoryPlan& p, const GenerationMemoryLimits& l, const MemoryAvailable& a) {
    const auto reason = memory_rejection(p, l, a);
    if (!reason.empty()) throw std::runtime_error("DeepSeek memory admission rejected before weight loading: " + reason);
}
} // namespace strata::deepseek41
