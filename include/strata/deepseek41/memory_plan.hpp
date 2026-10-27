#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace strata::deepseek41 {
// Admission bounds for the confirmed original 40-layer, five-input-token
// fixture runner. Not a general context planner or an OS/allocator hard cap.
// Dense sizes follow the pinned 1017 shapes; workspaces deliberately use an
// upper bound, tested against actual allocated arenas. No model/GPU allocation.
struct GenerationMemoryPlan {
    uint64_t dense_device = 0, expert_device = 0, workspace_device = 0;
    uint64_t weights_host = 0, source_host = 0, workspace_host = 0, expert_cache_host = 0;
    uint64_t device_bytes() const { return dense_device + expert_device + workspace_device; }
    uint64_t host_bytes() const { return weights_host + source_host + workspace_host + expert_cache_host; }
};
struct GenerationMemoryLimits {
    uint64_t device_bytes = 20ull << 30, host_bytes = 64ull << 30;
    uint64_t device_headroom = 2ull << 30, host_headroom = 4ull << 30;
};
struct MemoryAvailable {
    std::optional<uint64_t> device, host; // unknown differs from a known zero
};
// Original hidden/vocabulary/layers, ratio-2 capacity=2, ratio-1 capacity=5,
// Engram cache=4096 rows/reader and max_inflight=24. Other profiles fail closed.
GenerationMemoryPlan short_generation_memory(int slots_per_layer, int prefill_tokens, uint64_t cpu_cache_budget = 0);
// Tests both operator limits and live free memory using Strata's existing
// admission predicate. This is not a reservation against concurrent processes.
std::string memory_rejection(const GenerationMemoryPlan&, const GenerationMemoryLimits&, const MemoryAvailable&);
void require_memory(const GenerationMemoryPlan&, const GenerationMemoryLimits&, const MemoryAvailable&);
} // namespace strata::deepseek41
