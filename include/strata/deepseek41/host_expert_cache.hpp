#pragma once
#include "strata/deepseek41/artifact.hpp"
#include <cstdint>
#include <memory>

namespace strata::deepseek41 {
struct HostExpertCacheLayout {
    size_t slots = 0, blob_bytes = 0, payload_bytes = 0, metadata_bytes = 0;
    size_t allocated_bytes() const { return payload_bytes + metadata_bytes; }
};
// Budget includes payload and all heap-allocated index/LRU metadata. Zero disables
// the cache; a positive budget must fit one expert. No allocation in this planner.
HostExpertCacheLayout host_expert_cache_layout(const Geometry&, uint64_t budget_bytes);
struct HostExpertCacheStats { uint64_t hits = 0, misses = 0, evictions = 0; };
// One global cache, keyed by original (layer, expert), not a cache per layer.
// Single caller; pointers can be recycled by the next insert. No I/O or CUDA.
class HostExpertCache {
  public:
    HostExpertCache(const Geometry&, uint64_t budget_bytes);
    const uint8_t* find(int64_t layer, int64_t expert);
    const uint8_t* insert(int64_t layer, int64_t expert, std::span<const uint8_t> blob);
    const HostExpertCacheLayout& layout() const { return layout_; }
    HostExpertCacheStats stats() const { return stats_; }
    size_t resident_bytes() const { return filled_ * layout_.blob_bytes; }
  private:
    struct Slot { int32_t key = -1, older = -1, newer = -1; };
    static_assert(sizeof(Slot) == 3 * sizeof(int32_t));
    int32_t key(int64_t layer, int64_t expert) const;
    void unlink(int32_t slot);
    void newest(int32_t slot);
    int layers_, experts_;
    HostExpertCacheLayout layout_;
    std::unique_ptr<uint8_t[]> payload_;
    std::unique_ptr<int32_t[]> index_;
    std::unique_ptr<Slot[]> slots_;
    size_t filled_ = 0;
    int32_t oldest_ = -1, newest_ = -1;
    HostExpertCacheStats stats_;
};
} // namespace strata::deepseek41
