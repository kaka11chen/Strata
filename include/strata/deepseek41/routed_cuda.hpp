#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/deepseek41/expert_cuda.hpp"
#include "strata/deepseek41/routing.hpp"

namespace strata::deepseek41 {
struct CacheStats {
    uint64_t hits = 0, misses = 0, evictions = 0, expert_h2d_bytes = 0, router_d2h_bytes = 0;
};
// One text token's routed branch of ONE layer. Reuses ExpertCache storage,
// ExpertSource, DeviceArena, BF16 GEMV, Q8_1/NVFP4 projections and add_inplace.
// Synchronous correctness baseline: host top-6/LRU, same GPU math on hits/misses.
// No shared expert, final BF16 cast, attention, prefill, CPU expert fallback or graph capture.
class CudaRoutedLayer {
  public:
    // BF16 router [384, hidden], text bias [384]. Copies router/bias at construction.
    // Source outlives this object and returns exactly nvfp4_expert_bytes(geometry)
    // bytes in [gate|up|down|scale tail] order, losslessly packed by pack_nvfp4_expert.
    // Host scale metadata follows each cache slot through hits and replacements.
    // 1..384 slots for this layer. Smaller dimensions are allowed for tests only.
    CudaRoutedLayer(const Geometry& geometry, int layer, std::span<const uint16_t> router,
                    std::span<const float> bias, core::ExpertSource& source, int slots, int device = 0);
    // Disjoint, 4-byte aligned device F32 input/output [hidden]; input rounded to BF16.
    // Same activation validity/device/stream contract as CudaExpert. Single caller.
    // Returns selected rank order; output accumulates BF16 expert outputs in ascending
    // expert-ID order into F32, matching the official MoE before its shared branch.
    // Waits before returning or propagating an execution error. After such an error,
    // reconstruct the object (a partially overwritten cache slot must never be reused).
    Route forward(const float* input, float* output, void* stream);
    const CacheStats& stats() const { return stats_; }
    size_t cache_bytes() const { return size_t(cache_.bytes()); }
    size_t device_bytes() const { return cache_bytes() + arena_->capacity() + expert_->device_bytes(); }
    int64_t resident() const { return cache_.resident(); }
    int32_t slot_of(int expert) const { return cache_.slot_of(layer_, expert); }

  private:
    int32_t ensure_slot(int32_t expert, void* stream);
    int layer_, hidden_, device_;
    size_t blob_bytes_;
    core::ExpertSource& source_;
    core::ExpertCache cache_;
    std::unique_ptr<CudaExpert> expert_;
    std::unique_ptr<core::DeviceArena> arena_;
    uint16_t *router_ = nullptr, *input_bf16_ = nullptr;
    float *logits_ = nullptr, *expert_output_ = nullptr;
    std::array<float, 384> bias_{}, host_logits_{};
    std::vector<int32_t> owners_;
    std::vector<std::array<float, 3>> scales_;
    std::vector<uint64_t> ages_;
    uint64_t tick_ = 0;
    CacheStats stats_;
    bool failed_ = false;
};
} // namespace strata::deepseek41
