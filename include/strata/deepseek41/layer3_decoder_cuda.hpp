#pragma once

#include "strata/deepseek41/decoder_cuda.hpp"
#include "strata/deepseek41/layer3_attention_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 text consumers 3..7/9..13/15..19/21..23/25..27/29..31/33..35/37..39 and index sources 24/28/32/36: mHC/attention/MoE/residual joins. Borrows
// the original source-2/8/14/20/24/28/32/36 completed shared attention batch; owns only its window ring and expert cache.
// ExpertSource must outlive this instance. No Engram or all-layer generation.
class CudaSharedDecoder {
  public:
    CudaSharedDecoder(const Geometry& geometry, const DecoderWeights& weights,
                     core::ExpertSource& source, int slots, int device = 0, int layer = 3);
    CudaSharedDecoder(const Geometry& geometry, const DecoderWeights& weights,
                     const CandidateIndexQueryWeights& query, core::ExpertSource& source, int slots,
                     int capacity_tokens, int prefill_capacity, int device = 0);
    SharedAttentionBatch shared_batch() const { return attention_.shared_batch(); } // index sources 24/28/32/36 only
    // External aligned, disjoint F32 residuals [tokens,4,5120] and coefficients
    // [tokens,4]. Input/incoming are the PRECEDING LAYER's outputs for these same
    // tokens, not the preceding token's coefficients. Finite-input preconditions
    // match CudaDecoder. None of the buffers may alias either layer's caches.
    // The publication must match the exact batch, source sequence, stream/device.
    // Consume this layer before advancing/resetting/destroying the source. One
    // caller serializes both lifetimes. Fixed scratch; synchronous execution.
    // Invalid arguments preserve state; execution failure poisons this consumer
    // and invalidates partial output, without changing the source publication.
    void forward(const float* input, const float* incoming_pre, float* output, float* next_pre,
                 int tokens, int start_pos, const SharedAttentionBatch& shared, void* stream,
                 std::span<Route> routes = {});
    void forward_source20(const float* input, const float* incoming_pre, float* output, float* next_pre,
                 int tokens, int start_pos, const Source20AttentionBatch& shared, void* stream,
                 std::span<Route> routes = {});
    void reset(void* stream); // own state only; retains expert cache and cumulative stats
    int next_position() const { return attention_.next_position(); }
    const CacheStats& stats() const { return moe_.stats(); }
    size_t device_bytes() const; // excludes borrowed source storage
    const uint16_t* kv_cache() const { return attention_.kv_cache(); }
  private:
    int device_;
    void* stream_ = nullptr;
    bool failed_ = false;
    CudaMhc attention_mhc_;
    CudaSharedAttention attention_;
    CudaMhc ffn_mhc_;
    CudaMoE moe_;
    core::DeviceArena arena_;
    float *attention_output_, *middle_;
    void check_stream(void* stream) const;
};
// Preserve the step-6i entry point; other consumers require an explicit layer.
using CudaLayer3Decoder = CudaSharedDecoder;
} // namespace strata::deepseek41
