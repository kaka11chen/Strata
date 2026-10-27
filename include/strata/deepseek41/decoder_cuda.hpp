#pragma once

#include "strata/deepseek41/attention_cuda.hpp"
#include "strata/deepseek41/mhc_cuda.hpp"

namespace strata::deepseek41 {
struct DecoderWeights {
    MhcWeights attention_mhc, ffn_mhc;
    AttentionWeights attention;
    SharedExpertWeights shared;
    std::span<const uint16_t> router;
    std::span<const float> bias;
};

// Fixed 1017 text layer 0: mHC -> attention -> mHC -> MoE -> residual.
// Owns all component device weights/state; ExpertSource must outlive it.
// No Engram, compressed/shared KV, all-layer loop, embedding or output head.
class CudaDecoder {
  public:
    CudaDecoder(const Geometry& geometry, const DecoderWeights& weights,
                core::ExpertSource& source, int slots, int device = 0);
    // Synchronous, one caller and one explicit stream per instance. Device F32
    // input/output [tokens,4,5120], incoming/next_pre [tokens,4], all disjoint,
    // 4-byte aligned, finite and within the component activation contracts.
    // incoming_pre belongs to the SAME token in the preceding layer. next_pre
    // goes to the SAME token in the following layer, never to the next token.
    // First call is prefill (start_pos=0); later calls are contiguous one-token
    // decode. At most 1048576 positions. Optional host routes has exactly tokens
    // entries. Owned device memory is fixed, including during long prefill.
    // Validation errors preserve state; execution errors poison the whole layer
    // and invalidate outputs from that call. Reconstruct after execution failure.
    void forward(const float* input, const float* incoming_pre, float* output, float* next_pre,
                 int tokens, int start_pos, void* stream, std::span<Route> routes = {});
    // Clears sequence KV/position. Immutable expert weights stay cached and cache
    // statistics remain cumulative. Same stream; a poisoned layer cannot reset.
    void reset(void* stream);
    int next_position() const { return next_; }
    const CacheStats& stats() const { return moe_.stats(); }
    size_t device_bytes() const;
    const uint16_t* kv_cache() const { return attention_.kv_cache(); }
  private:
    int device_, next_ = 0;
    void* stream_ = nullptr;
    bool failed_ = false;
    CudaMhc attention_mhc_;
    CudaAttention attention_;
    CudaMhc ffn_mhc_;
    CudaMoE moe_;
    core::DeviceArena arena_;
    float *attention_output_, *middle_;
    void check_stream(void* stream) const;
};
} // namespace strata::deepseek41
