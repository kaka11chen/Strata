#pragma once

#include "strata/deepseek41/decoder_cuda.hpp"
#include "strata/deepseek41/source20_attention_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 text source 20, with ratio-1 compression and owned KV/index/candidate
// publication. Reuses the existing mHC, attention, MoE and Strata expert-cache paths.
// ExpertSource must outlive this instance. No Engram or all-layer generation.
class CudaSource20Decoder {
  public:
    CudaSource20Decoder(const Geometry& geometry, const DecoderWeights& weights,
                     const Source20CompressorWeights& compressor, const Source20IndexKeyWeights& key,
                     const Source20IndexQueryWeights& query, core::ExpertSource& source, int slots,
                     int context_capacity, int prefill_capacity, int device = 0);
    // Same buffer, stream, finite-input and sequence contract as CudaDecoder.
    // All four buffers are external and disjoint, including from exposed caches.
    // Two passes over input: hc_pre/norm + compressor, then full decoder tokens.
    // Scratch is fixed at startup; no prompt-sized hidden/intermediate array.
    // Capacity/argument rejections preserve state and the last shared publication.
    // Execution failures invalidate outputs/publication and poison the whole layer.
    void forward(const float* input, const float* incoming_pre, float* output, float* next_pre,
                 int tokens, int start_pos, void* stream, std::span<Route> routes = {});
    void reset(void* stream); // clears sequence state; keeps expert cache and cumulative stats
    int next_position() const { return attention_.next_position(); }
    int capacity() const { return attention_.capacity(); }
    int size() const { return attention_.size(); } // KV/index entries, one per token
    const CacheStats& stats() const { return moe_.stats(); }
    size_t device_bytes() const;
    const uint16_t* kv_cache() const { return attention_.kv_cache(); }
    const uint16_t* compressed_kv_cache() const { return attention_.compressed_kv_cache(); }
    const uint16_t* index_key_cache() const { return attention_.index_key_cache(); }
    // Available only after the ENTIRE decoder call succeeds. Consume dependent layers before advancing.
    Source20AttentionBatch shared_batch() const { return attention_.shared_batch(); }
  private:
    int device_;
    void* stream_ = nullptr;
    bool failed_ = false;
    CudaMhc attention_mhc_;
    CudaSource20Attention attention_;
    CudaMhc ffn_mhc_;
    CudaMoE moe_;
    core::DeviceArena arena_;
    float *attention_output_, *middle_;
    void check_stream(void* stream) const;
};
} // namespace strata::deepseek41
