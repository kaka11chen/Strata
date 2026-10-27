#pragma once

#include "strata/deepseek41/decoder_cuda.hpp"
#include "strata/deepseek41/layer2_attention_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 text source layers 2 and 8, with ratio-2 compression and owned shared KV/index
// state. Reuses the existing mHC, attention, MoE and Strata expert-cache paths.
// ExpertSource must outlive this instance. No Engram or all-layer generation.
class CudaSourceDecoder {
  public:
    CudaSourceDecoder(const Geometry& geometry, const DecoderWeights& weights,
                     const CompressorWeights& compressor, const IndexKeyWeights& key,
                     const IndexQueryWeights& query, core::ExpertSource& source, int slots,
                     int capacity_groups, int device = 0, int shared_prefill_capacity = 0, int layer = 2);
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
    int compressed_groups() const { return attention_.compressed_groups(); }
    const CacheStats& stats() const { return moe_.stats(); }
    size_t device_bytes() const;
    const uint16_t* kv_cache() const { return attention_.kv_cache(); }
    const uint16_t* compressed_kv_cache() const { return attention_.compressed_kv_cache(); }
    const uint16_t* index_key_cache() const { return attention_.index_key_cache(); }
    // Available only with explicit shared_prefill_capacity, and only after the
    // ENTIRE decoder call succeeds. Consume dependent layers before advancing.
    SharedAttentionBatch shared_batch() const { return attention_.shared_batch(); }
  private:
    int device_;
    void* stream_ = nullptr;
    bool failed_ = false;
    CudaMhc attention_mhc_;
    CudaSourceAttention attention_;
    CudaMhc ffn_mhc_;
    CudaMoE moe_;
    core::DeviceArena arena_;
    float *attention_output_, *middle_;
    void check_stream(void* stream) const;
};
using CudaLayer2Decoder = CudaSourceDecoder;
} // namespace strata::deepseek41
