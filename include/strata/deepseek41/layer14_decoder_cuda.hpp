#pragma once

#include "strata/deepseek41/decoder_cuda.hpp"
#include "strata/deepseek41/engram.hpp"
#include "strata/deepseek41/engram_cuda.hpp"
#include "strata/deepseek41/layer2_attention_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 text layer 14: Engram before mHC, owned ratio-2 KV/index, then MoE.
// Reuses the confirmed component kernels and Strata row/expert caches.
// ExpertSource and immutable table files must outlive this instance.
class CudaLayer14Decoder {
  public:
    CudaLayer14Decoder(const Geometry& geometry, const DecoderWeights& weights,
                      const EngramWeights& engram, const CompressorWeights& compressor,
                      const IndexKeyWeights& key, const IndexQueryWeights& query,
                      std::span<const uint32_t> token_map, const std::string& table_path,
                      core::ExpertSource& source, int slots, int capacity_groups,
                      const EngramIoOptions& io = {}, int device = 0, int shared_prefill_capacity = 0);
    // Host original text tokens [T]; external disjoint F32 device residual/output
    // [T,4,5120] and incoming/next_pre [T,4]. Optional host routes has T entries.
    // incoming_pre belongs to the same token in the previous layer. All argument,
    // token and capacity checks precede mutation/I/O. Prefill then contiguous
    // single-token decode, one explicit stream, synchronous and single caller.
    // Two Engram/hash passes avoid a prompt-sized residual buffer: compress all
    // normalized tokens first, then execute attention/MoE in original prefill layout.
    // Owned scratch is fixed at startup. Component finite-value contracts apply.
    // Only a complete call publishes. I/O/execution failure poisons the whole
    // decoder, expires publications and invalidates this call's outputs.
    void forward(std::span<const int32_t> tokens, const float* input, const float* incoming_pre,
                 float* output, float* next_pre, int start_pos, void* stream, std::span<Route> routes = {});
    void reset(void* stream); // clear both hash histories and attention; retain bounded caches/stats
    int next_position() const { return attention_.next_position(); }
    int compressed_groups() const { return attention_.compressed_groups(); }
    int capacity() const { return attention_.capacity(); }
    const CacheStats& stats() const { return moe_.stats(); }
    std::array<ngram::ReaderStats, 2> engram_stats() const { return table_.stats(); }
    size_t device_bytes() const;
    const uint16_t* kv_cache() const { return attention_.kv_cache(); }
    const uint16_t* compressed_kv_cache() const { return attention_.compressed_kv_cache(); }
    const uint16_t* index_key_cache() const { return attention_.index_key_cache(); }
    SharedAttentionBatch shared_batch() const { return attention_.shared_batch(); }
  private:
    int device_;
    void* stream_ = nullptr;
    bool failed_ = false;
    EngramHash compressor_hash_, hash_;
    EngramTable table_;
    CudaEngram engram_;
    CudaMhc attention_mhc_;
    CudaSourceAttention attention_;
    CudaMhc ffn_mhc_;
    CudaMoE moe_;
    core::DeviceArena arena_;
    float *lookup_, *engram_residual_, *attention_output_, *middle_;
    std::array<uint32_t, 24> rows_{};
    std::array<uint16_t, 6144> lookup_bits_{};
    std::array<float, 6144> lookup_values_{};
    void check_stream(void* stream) const;
    void prepare_engram(EngramHash& hash, std::span<const int32_t> token,
                        const float* residual, int position, void* stream);
};
} // namespace strata::deepseek41
