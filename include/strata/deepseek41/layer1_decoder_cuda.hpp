#pragma once

#include "strata/deepseek41/decoder_cuda.hpp"
#include "strata/deepseek41/engram.hpp"
#include "strata/deepseek41/engram_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 text layer 1. Owns Engram history/readers/projection, mHC, pure-SWA
// attention and MoE. Original ExpertSource must outlive this object. No image
// routing or token mask: every supplied token is an ordinary text token.
class CudaLayer1Decoder {
  public:
    CudaLayer1Decoder(const Geometry& geometry, const DecoderWeights& weights,
                     const Layer1EngramWeights& engram, std::span<const uint32_t> token_map,
                     const std::string& table_path, core::ExpertSource& source, int slots,
                     const EngramIoOptions& io = {}, int device = 0);
    // Host original tokens [T]; device F32 residual/output [T,4,5120] and
    // incoming/next_pre [T,4]. All buffers disjoint/aligned; optional host routes
    // has T entries. incoming_pre belongs to this token in the PREVIOUS layer.
    // Engram updates the residual before mHC, preserving incoming coefficients.
    // First call is prefill; later calls are contiguous one-token decode, up to
    // 1048576 positions. Single caller, synchronous, one explicit device stream.
    // Finite inputs and existing component range contracts apply. Files immutable.
    // All argument/token checks precede mutation. I/O or execution failure poisons
    // the complete layer; outputs from that call are invalid and it must be rebuilt.
    void forward(std::span<const int32_t> tokens, const float* input, const float* incoming_pre,
                 float* output, float* next_pre, int start_pos, void* stream, std::span<Route> routes = {});
    // Clear token history and KV/position; retain row/expert caches and counters.
    // Reset cannot revive a poisoned layer and must use the same stream.
    void reset(void* stream);
    int next_position() const { return next_; }
    const CacheStats& stats() const { return moe_.stats(); }
    std::array<ngram::ReaderStats,2> engram_stats() const { return table_.stats(); }
    size_t device_bytes() const;
    const uint16_t* kv_cache() const { return attention_.kv_cache(); }
  private:
    int device_, next_ = 0;
    void* stream_ = nullptr;
    bool failed_ = false;
    Layer1EngramHash hash_;
    Layer1EngramTable table_;
    CudaLayer1Engram engram_;
    CudaMhc attention_mhc_;
    CudaAttention attention_;
    CudaMhc ffn_mhc_;
    CudaMoE moe_;
    core::DeviceArena arena_;
    float *lookup_, *engram_residual_, *attention_output_, *middle_;
    std::array<uint32_t,24> rows_{};
    std::array<uint16_t,6144> lookup_bits_{};
    std::array<float,6144> lookup_values_{};
    void check_stream(void* stream) const;
};
} // namespace strata::deepseek41
