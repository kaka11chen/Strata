#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/compressor_weights.hpp"

#include <memory>

namespace strata::deepseek41 {
struct CompressedBatch {
    int count;
    int first_position; // -1 if count=0; group i represents first_position + 2*i
};

// Fixed 1017 layer-2 ratio-2 compressor, batch size 1. Pre-RoPE latents only.
// Owns expanded F32 weights and bounded scratch. Uses cuBLAS F32 projection and
// Strata RMSNorm. No activation quantization, indexer or compressed KV cache here.
class CudaCompressor {
  public:
    explicit CudaCompressor(const CompressorWeights& weights, int device = 0);
    ~CudaCompressor();
    // F32 device input [tokens,5120], rounded to BF16 before F32 projection.
    // Finite inputs/intermediates within F32 range are required. Output is
    // BF16-valued F32 [groups,512], before RoPE. groups=(start_pos+tokens)/2-start_pos/2.
    // Output may be null when groups=0 and is then never accessed. Otherwise
    // both buffers must be disjoint, aligned and outside this object's arena.
    // First call: start_pos=0, tokens>=1. Later calls: one token at next_position().
    // Total positions <=1048576. One caller and one explicit owning-device stream.
    // Synchronous; no per-token allocation. Execution errors invalidate state and
    // output and require reconstruction; rejected arguments leave state unchanged.
    CompressedBatch forward(const float* input, float* output, int tokens, int start_pos, void* stream);
    void reset(void* stream); // same stream; discards any incomplete pair
    int next_position() const;
    size_t device_bytes() const; // excludes caller buffers, CUDA/cuBLAS resources
    // Read-only F32 device views [512], valid only while next_position() is odd,
    // until the next forward/reset. Null when there is no pending token.
    const float* pending_kv() const;
    const float* pending_scores() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::deepseek41
