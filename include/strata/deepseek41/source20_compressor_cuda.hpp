#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/source20_contract.hpp"

#include <memory>

namespace strata::deepseek41 {

// Fixed 1017 source-20 ratio-1 compressor, batch size one, before RoPE.
// Reuses Strata BF16 GEMV/conversion/RMSNorm; owns BF16 projection weights and
// constant scratch. No gate, pair state, candidate selection or KV cache.
class CudaSource20Compressor {
  public:
    CudaSource20Compressor(const Source20CompressorWeights& weights,
                           int context_capacity, int prefill_capacity, int device = 0);
    ~CudaSource20Compressor();

    // F32 device input [tokens,5120], rounded to BF16 before GEMV. Both projection
    // and normalized output round to BF16. Output is BF16-valued F32 [tokens,512].
    // Inputs after BF16 rounding and all intermediates must remain finite.
    // Buffers must be aligned, disjoint, on the owning device and outside this
    // object's arena. One caller and one explicit stream. Synchronous, with no
    // per-call allocation. Initial bounded prefill, then contiguous single-token
    // decode as described by Source20Batch. It cannot be used as a ratio-2 batch.
    // Rejected arguments leave state/output unchanged. Execution exceptions
    // invalidate output and the instance; reconstruction is then required.
    Source20Batch forward(const float* input, float* output, int tokens, int start, void* stream);
    void reset(void* stream); // Same stream; invalidates the last projection view.
    int next_position() const;
    size_t device_bytes() const; // Excludes caller buffers and CUDA runtime resources.

    // Diagnostic read-only device view [512]: BF16-valued projection of the
    // last token, before RMSNorm. Valid after successful forward until the next
    // forward/reset/destruction. Null before forward, after reset or on failure.
    const float* last_projection() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::deepseek41
