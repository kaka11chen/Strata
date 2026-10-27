#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/engram_projection.hpp"

#include <memory>

namespace strata::deepseek41 {
struct EngramProjectionView {
    const float* kv;   // BF16-valued F32 [25600]: four keys, then one shared value
    const float* gate; // F32 [4], after optional token masking
};
// One token, original layer 1 or 14; sequence hashing and row I/O remain separate.
// Reuses FP8 activation conversion and Strata BF16 GEMV. Startup expands the
// projection losslessly in <=4096-row chunks, retaining a 300 MiB BF16 matrix.
class CudaEngram {
  public:
    explicit CudaEngram(const EngramWeights& weights, int device = 0);
    ~CudaEngram();
    // Device lookup [6144], residual/output [4,5120], mutually disjoint/aligned
    // and outside owned storage. F32 inputs are rounded at the BF16 boundaries.
    // Finite inputs/intermediates; lookup obeys quantize_fp8_bf16's range contract.
    // live=false suppresses the update; output is the BF16-rounded residual.
    // Synchronous, single caller, one explicit stream on the allocation device.
    // Argument rejection preserves output; execution failure poisons this object
    // and requires discarding output/reconstruction. No per-call allocation.
    // Returned read-only views last until the next forward or destruction.
    EngramProjectionView forward(const float* lookup, const float* residual,
                                 float* output, void* stream, bool live = true);
    int layer() const { return layer_; }
    size_t device_bytes() const { return arena_->capacity(); } // excludes CUDA context
  private:
    const int layer_;
    int device_;
    std::unique_ptr<core::DeviceArena> arena_;
    uint16_t *weights_ = nullptr, *activation_ = nullptr;
    float *product_ = nullptr, *kv_ = nullptr, *gate_ = nullptr;
    void* stream_ = nullptr;
    bool failed_ = false;
};
using CudaLayer1Engram = CudaEngram; // compatibility name for existing layer-1 callers
} // namespace strata::deepseek41
