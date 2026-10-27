#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/shared_weights.hpp"

#include <memory>

namespace strata::deepseek41 {
// One 32-value scale per activation block. Round input to BF16, quantize to
// E4M3FN with ceil-power-of-two E8M0 scales, restore into BF16 for existing GEMV.
// Device buffers; input finite after BF16 rounding with |x| <= 2^120. Caller
// owns this validity contract. n is a positive multiple of 32, at most 32768.
// No allocation/wait. Current device/nonnull stream must own the disjoint buffers.
void quantize_fp8_bf16(const float* input, uint16_t* output, int n, void* stream);

class CudaSharedExpert {
  public:
    // Startup expands FP8 weights losslessly to BF16 and owns that device copy.
    CudaSharedExpert(const Geometry& geometry, const SharedExpertWeights& weights, int device = 0);
    // F32 device input/output [hidden], disjoint and 4-byte aligned. Explicit
    // stream on the allocation device; single caller, wait before destruction.
    // Uses FP8 activations and existing BF16 GEMV, with FP32 accumulation order
    // different from the official FP8 GEMM. No route/scalar sigmoid gating.
    void forward(const float* input, float* output, void* stream);
    size_t weight_bytes() const { return weight_bytes_; }
    size_t device_bytes() const { return arena_->capacity(); }
  private:
    int hidden_, width_, device_;
    size_t weight_bytes_;
    std::unique_ptr<core::DeviceArena> arena_;
    std::array<const uint16_t*, 3> weights_{};
    uint16_t *input_q_ = nullptr, *hidden_q_ = nullptr;
    float *gate_ = nullptr, *up_ = nullptr, *hidden_values_ = nullptr;
};
} // namespace strata::deepseek41
