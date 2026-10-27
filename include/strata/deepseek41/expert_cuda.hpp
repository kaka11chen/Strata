#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/artifact.hpp"

#include <memory>

namespace strata::deepseek41 {

// One token/expert, CUDA only. Owns a lossless NVFP4 copy of the input MXFP4 weights.
// Uses Strata's DeviceArena, Q8_1 quantizer and iq_mmvq projections. All tensor math stays on the GPU.
class CudaExpert {
  public:
    CudaExpert(const Geometry& geometry, const ExpertView& weights, int device = 0);
    // Scratch only; forward_from_blob reads a caller-owned standard NVFP4 cache slot.
    // Its host scales MUST come from that same packed slot and follow it through
    // replacement; no device-to-host scale read or per-call allocation is needed.
    explicit CudaExpert(const Geometry& geometry, int device = 0);
    void forward_from_blob(const uint8_t* blob, size_t bytes, const std::array<float, 3>& scales,
                           const float* input, float route_weight,
                           float* output, void* stream);
    // Device float buffers, each geometry.hidden elements, at least 4-byte aligned.
    // Input is rounded to BF16. Its block scales and sums must be finite/FP16-representable,
    // as required by native_quantize_q8_1. The caller owns this activation validity contract.
    // Current CUDA device and the explicit non-null stream must match the allocation device.
    // Enqueues without allocation or waiting.
    // One in-flight call per instance. Caller waits before destroying it or its buffers.
    void forward(const float* input, float route_weight, float* output, void* stream);
    size_t weight_bytes() const { return weight_bytes_; }
    size_t scratch_bytes() const { return scratch_bytes_; }
    size_t device_bytes() const { return arena_->capacity() + (weight_arena_ ? weight_arena_->capacity() : 0); }

  private:
    void forward_weights(const std::array<const void*, 3>& weights, const std::array<float, 3>& scales,
                         const float* input, float route_weight,
                         float* output, void* stream);
    int hidden_ = 0, expert_width_ = 0;
    size_t weight_bytes_ = 0, scratch_bytes_ = 0, blob_bytes_ = 0;
    std::unique_ptr<core::DeviceArena> arena_, weight_arena_;
    std::array<const void*, 3> weights_{};
    std::array<float, 3> scales_{1.f, 1.f, 1.f};
    float *input_ = nullptr, *gate_ = nullptr, *up_ = nullptr, *intermediate_ = nullptr;
    void *input_q_ = nullptr, *hidden_q_ = nullptr;
};

} // namespace strata::deepseek41
