#pragma once

#include "strata/deepseek41/routed_cuda.hpp"
#include "strata/deepseek41/shared_cuda.hpp"

namespace strata::deepseek41 {
// One token's complete text MoE: original routed top-6 + one unweighted shared
// expert, then a single BF16 cast. Routed experts retain the tested Q8_1 baseline;
// shared experts use FP8 activations with losslessly expanded BF16 weights.
// This is not an attention/decoder layer and does not claim official full-MoE FP8 parity.
class CudaMoE {
  public:
    CudaMoE(const Geometry& geometry, int layer, std::span<const uint16_t> router, std::span<const float> bias,
            core::ExpertSource& source, int slots, const SharedExpertWeights& shared, int device = 0);
    // Same input/stream contract as CudaRoutedLayer; synchronous, one caller.
    // Any execution failure invalidates this instance. Source outlives it.
    Route forward(const float* input, float* output, void* stream);
    int hidden_size() const { return hidden_; }
    const CacheStats& stats() const { return routed_.stats(); }
    size_t device_bytes() const { return routed_.device_bytes() + shared_.device_bytes() + arena_.capacity(); }
  private:
    int hidden_;
    CudaRoutedLayer routed_;
    CudaSharedExpert shared_;
    core::DeviceArena arena_;
    float* shared_output_;
    bool failed_ = false;
};
} // namespace strata::deepseek41
