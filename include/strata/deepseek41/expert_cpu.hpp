// One DeepSeek expert, using Strata's existing CPU projection/activation functions.
// MXFP4 weights + ggml Q8_0 activations: this is a distinct baseline from official FP8 activation math.
#pragma once

#include "strata/deepseek41/artifact.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

namespace strata::deepseek41 {

// Weights remain owned by the GgufModel behind ExpertView. That model must outlive this object.
// Scratch is sized to this expert, not the Qwen pool's fixed buffers. One caller at a time.
class CpuExpert {
  public:
    CpuExpert(const Geometry& geometry, const ExpertView& weights);
    // Float carriers of BF16 input/output. Route weight is applied BEFORE hidden BF16/Q8 rounding.
    // 1017's positive normalized top-6 weights sum to 1.5, so each lies in [0, 1.5].
    void forward(std::span<const float> input, float route_weight, std::span<float> output);
    size_t scratch_bytes() const;

  private:
    ExpertView weights_;
    std::array<kernels::cpu::NativeFmt, 3> projection_;
    std::vector<float> input_, gate_, up_, hidden_;
    std::vector<uint8_t> input_q_, hidden_q_;
    void project(int role, const void* activation, std::span<float> output) const;
};

} // namespace strata::deepseek41
