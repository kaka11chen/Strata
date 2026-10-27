#include "strata/deepseek41/expert_cpu.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <mutex>

namespace strata::deepseek41 {
namespace {
float bf16(float value) {
    return kernels::f32_from_bf16(kernels::bf16_from_f32(value));
}

kernels::cpu::NativeFmt projection(int input, int output, const ExpertPart& weight) {
    if (input <= 0 || output <= 0 || input % 32 || weight.type != GGML_TYPE_MXFP4)
        throw std::invalid_argument("DeepSeek CPU expert requires whole-block MXFP4 matrices");
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_0 || traits->nrows != 1)
        throw std::runtime_error("ggml lacks the expected MXFP4/Q8_0 CPU dot product");
    kernels::cpu::NativeFmt f;
    // native_down_rows is Strata's single-matrix projection entry. Use it separately for
    // gate/up/down: native_gu_rows would fuse Qwen's unclamped SwiGLU before we can round/clamp.
    f.d_type = GGML_TYPE_MXFP4;
    f.d_act = GGML_TYPE_Q8_0;
    f.n_ff = input;
    f.n_embd = output;
    f.d_row = ggml_row_size(GGML_TYPE_MXFP4, input);
    f.h_bytes = ggml_row_size(GGML_TYPE_Q8_0, input);
    f.bytes = f.d_row * size_t(output);
    if (weight.data.size() != f.bytes)
        throw std::invalid_argument("DeepSeek CPU expert matrix byte count disagrees with its dimensions");
    if (!weight.data.data())
        throw std::invalid_argument("DeepSeek CPU expert has no weight storage");
    for (size_t b = 0; b < weight.data.size(); b += 17)
        if (weight.data[b] == 255)
            throw std::invalid_argument("DeepSeek CPU expert has a nonfinite MXFP4 block scale");
    return f;
}
} // namespace

CpuExpert::CpuExpert(const Geometry& g, const ExpertView& weights) : weights_(weights) {
    // These dimensions also bound every allocation and the product used in the layout calculation.
    if (g.hidden <= 0 || g.hidden > 5120 || g.expert_width <= 0 || g.expert_width > 2304 || g.hidden % 32 ||
        g.expert_width % 32)
        throw std::invalid_argument("DeepSeek CPU expert dimensions must be whole blocks up to 5120/2304");
    static std::once_flag init;
    std::call_once(init, [] { ggml_cpu_init(); });
    projection_[0] = projection(g.hidden, g.expert_width, weights.parts[0]);
    projection_[1] = projection(g.hidden, g.expert_width, weights.parts[1]);
    projection_[2] = projection(g.expert_width, g.hidden, weights.parts[2]);
    input_.resize(size_t(g.hidden));
    gate_.resize(size_t(g.expert_width));
    up_.resize(size_t(g.expert_width));
    hidden_.resize(size_t(g.expert_width));
    input_q_.resize(projection_[0].h_bytes);
    hidden_q_.resize(projection_[2].h_bytes);
}

void CpuExpert::project(int role, const void* activation, std::span<float> output) const {
    const void* act[] = {activation};
    float* out[] = {output.data()};
    kernels::cpu::native_down_rows(projection_[size_t(role)], weights_.parts[size_t(role)].data.data(), act, 1, out, 0,
                                   int(output.size()));
}

void CpuExpert::forward(std::span<const float> input, float route_weight, std::span<float> output) {
    if (input.size() != input_.size() || output.size() != input_.size())
        throw std::invalid_argument("DeepSeek CPU expert input/output width mismatch");
    if (!std::isfinite(route_weight) || route_weight < 0.f || route_weight > 1.5f)
        throw std::invalid_argument("DeepSeek 1017 expert route weight must be finite and in [0, 1.5]");
    // ggml Q8_0 stores its scale in FP16. Refuse overflow rather than feed an infinite scale to its dot product.
    constexpr float max_q8 = 65504.f * 127.f;
    for (size_t i = 0; i < input.size(); ++i) {
        const float v = bf16(input[i]);
        if (!std::isfinite(v) || std::abs(v) > max_q8)
            throw std::invalid_argument("DeepSeek CPU expert input is outside the finite Q8_0 range");
        input_[i] = v;
    }
    kernels::cpu::native_quant_h(projection_[0], input_.data(), input_q_.data());
    project(0, input_q_.data(), gate_);
    project(1, input_q_.data(), up_);
    for (size_t i = 0; i < hidden_.size(); ++i) {
        const float gate = std::min(bf16(gate_[i]), 10.f);
        const float up = std::clamp(bf16(up_[i]), -10.f, 10.f);
        hidden_[i] = bf16(((gate / (1.f + std::exp(-gate))) * up) * route_weight);
        if (!std::isfinite(hidden_[i]))
            throw std::runtime_error("DeepSeek CPU expert produced a nonfinite hidden activation");
    }
    kernels::cpu::native_quant_h(projection_[2], hidden_.data(), hidden_q_.data());
    project(2, hidden_q_.data(), output);
    for (float& v : output) {
        v = bf16(v);
        if (!std::isfinite(v))
            throw std::runtime_error("DeepSeek CPU expert produced a nonfinite output");
    }
}

size_t CpuExpert::scratch_bytes() const {
    return (input_.size() + gate_.size() + up_.size() + hidden_.size()) * sizeof(float) + input_q_.size() +
           hidden_q_.size();
}
} // namespace strata::deepseek41
