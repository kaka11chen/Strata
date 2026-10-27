#include "strata/deepseek41/expert_cuda.hpp"
#include "expert_ops.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace strata::deepseek41 {
namespace {
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
__device__ float bf16(float x) {
    return kernels::f32_from_bf16(kernels::bf16_from_f32(x));
}
// Model-specific elementwise glue only; all three matrix projections use Strata's existing kernels.
__global__ void round_bf16(const float* input, float* output, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n)
        output[i] = bf16(input[i]);
}
__global__ void swiglu_1017(const float* gate, const float* up, float route, float* hidden, int n) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n)
        return;
    const float g = fminf(bf16(gate[i]), 10.f);
    const float u = fmaxf(-10.f, fminf(bf16(up[i]), 10.f));
    hidden[i] = bf16(((g / (1.f + expf(-g))) * u) * route);
}
size_t aligned(size_t n) {
    return (n + 255) / 256 * 256;
}
} // namespace

namespace detail {
void round_bf16_values(const float* input, float* output, int n, void* stream) {
    round_bf16<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(input, output, n);
    ck(cudaGetLastError(), "DeepSeek BF16 rounding");
}
void swiglu_1017_values(const float* gate, const float* up, float route, float* hidden, int n, void* stream) {
    swiglu_1017<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(gate, up, route, hidden, n);
    ck(cudaGetLastError(), "DeepSeek clamped SwiGLU");
}
} // namespace detail

CudaExpert::CudaExpert(const Geometry& g, const ExpertView& weights, int device) : CudaExpert(g, device) {
    std::vector<uint8_t> packed(blob_bytes_);
    pack_nvfp4_expert(g, weights, packed);
    scales_ = nvfp4_expert_scales(packed);
    weight_arena_ = std::make_unique<core::DeviceArena>(blob_bytes_, device);
    auto* dst = static_cast<uint8_t*>(weight_arena_->alloc(blob_bytes_));
    ck(cudaMemcpy(dst, packed.data(), packed.size(), cudaMemcpyHostToDevice), "upload DeepSeek expert");
    // Pageable H2D cudaMemcpy may return after host staging, before the device
    // copy finishes. A later nonblocking execution stream does not wait for it.
    ck(cudaStreamSynchronize(nullptr), "wait for DeepSeek expert upload");
    weight_bytes_ = blob_bytes_;
    for (size_t role = 0; role < 3; ++role)
        weights_[role] = dst + role * ((blob_bytes_ - nvfp4_scale_tail_bytes) / 3);
}

CudaExpert::CudaExpert(const Geometry& g, int device)
    : hidden_(g.hidden), expert_width_(g.expert_width), blob_bytes_(nvfp4_expert_bytes(g)) {
    const size_t input_bytes = size_t(g.hidden) * sizeof(float), ff_bytes = size_t(g.expert_width) * sizeof(float);
    const size_t input_q_bytes = kernels::native_q8_1_bytes(g.hidden),
                 hidden_q_bytes = kernels::native_q8_1_bytes(g.expert_width);
    scratch_bytes_ = input_bytes + 3 * ff_bytes + input_q_bytes + hidden_q_bytes;
    size_t capacity = 0;
    for (size_t bytes : {input_bytes, ff_bytes, ff_bytes, ff_bytes, input_q_bytes, hidden_q_bytes})
        capacity = aligned(capacity) + bytes;
    arena_ = std::make_unique<core::DeviceArena>(capacity, device);
    input_ = static_cast<float*>(arena_->alloc(input_bytes));
    gate_ = static_cast<float*>(arena_->alloc(ff_bytes));
    up_ = static_cast<float*>(arena_->alloc(ff_bytes));
    intermediate_ = static_cast<float*>(arena_->alloc(ff_bytes));
    input_q_ = arena_->alloc(input_q_bytes);
    hidden_q_ = arena_->alloc(hidden_q_bytes);
}

void CudaExpert::forward(const float* input, float route_weight, float* output, void* stream) {
    if (!weight_arena_)
        throw std::logic_error("Scratch-only CUDA expert needs forward_from_blob");
    forward_weights(weights_, scales_, input, route_weight, output, stream);
}

void CudaExpert::forward_from_blob(const uint8_t* blob, size_t bytes, const std::array<float, 3>& scales,
                                   const float* input, float route_weight,
                                   float* output, void* stream) {
    if (!blob || uintptr_t(blob) % 4 || bytes != blob_bytes_)
        throw std::invalid_argument("DeepSeek CUDA cache blob has the wrong size or alignment");
    const size_t part = (bytes - nvfp4_scale_tail_bytes) / 3;
    forward_weights({blob, blob + part, blob + 2 * part}, scales, input, route_weight, output, stream);
}

void CudaExpert::forward_weights(const std::array<const void*, 3>& weights, const std::array<float, 3>& scales,
                                 const float* input,
                                 float route_weight, float* output, void* stream) {
    if (!input || !output || !stream || uintptr_t(input) % 4 || uintptr_t(output) % 4)
        throw std::invalid_argument("DeepSeek CUDA expert requires aligned device buffers and a non-null stream");
    if (!std::isfinite(route_weight) || route_weight < 0.f || route_weight > 1.5f)
        throw std::invalid_argument("DeepSeek 1017 expert route weight must be finite and in [0, 1.5]");
    for (float scale : scales)
        if (scale != 1.f && scale != .5f)
            throw std::invalid_argument("DeepSeek CUDA expert needs exact supported projection scales");
    detail::round_bf16_values(input, input_, hidden_, stream);
    kernels::native_quantize_q8_1(input_, input_q_, hidden_, 1, stream);
    kernels::iq_mmvq(40, weights[0], input_q_, gate_, hidden_, expert_width_, 1, stream);
    if (scales[0] != 1.f) kernels::scale_inplace(gate_, expert_width_, scales[0], stream);
    kernels::iq_mmvq(40, weights[1], input_q_, up_, hidden_, expert_width_, 1, stream);
    if (scales[1] != 1.f) kernels::scale_inplace(up_, expert_width_, scales[1], stream);
    detail::swiglu_1017_values(gate_, up_, route_weight, intermediate_, expert_width_, stream);
    kernels::native_quantize_q8_1(intermediate_, hidden_q_, expert_width_, 1, stream);
    kernels::iq_mmvq(40, weights[2], hidden_q_, output, expert_width_, hidden_, 1, stream);
    if (scales[2] != 1.f) kernels::scale_inplace(output, hidden_, scales[2], stream);
    detail::round_bf16_values(output, output, hidden_, stream);
}
} // namespace strata::deepseek41
