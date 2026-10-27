#include "strata/deepseek41/shared_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace strata::deepseek41 {
namespace {
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
__global__ void fp8_restore(const float* input, uint16_t* output) {
    // One warp per 32-value group. CUDA's E4M3 conversion has a software path
    // on SM75; this does not require native FP8 Tensor Cores.
    const int i = int(blockIdx.x) * 32 + int(threadIdx.x);
    const float x = kernels::f32_from_bf16(kernels::bf16_from_f32(input[i]));
    float amax = fmaxf(fabsf(x), 1e-4f);
    for (int mask = 16; mask; mask >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    const unsigned bits = __float_as_uint(amax * (1.f / 448.f));
    const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
    const float scale = __uint_as_float(unsigned(exponent + 127) << 23);
    const float scaled = fmaxf(-448.f, fminf(448.f, x / scale));
    const auto code = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    const float restored = __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)) * scale;
    output[i] = kernels::bf16_from_f32(restored);
}
} // namespace

void quantize_fp8_bf16(const float* input, uint16_t* output, int n, void* stream) {
    const uintptr_t in = uintptr_t(input), out = uintptr_t(output);
    if (!input || !output || !stream || in % 4 || out % 2 || n <= 0 || n > 32768 || n % 32 ||
        (in <= out ? out - in < size_t(n) * 4 : in - out < size_t(n) * 2))
        throw std::invalid_argument("DeepSeek FP8 activation needs disjoint aligned buffers, whole 32-value blocks and a stream");
    fp8_restore<<<n / 32, 32, 0, static_cast<cudaStream_t>(stream)>>>(input, output);
    ck(cudaGetLastError(), "DeepSeek FP8 activation quantization");
}

CudaSharedExpert::CudaSharedExpert(const Geometry& g, const SharedExpertWeights& weights, int device)
    : hidden_(g.hidden), width_(g.expert_width), device_(device), weight_bytes_(0) {
    if (hidden_ <= 0 || hidden_ > 5120 || width_ <= 0 || width_ > 2304 || hidden_ % 32 || width_ % 32)
        throw std::invalid_argument("DeepSeek shared expert dimensions must be whole 32-value blocks up to 5120/2304");
    const size_t matrix_bytes = size_t(hidden_) * width_ * sizeof(uint16_t);
    for (int role = 0; role < 3; ++role)
        if (weights.parts[size_t(role)].rows != (role == 2 ? hidden_ : width_) ||
            weights.parts[size_t(role)].cols != (role == 2 ? width_ : hidden_))
            throw std::invalid_argument("DeepSeek shared matrix dimensions disagree with geometry");
    weight_bytes_ = 3 * matrix_bytes;
    const size_t x_bytes = size_t(hidden_) * 2, h_bytes = size_t(width_) * 2, f_bytes = size_t(width_) * 4;
    arena_ = std::make_unique<core::DeviceArena>(3 * aligned(matrix_bytes) + aligned(x_bytes) + aligned(h_bytes) +
                                                2 * aligned(f_bytes) + f_bytes, device);
    for (size_t role = 0; role < 3; ++role) {
        const auto packed = fp8_block32_to_bf16(weights.parts[role]);
        auto* dst = static_cast<uint16_t*>(arena_->alloc(matrix_bytes));
        ck(cudaMemcpy(dst, packed.data(), matrix_bytes, cudaMemcpyHostToDevice), "upload DeepSeek shared weights");
        // Startup upload must be complete before the host buffer is reused and
        // before a caller launches on an independent nonblocking stream.
        ck(cudaStreamSynchronize(nullptr), "wait for DeepSeek shared upload");
        weights_[role] = dst;
    }
    input_q_ = static_cast<uint16_t*>(arena_->alloc(x_bytes));
    hidden_q_ = static_cast<uint16_t*>(arena_->alloc(h_bytes));
    gate_ = static_cast<float*>(arena_->alloc(f_bytes));
    up_ = static_cast<float*>(arena_->alloc(f_bytes));
    hidden_values_ = static_cast<float*>(arena_->alloc(f_bytes));
}

void CudaSharedExpert::forward(const float* input, float* output, void* stream) {
    const uintptr_t in = uintptr_t(input), out = uintptr_t(output);
    if (!input || !output || !stream || in % 4 || out % 4 ||
        (in > out ? in - out : out - in) < size_t(hidden_) * 4)
        throw std::invalid_argument("DeepSeek shared expert needs aligned disjoint buffers and an explicit stream");
    int device = -1;
    ck(cudaGetDevice(&device), "get DeepSeek shared device");
    if (device != device_)
        throw std::invalid_argument("DeepSeek shared expert called on the wrong device");
    quantize_fp8_bf16(input, input_q_, hidden_, stream);
    kernels::bf16_gemv_pair(input_q_, weights_[0], gate_, width_, weights_[1], up_, width_, hidden_, stream);
    detail::swiglu_1017_values(gate_, up_, 1.f, hidden_values_, width_, stream);
    quantize_fp8_bf16(hidden_values_, hidden_q_, width_, stream);
    kernels::bf16_gemv(hidden_q_, weights_[2], output, width_, hidden_, stream);
    detail::round_bf16_values(output, output, hidden_, stream);
}
} // namespace strata::deepseek41
