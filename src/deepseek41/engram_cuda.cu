#include "strata/deepseek41/engram_cuda.hpp"
#include "strata/deepseek41/shared_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int dim = 5120, copies = 4, input = 6144, projected = 25600;
constexpr size_t matrix_bytes = size_t(projected) * input * 2;
constexpr size_t owned_bytes = matrix_bytes + copies * dim * 4 + input * 2 + projected * 4 + copies * 4;
void ck(cudaError_t status, const char* label) {
    if (status != cudaSuccess) throw core::CudaError(std::string(label) + ": " + cudaGetErrorString(status), int(status));
}
bool valid(const void* p, size_t bytes, size_t alignment = 4) {
    return p && uintptr_t(p) % alignment == 0 && bytes <= UINTPTR_MAX - uintptr_t(p);
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn;
}
__device__ float rounded(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
__device__ float warp_sum(float x) {
    for (int shift = 16; shift; shift >>= 1) x += __shfl_xor_sync(0xffffffffu, x, shift);
    return x;
}
// Normalize each of the four streams separately. Keep q*k before h*weight*key,
// apply rstd after the dot, and preserve copysign at zero as in Engram.forward.
// Qwen PLE normalizes before its dot, uses sign(0)=0 and has an extra convolution.
__global__ void gated_residual(const float* residual, const float* kv, const float* product,
                               float* gate, float* output, bool live) {
    const int stream = int(blockIdx.x), lane = int(threadIdx.x) & 31;
    const int base = stream * dim;
    float hh = 0.f, kk = 0.f, dot = 0.f;
    for (int d = int(threadIdx.x); d < dim; d += int(blockDim.x)) {
        const float h = rounded(residual[base+d]), k = kv[base+d];
        hh += __fmul_rn(h,h); kk += __fmul_rn(k,k);
        dot += __fmul_rn(__fmul_rn(h, product[base+d]), k);
    }
    __shared__ float sums[3][8];
    hh = warp_sum(hh); kk = warp_sum(kk); dot = warp_sum(dot);
    if (!lane) { sums[0][threadIdx.x/32] = hh; sums[1][threadIdx.x/32] = kk; sums[2][threadIdx.x/32] = dot; }
    __syncthreads();
    hh = warp_sum(lane < 8 ? sums[0][lane] : 0.f);
    kk = warp_sum(lane < 8 ? sums[1][lane] : 0.f);
    dot = warp_sum(lane < 8 ? sums[2][lane] : 0.f);
    const float rstd = __fmul_rn(rsqrtf(hh / float(dim) + 1e-20f), rsqrtf(kk / float(dim) + 1e-20f));
    dot = __fmul_rn(__fmul_rn(dot, rstd), 0.013975424859373685f); // 5120**-0.5
    const float signed_root = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot);
    const float g = live ? 1.f / (1.f + expf(-signed_root)) : 0.f;
    if (!threadIdx.x) gate[stream] = g;
    for (int d = int(threadIdx.x); d < dim; d += int(blockDim.x)) {
        const float h = rounded(residual[base+d]);
        output[base+d] = rounded(__fadd_rn(h, __fmul_rn(g,kv[copies*dim+d])));
    }
}
} // namespace
CudaEngram::CudaEngram(const EngramWeights& w, int device) : layer_(w.layer), device_(device) {
    if (layer_ != 1 && layer_ != 14) throw std::invalid_argument("Engram projection requires original layer 1 or 14");
    if (w.wkv.rows != projected || w.wkv.cols != input || w.wkv.values.size() != size_t(projected)*input ||
        w.wkv.scales.size() != size_t(projected/32)*(input/32) || !w.wkv.values.data() || !w.wkv.scales.data() ||
        w.q.size() != copies*dim || w.k.size() != copies*dim || !valid(w.q.data(), w.q.size_bytes(), 2) || !valid(w.k.data(), w.k.size_bytes(), 2))
        throw std::invalid_argument("Engram needs fixed 1017 projection and per-stream BF16 q/k weights");
    std::vector<float> product(copies*dim);
    for (size_t i = 0; i < product.size(); ++i) {
        product[i] = kernels::f32_from_bf16(w.q[i]) * kernels::f32_from_bf16(w.k[i]);
        if (!std::isfinite(product[i])) throw std::invalid_argument("Nonfinite Engram q/k product");
    }
    arena_ = std::make_unique<core::DeviceArena>(owned_bytes, device);
    weights_ = static_cast<uint16_t*>(arena_->alloc(matrix_bytes));
    // Reuse the validated bounded converter without increasing its shape limits.
    for (int first = 0; first < projected; first += 4096) {
        const int rows = std::min(4096, projected-first);
        const auto packed = fp8_block32_to_bf16({w.wkv.values.subspan(size_t(first)*input, size_t(rows)*input),
            w.wkv.scales.subspan(size_t(first/32)*(input/32), size_t(rows/32)*(input/32)), rows, input});
        ck(cudaMemcpy(weights_+size_t(first)*input, packed.data(), packed.size()*2, cudaMemcpyHostToDevice), "upload Engram projection chunk");
        ck(cudaStreamSynchronize(nullptr), "wait for Engram projection chunk");
    }
    product_ = static_cast<float*>(arena_->alloc(product.size()*4));
    activation_ = static_cast<uint16_t*>(arena_->alloc(input*2));
    kv_ = static_cast<float*>(arena_->alloc(projected*4));
    gate_ = static_cast<float*>(arena_->alloc(copies*4));
    ck(cudaMemcpy(product_, product.data(), product.size()*4, cudaMemcpyHostToDevice), "upload Engram q/k product");
    ck(cudaStreamSynchronize(nullptr), "wait for Engram q/k upload");
}
CudaEngram::~CudaEngram() = default;
EngramProjectionView CudaEngram::forward(const float* lookup, const float* residual, float* output, void* stream, bool live) {
    if (failed_) throw std::logic_error("Engram must be reconstructed after execution failure");
    if (!stream || (stream_ && stream != stream_)) throw std::invalid_argument("Engram needs one explicit stream");
    int device = -1; ck(cudaGetDevice(&device), "get Engram device");
    if (device != device_) throw std::invalid_argument("Engram called on wrong device");
    const void* buffers[] = {lookup,residual,output}; const size_t bytes[] = {input*4,copies*dim*4,copies*dim*4};
    for (int i = 0; i < 3; ++i) {
        if (!valid(buffers[i],bytes[i]) || overlap(buffers[i],bytes[i],arena_->base(),arena_->capacity()))
            throw std::invalid_argument("Engram needs aligned external device buffers");
        for (int j = 0; j < i; ++j)
            if (overlap(buffers[i],bytes[i],buffers[j],bytes[j])) throw std::invalid_argument("Engram caller buffers overlap");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        stream_ = stream;
        quantize_fp8_bf16(lookup, activation_, input, stream);
        kernels::bf16_gemv(activation_, weights_, kv_, input, projected, stream);
        detail::round_bf16_values(kv_, kv_, projected, stream);
        gated_residual<<<copies,256,0,s>>>(residual,kv_,product_,gate_,output,live);
        ck(cudaGetLastError(), "Engram gate/residual launch");
        ck(cudaStreamSynchronize(s), "wait for Engram projection/gate");
        return {kv_,gate_};
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
} // namespace strata::deepseek41
