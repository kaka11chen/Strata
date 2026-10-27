#include "strata/deepseek41/mhc_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr float norm_eps = 1e-20f, hc_eps = 1e-6f;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
void blas_check(cublasStatus_t code, const char* what) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(code)));
}
size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
bool valid(const void* p, size_t n) {
    const auto a = uintptr_t(p);
    return p && a % 4 == 0 && n <= UINTPTR_MAX - a;
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) < uintptr_t(b) + bn && uintptr_t(b) < uintptr_t(a) + an;
}
__device__ float bf16(float v) { return kernels::f32_from_bf16(kernels::bf16_from_f32(v)); }
__device__ float warp_sum(float x) {
    for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
// The official graph normalizes AFTER the F32 projection, over all 4*hidden
// residual values. Pre-normalizing the projection input changes its rounding.
__global__ void normalize_mixes(const float* residual, float* mixes, int n) {
    float sum = 0.f;
    for (int i = int(threadIdx.x); i < n; i += int(blockDim.x))
        sum += __fmul_rn(residual[i], residual[i]);
    __shared__ float partial[8];
    sum = warp_sum(sum);
    const int lane = int(threadIdx.x) % 32;
    if (lane == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = warp_sum(lane < 8 ? partial[lane] : 0.f);
    if (threadIdx.x < 24) mixes[threadIdx.x] *= rsqrtf(sum / float(n) + norm_eps);
}
// Only 24 coefficients. A serial baseline makes the fixed update order explicit;
// all large projections and normalization use existing primitives.
__global__ void split_sinkhorn(const float* mixes, const float* scale, const float* base, float* out) {
    float comb[16];
    for (int i = 0; i < 4; ++i) {
        out[i] = 1.f / (1.f + expf(-__fadd_rn(__fmul_rn(mixes[i], scale[0]), base[i]))) + hc_eps;
        out[4 + i] = 2.f / (1.f + expf(-__fadd_rn(__fmul_rn(mixes[4 + i], scale[1]), base[4 + i])));
    }
    for (int row = 0; row < 4; ++row) {
        float maximum = -INFINITY, sum = 0.f;
        for (int col = 0; col < 4; ++col) {
            const int i = row * 4 + col;
            comb[i] = __fadd_rn(__fmul_rn(mixes[8 + i], scale[2]), base[8 + i]);
            maximum = fmaxf(maximum, comb[i]);
        }
        for (int col = 0; col < 4; ++col) { comb[row * 4 + col] = expf(comb[row * 4 + col] - maximum); sum += comb[row * 4 + col]; }
        for (int col = 0; col < 4; ++col) comb[row * 4 + col] = comb[row * 4 + col] / sum + hc_eps;
    }
    for (int iteration = 0; iteration < 20; ++iteration) {
        // Iteration zero follows the initial row softmax+eps with a column norm.
        if (iteration != 0)
            for (int row = 0; row < 4; ++row) {
                float sum = 0.f;
                for (int col = 0; col < 4; ++col) sum += comb[row * 4 + col];
                for (int col = 0; col < 4; ++col) comb[row * 4 + col] /= sum + hc_eps;
            }
        for (int col = 0; col < 4; ++col) {
            float sum = 0.f;
            for (int row = 0; row < 4; ++row) sum += comb[row * 4 + col];
            for (int row = 0; row < 4; ++row) comb[row * 4 + col] /= sum + hc_eps;
        }
    }
    for (int i = 0; i < 16; ++i) out[8 + i] = comb[i];
}
__global__ void collapse(const float* residual, const float* incoming, float* output, int n) {
    const int d = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= n) return;
    float sum = 0.f;
    for (int source = 0; source < 4; ++source) sum += __fmul_rn(incoming[source], residual[source * n + d]);
    output[d] = bf16(sum);
}
__global__ void expand(const float* sublayer, const float* residual, const float* coeff, float* output, int n) {
    const int d = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= n) return;
    for (int dest = 0; dest < 4; ++dest) {
        float mixed = 0.f;
        for (int source = 0; source < 4; ++source)
            mixed += __fmul_rn(coeff[8 + source * 4 + dest], residual[source * n + d]);
        output[dest * n + d] = bf16(__fadd_rn(__fmul_rn(coeff[4 + dest], bf16(sublayer[d])), mixed));
    }
}
} // namespace

void detail::collapse_mhc_values(const float* residual, const float* incoming, float* output, int hidden, void* stream) {
    collapse<<<(hidden + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(residual, incoming, output, hidden);
    ck(cudaGetLastError(), "DeepSeek mHC collapse");
}

struct CudaMhc::Blas {
    cublasHandle_t handle = nullptr;
    Blas() {
        blas_check(cublasCreate(&handle), "create DeepSeek mHC cuBLAS handle");
        try {
            blas_check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "set mHC FP32 math");
            blas_check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "set mHC deterministic reduction");
        } catch (...) { cublasDestroy(handle); throw; }
    }
    ~Blas() { cublasDestroy(handle); }
};
CudaMhc::~CudaMhc() = default;

CudaMhc::CudaMhc(const Geometry& g, const MhcWeights& w, int device) : hidden_(g.hidden), device_(device) {
    if (hidden_ <= 0 || hidden_ > 5120 || hidden_ % 32 || w.projection.size() != size_t(96) * hidden_ ||
        w.scale.size() != 3 || w.base.size() != 24 || w.norm.size() != size_t(hidden_) ||
        !w.projection.data() || !w.scale.data() || !w.base.data() || !w.norm.data())
        throw std::invalid_argument("DeepSeek mHC needs four streams, 24 F32 projection rows and matching norm weights");
    for (auto values : {w.projection, w.scale, w.base})
        for (float value : values)
            if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite DeepSeek mHC weight");
    std::vector<float> norm(size_t(hidden_), 0.f);
    for (int i = 0; i < hidden_; ++i) {
        norm[size_t(i)] = kernels::f32_from_bf16(w.norm[size_t(i)]);
        if (!std::isfinite(norm[size_t(i)])) throw std::invalid_argument("Nonfinite DeepSeek mHC norm weight");
    }
    const size_t n = size_t(hidden_) * 4;
    size_t capacity = 0;
    for (size_t bytes : {w.projection.size_bytes(), size_t(12), size_t(96), n, 4 * n, size_t(96), size_t(96), n, n})
        capacity = aligned(capacity) + bytes;
    arena_ = std::make_unique<core::DeviceArena>(capacity, device);
    auto upload = [&](const void* source, size_t bytes) {
        auto* dst = static_cast<float*>(arena_->alloc(bytes));
        ck(cudaMemcpy(dst, source, bytes, cudaMemcpyHostToDevice), "upload DeepSeek mHC weights");
        return dst;
    };
    projection_ = upload(w.projection.data(), w.projection.size_bytes());
    scale_ = upload(w.scale.data(), 12); base_ = upload(w.base.data(), 96); norm_ = upload(norm.data(), n);
    residual_ = static_cast<float*>(arena_->alloc(4 * n));
    mixes_ = static_cast<float*>(arena_->alloc(96)); coefficients_ = static_cast<float*>(arena_->alloc(96));
    normalized_ = static_cast<float*>(arena_->alloc(n)); moe_output_ = static_cast<float*>(arena_->alloc(n));
    ck(cudaStreamSynchronize(nullptr), "wait for DeepSeek mHC upload");
    blas_ = std::make_unique<Blas>();
}
void CudaMhc::check_stream(void* stream) const {
    if (!stream || (stream_ && stream_ != stream)) throw std::invalid_argument("DeepSeek mHC requires one explicit stream per instance");
    int device = -1; ck(cudaGetDevice(&device), "get DeepSeek mHC device");
    if (device != device_) throw std::invalid_argument("DeepSeek mHC called on wrong device");
    if (failed_) throw std::logic_error("DeepSeek mHC must be reconstructed after an execution failure");
}
void CudaMhc::check_external(const void* pointer, size_t bytes) const {
    if (!valid(pointer, bytes) || overlaps(pointer, bytes, arena_->base(), arena_->capacity()))
        throw std::invalid_argument("DeepSeek mHC needs an aligned external device buffer");
}
void CudaMhc::collapse_normalize(const float* incoming, void* stream) {
    detail::collapse_mhc_values(residual_, incoming, normalized_, hidden_, stream);
    kernels::native_qsa_rms_norm_weighted(normalized_, norm_, normalized_, hidden_, 1, norm_eps, stream);
    detail::round_bf16_values(normalized_, normalized_, hidden_, stream);
}
const float* CudaMhc::normalize_input(const float* residual, const float* incoming, void* stream) {
    check_stream(stream);
    if (pending_) throw std::logic_error("DeepSeek mHC normalize cannot interrupt prepare/finish");
    check_external(residual, size_t(hidden_) * 16); check_external(incoming, 16);
    if (overlaps(residual, size_t(hidden_) * 16, incoming, 16)) throw std::invalid_argument("DeepSeek mHC inputs overlap");
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        if (!stream_) { blas_check(cublasSetStream(blas_->handle, s), "set mHC stream"); stream_ = stream; }
        detail::round_bf16_values(residual, residual_, 4 * hidden_, stream);
        collapse_normalize(incoming, stream);
        return normalized_;
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
MhcPrepared CudaMhc::prepare(const float* residual, const float* incoming, void* stream) {
    check_stream(stream);
    if (pending_) throw std::logic_error("DeepSeek mHC prepare must be paired with finish");
    check_external(residual, size_t(hidden_) * 16); check_external(incoming, 16);
    if (overlaps(residual, size_t(hidden_) * 16, incoming, 16)) throw std::invalid_argument("DeepSeek mHC inputs overlap");
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        if (!stream_) { blas_check(cublasSetStream(blas_->handle, s), "set mHC stream"); stream_ = stream; }
        detail::round_bf16_values(residual, residual_, 4 * hidden_, stream);
        const float one = 1.f, zero = 0.f;
        blas_check(cublasSgemv(blas_->handle, CUBLAS_OP_T, 4 * hidden_, 24, &one, projection_, 4 * hidden_,
                               residual_, 1, &zero, mixes_, 1), "project DeepSeek mHC");
        normalize_mixes<<<1, 256, 0, s>>>(residual_, mixes_, 4 * hidden_);
        split_sinkhorn<<<1, 1, 0, s>>>(mixes_, scale_, base_, coefficients_);
        collapse_normalize(incoming, stream);
        pending_ = true;
        return {normalized_, coefficients_, coefficients_ + 4, coefficients_ + 8};
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaMhc::finish(const float* sublayer, float* output, void* stream) {
    check_stream(stream);
    if (!pending_) throw std::logic_error("DeepSeek mHC finish needs a preceding prepare");
    check_external(output, size_t(hidden_) * 16);
    if (!valid(sublayer, size_t(hidden_) * 4) || overlaps(sublayer, size_t(hidden_) * 4, output, size_t(hidden_) * 16))
        throw std::invalid_argument("DeepSeek mHC output overlaps its sublayer or sublayer is invalid");
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        expand<<<(hidden_ + 255) / 256, 256, 0, s>>>(sublayer, residual_, coefficients_, output, hidden_);
        ck(cudaGetLastError(), "DeepSeek mHC residual expansion");
        pending_ = false;
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
Route CudaMhc::forward_moe(CudaMoE& moe, const float* residual, const float* incoming,
                          float* output, float* next_pre, void* stream) {
    check_stream(stream);
    if (moe.hidden_size() != hidden_) throw std::invalid_argument("DeepSeek mHC and MoE widths differ");
    const void* buffers[] = {residual, incoming, output, next_pre};
    const size_t bytes[] = {size_t(hidden_) * 16, 16, size_t(hidden_) * 16, 16};
    for (int i = 0; i < 4; ++i) {
        check_external(buffers[i], bytes[i]);
        for (int j = 0; j < i; ++j)
            if (overlaps(buffers[i], bytes[i], buffers[j], bytes[j])) throw std::invalid_argument("DeepSeek mHC/MoE buffers overlap");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        const auto state = prepare(residual, incoming, stream);
        const Route route = moe.forward(state.normalized, moe_output_, stream);
        finish(moe_output_, output, stream);
        ck(cudaMemcpyAsync(next_pre, state.next_pre, 16, cudaMemcpyDeviceToDevice, s), "copy delayed DeepSeek pre coefficients");
        ck(cudaStreamSynchronize(s), "wait for DeepSeek mHC/MoE");
        return route;
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
} // namespace strata::deepseek41
