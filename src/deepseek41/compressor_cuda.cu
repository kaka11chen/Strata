#include "strata/deepseek41/compressor_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120, dim = 512, limit = 1048576;
constexpr size_t matrix_bytes = size_t(hidden) * dim * 4;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
void blas_check(cublasStatus_t code, const char* what) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(code)));
}
bool valid(const void* p, size_t n) {
    return p && uintptr_t(p) % 4 == 0 && n <= UINTPTR_MAX - uintptr_t(p);
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) < uintptr_t(b) + bn && uintptr_t(b) < uintptr_t(a) + an;
}
// Each channel has its own two-token softmax. Preserve the official F32
// multiply/sum and BF16 cast BEFORE RMSNorm; never round either projection.
__global__ void pool_pair(const float* kv, const float* scores, float* pooled) {
    const int d = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= dim) return;
    const float maximum = fmaxf(scores[d], scores[dim + d]);
    const float a = expf(scores[d] - maximum), b = expf(scores[dim + d] - maximum);
    const float sum = a + b;
    const float value = __fadd_rn(__fmul_rn(kv[d], a / sum), __fmul_rn(kv[dim + d], b / sum));
    pooled[d] = kernels::f32_from_bf16(kernels::bf16_from_f32(value));
}
} // namespace

struct CudaCompressor::Impl {
    int device, next = 0;
    bool failed = false;
    void* stream = nullptr;
    std::unique_ptr<core::DeviceArena> arena;
    cublasHandle_t blas = nullptr;
    float *wkv, *wgate, *norm, *input, *kv, *scores, *pooled;
    Impl(const CompressorWeights& w, int device_id) : device(device_id) {
        if (w.kv.size() != size_t(hidden) * dim || w.gate.size() != size_t(hidden) * dim ||
            w.norm.size() != dim || !w.kv.data() || !w.gate.data() || !w.norm.data())
            throw std::invalid_argument("DeepSeek compressor needs two [512,5120] BF16 projections and [512] norm");
        // Every allocation size is a multiple of the arena's 256-byte alignment.
        arena = std::make_unique<core::DeviceArena>(2 * matrix_bytes + (hidden + 6 * dim) * 4, device);
        auto upload = [&](std::span<const uint16_t> source) {
            std::vector<float> decoded; decoded.reserve(source.size());
            for (uint16_t bits : source) {
                const float value = kernels::f32_from_bf16(bits);
                if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite DeepSeek compressor weight");
                decoded.push_back(value);
            }
            auto* dst = static_cast<float*>(arena->alloc(decoded.size() * 4));
            ck(cudaMemcpy(dst, decoded.data(), decoded.size() * 4, cudaMemcpyHostToDevice), "upload compressor weights");
            ck(cudaStreamSynchronize(nullptr), "wait for compressor upload");
            return dst;
        };
        wkv = upload(w.kv); wgate = upload(w.gate); norm = upload(w.norm);
        input = static_cast<float*>(arena->alloc(hidden * 4));
        kv = static_cast<float*>(arena->alloc(2 * dim * 4));
        scores = static_cast<float*>(arena->alloc(2 * dim * 4));
        pooled = static_cast<float*>(arena->alloc(dim * 4));
        ck(cudaMemset(kv, 0, 2 * dim * 4), "initialize compressor KV state");
        ck(cudaMemset(scores, 0, 2 * dim * 4), "initialize compressor score state");
        ck(cudaStreamSynchronize(nullptr), "wait for compressor initialization");
        blas_check(cublasCreate(&blas), "create compressor cuBLAS handle");
        try {
            blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH), "set compressor F32 math");
            blas_check(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED), "set compressor deterministic reduction");
        } catch (...) { cublasDestroy(blas); throw; }
    }
    ~Impl() { if (blas) cublasDestroy(blas); }
    void check_stream(void* s) const {
        if (!s || (stream && stream != s)) throw std::invalid_argument("DeepSeek compressor needs one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get compressor device");
        if (current != device) throw std::invalid_argument("DeepSeek compressor called on wrong device");
        if (failed) throw std::logic_error("DeepSeek compressor must be reconstructed after an execution failure");
    }
    void check_external(const void* p, size_t bytes) const {
        if (!valid(p, bytes) || overlaps(p, bytes, arena->base(), arena->capacity()))
            throw std::invalid_argument("DeepSeek compressor needs aligned external device buffers");
    }
    void bind(cudaStream_t s) {
        if (!stream) { blas_check(cublasSetStream(blas, s), "set compressor stream"); stream = s; }
    }
    void project(const float* src, int slot, cudaStream_t s) {
        detail::round_bf16_values(src, input, hidden, s);
        const float one = 1.f, zero = 0.f;
        blas_check(cublasSgemv(blas, CUBLAS_OP_T, hidden, dim, &one, wkv, hidden,
                               input, 1, &zero, kv + slot * dim, 1), "project compressor KV");
        blas_check(cublasSgemv(blas, CUBLAS_OP_T, hidden, dim, &one, wgate, hidden,
                               input, 1, &zero, scores + slot * dim, 1), "project compressor scores");
    }
};

CudaCompressor::CudaCompressor(const CompressorWeights& weights, int device)
    : impl_(std::make_unique<Impl>(weights, device)) {}
CudaCompressor::~CudaCompressor() = default;
CompressedBatch CudaCompressor::forward(const float* input, float* output, int tokens, int position, void* stream) {
    auto& p = *impl_;
    p.check_stream(stream);
    if (position != p.next || tokens <= 0 || tokens > limit - p.next || (p.next != 0 && tokens != 1))
        throw std::invalid_argument("DeepSeek compressor needs initial prefill then contiguous single-token decode within 1048576 positions");
    const int groups = (position + tokens) / 2 - position / 2;
    const size_t in_bytes = size_t(tokens) * hidden * 4, out_bytes = size_t(groups) * dim * 4;
    p.check_external(input, in_bytes);
    if (groups) {
        p.check_external(output, out_bytes);
        if (overlaps(input, in_bytes, output, out_bytes)) throw std::invalid_argument("DeepSeek compressor buffers overlap");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.bind(s);
        int emitted = 0;
        for (int t = 0; t < tokens; ++t) {
            const int slot = (position + t) % 2;
            p.project(input + size_t(t) * hidden, slot, s);
            if (slot == 0) continue;
            pool_pair<<<2, 256, 0, s>>>(p.kv, p.scores, p.pooled);
            ck(cudaGetLastError(), "compressor gated pooling");
            auto* dst = output + size_t(emitted++) * dim;
            kernels::native_qsa_rms_norm_weighted(p.pooled, p.norm, dst, dim, 1, 1e-20f, stream);
            detail::round_bf16_values(dst, dst, dim, stream);
        }
        ck(cudaStreamSynchronize(s), "wait for compressor");
        p.next += tokens;
        return {groups, groups ? (position / 2) * 2 : -1};
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaCompressor::reset(void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.bind(s);
        ck(cudaMemsetAsync(p.kv, 0, 2 * dim * 4, s), "reset compressor KV state");
        ck(cudaMemsetAsync(p.scores, 0, 2 * dim * 4, s), "reset compressor score state");
        ck(cudaStreamSynchronize(s), "wait for compressor reset");
        p.next = 0;
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
int CudaCompressor::next_position() const { return impl_->next; }
size_t CudaCompressor::device_bytes() const { return impl_->arena->capacity(); }
const float* CudaCompressor::pending_kv() const { return impl_->next % 2 ? impl_->kv : nullptr; }
const float* CudaCompressor::pending_scores() const { return impl_->next % 2 ? impl_->scores : nullptr; }
} // namespace strata::deepseek41
