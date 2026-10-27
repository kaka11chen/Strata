#include "strata/deepseek41/attention_cuda.hpp"
#include "strata/deepseek41/source20_attention_cuda.hpp"
#include "strata/deepseek41/layer2_attention_cuda.hpp"
#include "strata/deepseek41/layer3_attention_cuda.hpp"
#include "strata/deepseek41/compressed_attention_cuda.hpp"
#include "compressed_rope.hpp"
#include "strata/deepseek41/shared_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int heads = 64, dim = 512, window = 128, qsize = heads * dim, max_position = 1048576;
constexpr size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
void bc(cublasStatus_t code, const char* what) {
    if (code != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(int(code)));
}
struct Blas {
    cublasHandle_t handle = nullptr;
    Blas() {
        bc(cublasCreate(&handle), "create attention cuBLAS handle");
        try {
            bc(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH), "set attention FP32 math");
            bc(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "set attention deterministic reduction");
        } catch (...) { cublasDestroy(handle); throw; }
    }
    ~Blas() { cublasDestroy(handle); }
};
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto x = uintptr_t(a), y = uintptr_t(b);
    return x <= y ? y - x < an : x - y < bn;
}
void buffers(const float* a, size_t an, const float* b, size_t bn) {
    if (!a || !b || uintptr_t(a) % 4 || uintptr_t(b) % 4 || overlap(a, an * 4, b, bn * 4))
        throw std::invalid_argument("Attention requires aligned disjoint device buffers");
}
__device__ float rounded(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
// Exactly the official slot ordering, including invalid prefix slots during decode.
// For prefill we map absolute causal positions into the ring as each query runs.
__device__ int slot(int candidate, int pos, int prefill_width) {
    if (prefill_width) {
        const int absolute = max(0, pos - window + 1) + candidate;
        return candidate < prefill_width && absolute <= pos ? absolute % window : -1;
    }
    const int i = (pos % window + 1 + candidate) % window;
    return i <= pos ? i : -1;
}
__global__ void rotate(float* x, const float* frequencies, int position, int nheads, bool inverse) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= nheads * 32) return;
    const int pair = i % 32, at = (i / 32) * dim + dim - 64 + pair * 2;
    const float angle = float(position) * frequencies[pair];
    float s, c; sincosf(angle, &s, &c);
    if (inverse) s = -s;
    const float a = x[at], b = x[at + 1];
    // Separate FP32 products/adds match complex multiplication before BF16 copy.
    x[at] = rounded(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
    x[at + 1] = rounded(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)));
}
__global__ void gather(const uint16_t* ring, float* kv, int position, int prefill_width) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= window * dim) return;
    const int source = slot(i / dim, position, prefill_width);
    kv[i] = source < 0 ? 0.f : kernels::f32_from_bf16(ring[source * dim + i % dim]);
}
__global__ void probabilities(const float* scores, float* prob, float* factors, const float* sink,
                              int position, int prefill_width) {
    const int h = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (h >= heads) return;
    float maximum = -1e30f, denominator = 0.f, correction = 1.f;
    for (int chunk = 0; chunk < 2; ++chunk) {
        const float previous = maximum;
        for (int j = chunk * 64; j < (chunk + 1) * 64; ++j)
            if (slot(j, position, prefill_width) >= 0)
                maximum = fmaxf(maximum, scores[h * window + j] * 0.04419417382415922f);
        correction = expf(previous - maximum);
        float sum = 0.f;
        for (int j = chunk * 64; j < (chunk + 1) * 64; ++j) {
            const float p = slot(j, position, prefill_width) < 0 ? 0.f :
                            expf(scores[h * window + j] * 0.04419417382415922f - maximum);
            sum += p;
            prob[h * window + j] = rounded(p); // numerator only; denominator stays FP32
        }
        denominator = __fadd_rn(__fmul_rn(denominator, correction), sum);
    }
    factors[h * 2] = correction; // rescale chunk 0 into chunk 1's maximum
    factors[h * 2 + 1] = denominator + expf(sink[h] - maximum);
}
__global__ void combine(const float* parts, const float* factors, float* output) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= qsize) return;
    const int h = i / dim;
    output[i] = rounded(__fadd_rn(__fmul_rn(parts[i], factors[h * 2]), parts[qsize + i]) / factors[h * 2 + 1]);
}
constexpr size_t swa_bytes = aligned(64 * 4) + aligned(32 * 4) + window * dim * 2 + qsize * 4 + dim * 4 +
                              window * dim * 4 + 2 * heads * window * 4 + 2 * qsize * 4 + heads * 2 * 4;
constexpr std::array<int, 5> rows{1280, 32768, 512, 8192, 5120}, cols{5120, 1280, 5120, 4096, 8192};
constexpr size_t attention_weight_bytes = 2ull * (1280 * 5120 + 32768 * 1280 + 512 * 5120 + 8192 * 4096 + 5120 * 8192);
constexpr size_t projection_bytes = attention_weight_bytes + (1280 + 512) * 4 + qsize * 2 + (1280 + qsize + 512 + qsize + 8192) * 4;
void validate_weights(const AttentionWeights& w) {
    if (w.q_norm.size() != 1280 || w.kv_norm.size() != 512 || !w.q_norm.data() || !w.kv_norm.data())
        throw std::invalid_argument("Invalid 1017 attention norm shape");
    for (auto norm : {w.q_norm, w.kv_norm})
        for (uint16_t v : norm)
            if (!std::isfinite(kernels::f32_from_bf16(v))) throw std::invalid_argument("Nonfinite attention norm");
    for (size_t i = 0; i < 5; ++i) {
        const auto& m = w.projections[i];
        if (m.rows != rows[i] || m.cols != cols[i] || !m.values.data() || !m.scales.data() ||
            m.values.size() != size_t(rows[i]) * cols[i] || m.scales.size() != size_t(rows[i] / 32) * (cols[i] / 32))
            throw std::invalid_argument("Invalid 1017 attention projection shape");
    }
}
} // namespace

struct CudaWindowAttention::Impl {
    core::DeviceArena arena;
    Blas blas;
    int device, next = 0;
    bool poisoned = false;
    void* stream = nullptr;
    float *sink, *frequencies, *q, *kv, *gathered, *scores, *prob, *parts, *factors;
    uint16_t* ring;
    Impl(std::span<const float> host_sink, int dev) : arena(swa_bytes, dev), device(dev) {
        sink = static_cast<float*>(arena.alloc(64 * 4));
        frequencies = static_cast<float*>(arena.alloc(32 * 4));
        ring = static_cast<uint16_t*>(arena.alloc(window * dim * 2));
        q = static_cast<float*>(arena.alloc(qsize * 4));
        kv = static_cast<float*>(arena.alloc(dim * 4));
        gathered = static_cast<float*>(arena.alloc(window * dim * 4));
        scores = static_cast<float*>(arena.alloc(heads * window * 4));
        prob = static_cast<float*>(arena.alloc(heads * window * 4));
        parts = static_cast<float*>(arena.alloc(2 * qsize * 4));
        factors = static_cast<float*>(arena.alloc(heads * 2 * 4));
        std::array<float, 32> freq{};
        for (size_t i = 0; i < freq.size(); ++i) freq[i] = 1.f / std::pow(10000.f, float(i) / 32.f);
        ck(cudaMemcpy(sink, host_sink.data(), 64 * 4, cudaMemcpyHostToDevice), "upload attention sink");
        ck(cudaMemcpy(frequencies, freq.data(), 32 * 4, cudaMemcpyHostToDevice), "upload attention frequencies");
        ck(cudaMemset(ring, 0, window * dim * 2), "clear attention ring");
        ck(cudaStreamSynchronize(nullptr), "wait attention startup upload");
    }
    void bind(void* s) {
        if (poisoned) throw std::runtime_error("Attention state poisoned after an execution failure");
        if (!s || (stream && stream != s)) throw std::invalid_argument("Attention requires one explicit stream per instance");
        int active = -1; ck(cudaGetDevice(&active), "get attention device");
        if (active != device) throw std::invalid_argument("Attention called on wrong device");
        if (!stream) { bc(cublasSetStream(blas.handle, static_cast<cudaStream_t>(s)), "set attention stream"); stream = s; }
    }
    void validate(int tokens, int position, void* s) {
        if (tokens <= 0 || tokens > max_position || position < 0 || position != next ||
            position > max_position - tokens || (position != 0 && tokens != 1))
            throw std::invalid_argument("Attention requires initial prefill, then contiguous single-token decode within 1048576 positions");
        bind(s);
    }
    void step(const float* query, const float* key, float* output, int position, int prefill_width) {
        const auto s = static_cast<cudaStream_t>(stream);
        detail::round_bf16_values(query, q, qsize, stream);
        detail::round_bf16_values(key, kv, dim, stream);
        rotate<<<8, 256, 0, s>>>(q, frequencies, position, heads, false);
        rotate<<<1, 32, 0, s>>>(kv, frequencies, position, 1, false);
        quantize_fp8_bf16(kv, ring + (position % window) * dim, dim, stream);
        gather<<<256, 256, 0, s>>>(ring, gathered, position, prefill_width);
        const float one = 1.f, zero = 0.f;
        // Row-major Q @ KV^T through column-major cuBLAS, no TF32 downcast.
        bc(cublasSgemm(blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, window, heads, dim, &one,
                      gathered, dim, q, dim, &zero, scores, window), "attention QK dot");
        probabilities<<<2, 32, 0, s>>>(scores, prob, factors, sink, position, prefill_width);
        for (int chunk = 0; chunk < 2; ++chunk)
            bc(cublasSgemm(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, heads, 64, &one,
                          gathered + chunk * 64 * dim, dim, prob + chunk * 64, window,
                          &zero, parts + chunk * qsize, dim), "attention probability KV dot");
        combine<<<128, 256, 0, s>>>(parts, factors, output);
        rotate<<<8, 256, 0, s>>>(output, frequencies, position, heads, true);
        ck(cudaGetLastError(), "DeepSeek SWA operations");
    }
};

CudaWindowAttention::CudaWindowAttention(std::span<const float> sink, int device) {
    if (sink.size() != 64 || !sink.data()) throw std::invalid_argument("Attention needs 64 sink values");
    for (float v : sink) if (!std::isfinite(v)) throw std::invalid_argument("Nonfinite attention sink");
    impl_ = std::make_unique<Impl>(sink, device);
}
CudaWindowAttention::~CudaWindowAttention() = default;
void CudaWindowAttention::forward(const float* query, const float* kv, float* output, int tokens, int start_pos, void* stream) {
    if (tokens <= 0 || tokens > max_position) throw std::invalid_argument("Invalid attention token count");
    const size_t qn = size_t(tokens) * qsize, kn = size_t(tokens) * dim;
    buffers(query, qn, output, qn); buffers(kv, kn, output, qn); buffers(query, qn, kv, kn);
    impl_->validate(tokens, start_pos, stream);
    try {
        const int width = start_pos == 0 ? std::min(tokens, window) : 0;
        for (int t = 0; t < tokens; ++t)
            impl_->step(query + size_t(t) * qsize, kv + size_t(t) * dim, output + size_t(t) * qsize, start_pos + t, width);
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait attention execution");
        impl_->next += tokens;
    } catch (...) { impl_->poisoned = true; throw; }
}
void CudaWindowAttention::reset(void* stream) {
    impl_->bind(stream);
    try {
        ck(cudaMemsetAsync(impl_->ring, 0, window * dim * 2, static_cast<cudaStream_t>(stream)), "reset attention ring");
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait attention reset");
        impl_->next = 0;
    } catch (...) { impl_->poisoned = true; throw; }
}
int CudaWindowAttention::next_position() const { return impl_->next; }
size_t CudaWindowAttention::device_bytes() const { return impl_->arena.capacity(); }
const uint16_t* CudaWindowAttention::kv_cache() const { return impl_->ring; }

namespace {
// One projection implementation for the fixed layer-0, layer-2 and layer-3 paths.
struct AttentionProjections {
    core::DeviceArena arena;
    std::array<const uint16_t*, 5> weights{};
    float *q_norm, *kv_norm, *qr, *q, *kv, *attn, *lowrank;
    uint16_t* activation;
    AttentionProjections(const AttentionWeights& w, int device) : arena(projection_bytes, device) {
        for (size_t i = 0; i < weights.size(); ++i) {
            const auto expanded = fp8_block32_to_bf16(w.projections[i]);
            auto* dst = static_cast<uint16_t*>(arena.alloc(expanded.size() * 2));
            ck(cudaMemcpy(dst, expanded.data(), expanded.size() * 2, cudaMemcpyHostToDevice), "upload attention weight");
            ck(cudaStreamSynchronize(nullptr), "wait attention weight upload");
            weights[i] = dst;
        }
        auto upload_norm = [&](std::span<const uint16_t> norm) {
            std::vector<float> host; host.reserve(norm.size());
            for (uint16_t v : norm) host.push_back(kernels::f32_from_bf16(v));
            auto* dst = static_cast<float*>(arena.alloc(host.size() * 4));
            ck(cudaMemcpy(dst, host.data(), host.size() * 4, cudaMemcpyHostToDevice), "upload attention norm");
            ck(cudaStreamSynchronize(nullptr), "wait attention norm upload");
            return dst;
        };
        q_norm = upload_norm(w.q_norm); kv_norm = upload_norm(w.kv_norm);
        activation = static_cast<uint16_t*>(arena.alloc(qsize * 2));
        qr = static_cast<float*>(arena.alloc(1280 * 4));
        q = static_cast<float*>(arena.alloc(qsize * 4));
        kv = static_cast<float*>(arena.alloc(dim * 4));
        attn = static_cast<float*>(arena.alloc(qsize * 4));
        lowrank = static_cast<float*>(arena.alloc(8192 * 4));
    }
    void project_qkv(const float* input, void* stream) {
        quantize_fp8_bf16(input, activation, 5120, stream);
        kernels::bf16_gemv(activation, weights[0], qr, 5120, 1280, stream);
        kernels::bf16_gemv(activation, weights[2], kv, 5120, dim, stream);
        detail::round_bf16_values(qr, qr, 1280, stream);
        detail::round_bf16_values(kv, kv, dim, stream);
        kernels::native_qsa_rms_norm_weighted(qr, q_norm, qr, 1280, 1, 1e-20f, stream);
        kernels::native_qsa_rms_norm_weighted(kv, kv_norm, kv, dim, 1, 1e-20f, stream);
        detail::round_bf16_values(qr, qr, 1280, stream);
        detail::round_bf16_values(kv, kv, dim, stream);
        quantize_fp8_bf16(qr, activation, 1280, stream);
        kernels::bf16_gemv(activation, weights[1], q, 1280, qsize, stream);
        // Window core supplies the BF16 boundary before query RoPE.
    }
    void project_output(float* output, void* stream) {
        quantize_fp8_bf16(attn, activation, qsize, stream);
        for (int group = 0; group < 8; ++group)
            kernels::bf16_gemv(activation + group * 4096, weights[3] + size_t(group) * 1024 * 4096,
                               lowrank + group * 1024, 4096, 1024, stream);
        detail::round_bf16_values(lowrank, lowrank, 8192, stream);
        quantize_fp8_bf16(lowrank, activation, 8192, stream);
        kernels::bf16_gemv(activation, weights[4], output, 8192, 5120, stream);
        detail::round_bf16_values(output, output, 5120, stream);
    }
};
} // namespace
struct CudaAttention::Impl {
    CudaWindowAttention window_core;
    AttentionProjections projections;
    Impl(const AttentionWeights& w, int device) : window_core(w.sink, device), projections(w, device) {}
};
CudaAttention::CudaAttention(const AttentionWeights& w, int device, int layer) {
    if ((layer != 0 && layer != 1) || w.layer != layer)
        throw std::invalid_argument("CudaAttention requires matching explicit layer-0/layer-1 pure-SWA weights");
    validate_weights(w); impl_ = std::make_unique<Impl>(w, device);
}
CudaAttention::~CudaAttention() = default;
void CudaAttention::forward(const float* input, float* output, int tokens, int start_pos, void* stream) {
    if (tokens <= 0 || tokens > max_position) throw std::invalid_argument("Invalid attention token count");
    buffers(input, size_t(tokens) * 5120, output, size_t(tokens) * 5120);
    auto& state = *impl_->window_core.impl_;
    state.validate(tokens, start_pos, stream);
    try {
        const int width = start_pos == 0 ? std::min(tokens, window) : 0;
        for (int t = 0; t < tokens; ++t) {
            impl_->projections.project_qkv(input + size_t(t) * 5120, stream);
            state.step(impl_->projections.q, impl_->projections.kv, impl_->projections.attn, start_pos + t, width);
            impl_->projections.project_output(output + size_t(t) * 5120, stream);
        }
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait full attention execution");
        state.next += tokens;
    } catch (...) { state.poisoned = true; throw; }
}
void CudaAttention::forward_token(const float* input, float* output, int position, int prefill_width, void* stream) {
    buffers(input, 5120, output, 5120);
    auto& state = *impl_->window_core.impl_;
    state.validate(1, position, stream);
    try {
        impl_->projections.project_qkv(input, stream);
        state.step(impl_->projections.q, impl_->projections.kv, impl_->projections.attn, position, prefill_width);
        impl_->projections.project_output(output, stream);
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait decoder attention token");
        ++state.next;
    } catch (...) { state.poisoned = true; throw; }
}
void CudaAttention::reset(void* stream) { impl_->window_core.reset(stream); }
int CudaAttention::next_position() const { return impl_->window_core.next_position(); }
size_t CudaAttention::weight_bytes() const { return attention_weight_bytes; }
size_t CudaAttention::device_bytes() const { return impl_->projections.arena.capacity() + impl_->window_core.device_bytes(); }
const uint16_t* CudaAttention::kv_cache() const { return impl_->window_core.kv_cache(); }
} // namespace strata::deepseek41

namespace strata::deepseek41 {
namespace {
constexpr int sparse_candidates = window + 512;
constexpr size_t compressed_attention_bytes = aligned(heads * 4) + aligned(window * dim * 2) +
    aligned(qsize * 4) + aligned(dim * 4) + aligned(sparse_candidates * dim * 4) +
    aligned(heads * sparse_candidates * 4) + aligned(heads * 64 * 4) + 2 * aligned(qsize * 4) +
    3 * aligned(heads * 4) + aligned(sparse_candidates * 4) + aligned(4);
__global__ void validate_selection(const int32_t* ids, int count, int groups, int offset,
                                   int position, int* invalid) {
    const int j = int(threadIdx.x), row = int(blockIdx.x);
    if (j >= count) return;
    const int id = ids[size_t(row) * count + j];
    const int previous = j ? ids[size_t(row) * count + j - 1] : -2;
    const bool bad = id < -1 || (id != -1 && (id < offset || id - offset >= groups ||
        id - offset >= (position + row + 1) / 2 || (j && (previous < 0 || previous >= id))));
    if (bad) atomicOr(invalid, 1);
}
__global__ void gather_selected(const uint16_t* ring, const uint16_t* compressed, const int32_t* ids,
                                 float* values, int* valid, int width, int count, int offset,
                                 int stride, int position, int prefill_width) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= stride * dim) return;
    const int candidate = i / dim, d = i % dim;
    const uint16_t* source = nullptr;
    if (candidate < width) {
        const int at = slot(candidate, position, prefill_width);
        if (at >= 0) source = ring + at * dim;
    } else if (candidate < width + count) {
        const int id = ids[candidate - width];
        if (id >= 0) source = compressed + size_t(id - offset) * dim;
    }
    values[i] = source ? kernels::f32_from_bf16(source[d]) : 0.f;
    if (!d) valid[candidate] = source != nullptr;
}
__global__ void chunk_probabilities(const float* scores, const int* valid, float* prob,
                                     float* maximum, float* denominator, float* alpha, int stride, int chunk) {
    const int h = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (h >= heads) return;
    const float previous = chunk ? maximum[h] : -1e30f;
    float m = previous;
    for (int j = 0; j < 64; ++j) {
        const int at = chunk * 64 + j;
        if (valid[at]) m = fmaxf(m, scores[h * stride + at] * 0.04419417382415922f);
    }
    const float correction = expf(previous - m);
    float sum = 0.f;
    for (int j = 0; j < 64; ++j) {
        const int at = chunk * 64 + j;
        const float p = valid[at] ? expf(scores[h * stride + at] * 0.04419417382415922f - m) : 0.f;
        sum += p;
        prob[h * 64 + j] = rounded(p);
    }
    maximum[h] = m; alpha[h] = correction;
    denominator[h] = __fadd_rn(__fmul_rn(chunk ? denominator[h] : 0.f, correction), sum);
}
__global__ void accumulate_chunk(const float* part, const float* alpha, float* acc, int chunk) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < qsize) acc[i] = chunk ? __fadd_rn(__fmul_rn(acc[i], alpha[i / dim]), part[i]) : part[i];
}
__global__ void finish_selected(const float* acc, const float* maximum, const float* denominator,
                                 const float* sink, float* output) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < qsize) {
        const int h = i / dim;
        output[i] = rounded(acc[i] / (denominator[h] + expf(sink[h] - maximum[h])));
    }
}
} // namespace

struct CudaCompressedAttention::Impl {
    core::DeviceArena arena;
    Blas blas;
    int device, next = 0;
    bool poisoned = false;
    void* stream = nullptr;
    float *sink, *q, *kv, *gathered, *scores, *prob, *part, *acc, *maximum, *denominator, *alpha;
    uint16_t* ring;
    int *valid, *invalid;
    Impl(std::span<const float> host_sink, int dev) : arena(compressed_attention_bytes, dev), device(dev) {
        sink = static_cast<float*>(arena.alloc(heads * 4));
        ring = static_cast<uint16_t*>(arena.alloc(window * dim * 2));
        q = static_cast<float*>(arena.alloc(qsize * 4)); kv = static_cast<float*>(arena.alloc(dim * 4));
        gathered = static_cast<float*>(arena.alloc(sparse_candidates * dim * 4));
        scores = static_cast<float*>(arena.alloc(heads * sparse_candidates * 4));
        prob = static_cast<float*>(arena.alloc(heads * 64 * 4));
        part = static_cast<float*>(arena.alloc(qsize * 4)); acc = static_cast<float*>(arena.alloc(qsize * 4));
        maximum = static_cast<float*>(arena.alloc(heads * 4));
        denominator = static_cast<float*>(arena.alloc(heads * 4)); alpha = static_cast<float*>(arena.alloc(heads * 4));
        valid = static_cast<int*>(arena.alloc(sparse_candidates * 4)); invalid = static_cast<int*>(arena.alloc(4));
        ck(cudaMemcpy(sink, host_sink.data(), heads * 4, cudaMemcpyHostToDevice), "upload compressed attention sink");
        ck(cudaMemset(ring, 0, window * dim * 2), "clear compressed attention ring");
        ck(cudaStreamSynchronize(nullptr), "wait compressed attention startup");
    }
    void validate_stream(void* s) {
        if (poisoned) throw std::runtime_error("Compressed attention state poisoned after execution failure");
        if (!s || (stream && stream != s)) throw std::invalid_argument("Compressed attention requires one explicit stream");
        int active = -1; ck(cudaGetDevice(&active), "get compressed attention device");
        if (active != device) throw std::invalid_argument("Compressed attention called on wrong device");
    }
    void bind(void* s) {
        if (!stream) { bc(cublasSetStream(blas.handle, static_cast<cudaStream_t>(s)), "set compressed attention stream"); stream = s; }
    }
    void step(const float* query, const float* key, float* output, const CompressedAttentionSelection& selection,
              const int32_t* ids, int position, int prefill_width) {
        const auto s = static_cast<cudaStream_t>(stream);
        detail::round_bf16_values(query, q, qsize, stream);
        detail::round_bf16_values(key, kv, dim, stream);
        detail::rotate_compressed(q, heads, position, false, stream);
        detail::rotate_compressed(kv, 1, position, false, stream);
        quantize_fp8_bf16(kv, ring + (position % window) * dim, dim, stream);
        const int width = prefill_width ? prefill_width : window;
        // Pad only the END of the joined list. Padding the window on its own
        // changes BF16 probability rounding when prefill has fewer than 128 tokens.
        const int stride = (width + selection.count + 63) / 64 * 64;
        gather_selected<<<stride * dim / 256, 256, 0, s>>>(ring, selection.kv, ids, gathered, valid,
            width, selection.count, selection.offset, stride, position, prefill_width);
        const float one = 1.f, zero = 0.f;
        bc(cublasSgemm(blas.handle, CUBLAS_OP_T, CUBLAS_OP_N, stride, heads, dim, &one,
                      gathered, dim, q, dim, &zero, scores, stride), "compressed attention QK dot");
        for (int chunk = 0; chunk < stride / 64; ++chunk) {
            chunk_probabilities<<<2, 32, 0, s>>>(scores, valid, prob, maximum, denominator, alpha, stride, chunk);
            bc(cublasSgemm(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, heads, 64, &one,
                          gathered + chunk * 64 * dim, dim, prob, 64, &zero, part, dim), "compressed attention PV dot");
            accumulate_chunk<<<128, 256, 0, s>>>(part, alpha, acc, chunk);
        }
        finish_selected<<<128, 256, 0, s>>>(acc, maximum, denominator, sink, output);
        detail::rotate_compressed(output, heads, position, true, stream);
        ck(cudaGetLastError(), "DeepSeek compressed attention operations");
    }
};

CudaCompressedAttention::CudaCompressedAttention(std::span<const float> sink, int device) {
    if (!sink.data() || sink.size() != heads) throw std::invalid_argument("Compressed attention needs 64 sink values");
    for (float value : sink) if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite compressed attention sink");
    impl_ = std::make_unique<Impl>(sink, device);
}
CudaCompressedAttention::~CudaCompressedAttention() = default;
void CudaCompressedAttention::forward(const float* query, const float* kv, float* output, int tokens,
                                      int start_pos, CompressedAttentionSelection selection, void* stream) {
    auto& p = *impl_;
    if (tokens <= 0 || tokens > max_position || start_pos < 0 || start_pos != p.next ||
        start_pos > max_position - tokens || (start_pos && tokens != 1))
        throw std::invalid_argument("Compressed attention requires initial prefill then contiguous single-token decode");
    if (selection.groups != (start_pos + tokens) / 2 || selection.count < 0 ||
        selection.count > std::min(512, selection.groups) || selection.offset != (start_pos ? window : tokens))
        throw std::invalid_argument("Invalid compressed attention groups, count or offset");
    const void* pointers[] = {query, kv, output, selection.kv, selection.indices};
    const size_t sizes[] = {size_t(tokens) * qsize * 4, size_t(tokens) * dim * 4, size_t(tokens) * qsize * 4,
                           size_t(selection.groups) * dim * 2, size_t(tokens) * selection.count * 4};
    for (int i = 0; i < 5; ++i) {
        if (!sizes[i]) continue;
        if (!pointers[i] || uintptr_t(pointers[i]) % (i == 3 ? 2 : 4) || sizes[i] > UINTPTR_MAX - uintptr_t(pointers[i]) ||
            overlap(pointers[i], sizes[i], p.arena.base(), p.arena.capacity()))
            throw std::invalid_argument("Compressed attention needs aligned external buffers");
        for (int j = 0; j < i; ++j)
            if (sizes[j] && overlap(pointers[i], sizes[i], pointers[j], sizes[j]))
                throw std::invalid_argument("Compressed attention buffers overlap");
    }
    p.validate_stream(stream);
    int bad = 0;
    if (selection.count) {
        try {
            const auto s = static_cast<cudaStream_t>(stream);
            ck(cudaMemsetAsync(p.invalid, 0, 4, s), "clear selection validation");
            validate_selection<<<tokens, 512, 0, s>>>(selection.indices, selection.count, selection.groups,
                selection.offset, start_pos, p.invalid);
            ck(cudaGetLastError(), "validate compressed attention indices");
            ck(cudaMemcpyAsync(&bad, p.invalid, 4, cudaMemcpyDeviceToHost, s), "read selection validity");
            ck(cudaStreamSynchronize(s), "wait selection validation");
        } catch (...) { p.poisoned = true; throw; }
    }
    if (bad) throw std::invalid_argument("Compressed attention indices must be sorted causal group IDs then -1 padding");
    try {
        p.bind(stream);
        const int width = start_pos ? 0 : std::min(tokens, window);
        for (int t = 0; t < tokens; ++t)
            p.step(query + size_t(t) * qsize, kv + size_t(t) * dim, output + size_t(t) * qsize, selection,
                   selection.count ? selection.indices + size_t(t) * selection.count : nullptr, start_pos + t, width);
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait compressed attention");
        p.next += tokens;
    } catch (...) { p.poisoned = true; throw; }
}
void CudaCompressedAttention::reset(void* stream) {
    auto& p = *impl_; p.validate_stream(stream);
    try {
        p.bind(stream);
        ck(cudaMemsetAsync(p.ring, 0, window * dim * 2, static_cast<cudaStream_t>(stream)), "reset compressed attention ring");
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait compressed attention reset");
        p.next = 0;
    } catch (...) { p.poisoned = true; throw; }
}
int CudaCompressedAttention::next_position() const { return impl_->next; }
size_t CudaCompressedAttention::device_bytes() const { return impl_->arena.capacity(); }
const uint16_t* CudaCompressedAttention::kv_cache() const { return impl_->ring; }
} // namespace strata::deepseek41

namespace strata::deepseek41 {
struct SharedAttentionBatch::State {
    bool valid = false;
    uint64_t publication = 0, sequence = 0;
    int source_layer = -1, start = 0, tokens = 0, groups = 0, count = 0, offset = 0, device = 0;
    void* stream = nullptr;
    const uint16_t* kv = nullptr;
    const int32_t* indices = nullptr;
    // Optional source-20 publication; consumers borrow only KV/indices.
    const uint16_t* keys = nullptr;
    const int32_t* candidates = nullptr;
    int candidate_stride = 0;
    struct Range { const void* begin; size_t bytes; };
    std::array<Range, 5> source_ranges{};
    // An index-only owner borrows KV from a second live publication. Its own
    // indices expire when either owner advances, resets, fails or is destroyed.
    bool borrows_kv = false;
    std::weak_ptr<State> kv_source;
    uint64_t kv_publication = 0;
    bool current(uint64_t expected) const {
        if (!valid || publication != expected) return false;
        if (!borrows_kv) return true;
        const auto parent = kv_source.lock();
        return parent && parent->current(kv_publication);
    }
    void check_external(const void* p, size_t bytes) const {
        for (const auto& range : source_ranges)
            if (range.bytes && overlap(p, bytes, range.begin, range.bytes))
                throw std::invalid_argument("Shared attention buffer aliases source storage");
        if (borrows_kv) {
            const auto parent = kv_source.lock();
            if (!parent) throw std::invalid_argument("Shared KV owner expired");
            parent->check_external(p, bytes);
        }
    }
};
int SharedAttentionBatch::source_layer() const {
    const auto state = state_.lock();
    if (!state || !state->current(publication_))
        throw std::invalid_argument("No current shared attention publication");
    return state->source_layer;
}
struct CudaSourceAttention::Impl {
    AttentionProjections projections;
    CudaCompressedAttention core;
    CudaCompressor compressor;
    CudaCompressedKvCache compressed;
    CudaIndexKeyCache keys;
    CudaIndexQuery indexer;
    core::DeviceArena scratch;
    float* latent;
    int shared_prefill_capacity;
    int32_t* published_indices = nullptr;
    std::shared_ptr<SharedAttentionBatch::State> publication;
    bool failed = false;
    Impl(const AttentionWeights& a, const CompressorWeights& c, const IndexKeyWeights& k,
         const IndexQueryWeights& q, int capacity, int device, int shared_capacity)
        : projections(a, device), core(a.sink, device), compressor(c, device), compressed(capacity, device),
          keys(k, capacity, device), indexer(q, capacity, device),
          scratch(dim * 4 + aligned(size_t(shared_capacity) * std::min(512, capacity) * 4), device),
          shared_prefill_capacity(shared_capacity) {
        latent = static_cast<float*>(scratch.alloc(dim * 4));
        if (shared_capacity) {
            published_indices = static_cast<int32_t*>(scratch.alloc(size_t(shared_capacity) * std::min(512, capacity) * 4));
            publication = std::make_shared<SharedAttentionBatch::State>();
            publication->source_layer = a.layer;
            publication->kv = compressed.data(); publication->indices = published_indices; publication->device = device;
            publication->source_ranges = {{{scratch.base(), scratch.capacity()},
                {projections.arena.base(), projections.arena.capacity()}, {core.impl_->arena.base(), core.impl_->arena.capacity()},
                {compressed.data(), size_t(capacity) * dim * 2}, {keys.data(), size_t(capacity) * 128 * 2}}};
        }
    }
    void check_stream(void* stream) const {
        if (failed) throw std::logic_error("Source attention must be reconstructed after execution failure");
        core.impl_->validate_stream(stream);
    }
    void check_external(const void* p, size_t bytes) const {
        if (!p || uintptr_t(p) % 4 || bytes > UINTPTR_MAX - uintptr_t(p) ||
            overlap(p, bytes, scratch.base(), scratch.capacity()) ||
            overlap(p, bytes, projections.arena.base(), projections.arena.capacity()) ||
            overlap(p, bytes, core.impl_->arena.base(), core.impl_->arena.capacity()) ||
            overlap(p, bytes, compressed.data(), size_t(compressed.capacity()) * dim * 2) ||
            overlap(p, bytes, keys.data(), size_t(keys.capacity()) * 128 * 2))
            throw std::invalid_argument("Source attention needs aligned external buffers");
    }
};
CudaSourceAttention::CudaSourceAttention(const AttentionWeights& a, const CompressorWeights& c,
                                        const IndexKeyWeights& k, const IndexQueryWeights& q,
                                        int capacity_groups, int device, int shared_prefill_capacity, int layer) {
    if ((layer != 2 && layer != 8 && layer != 14) || a.layer != layer || c.layer != layer || k.layer != layer || q.layer != layer)
        throw std::invalid_argument("Source attention requires matching explicit projection/compressor/index weights for layer 2, 8 or 14");
    if (capacity_groups <= 0 || capacity_groups > max_position / 2)
        throw std::invalid_argument("Source attention capacity must be 1..524288 groups");
    if (shared_prefill_capacity < 0 || shared_prefill_capacity > max_position)
        throw std::invalid_argument("Shared prefill capacity must be 0..1048576 tokens");
    validate_weights(a);
    impl_ = std::make_unique<Impl>(a, c, k, q, capacity_groups, device, shared_prefill_capacity);
}
CudaSourceAttention::~CudaSourceAttention() = default;
void CudaSourceAttention::validate_forward(int tokens, int start_pos, void* stream) const {
    const auto& p = *impl_; p.check_stream(stream);
    if (tokens <= 0 || tokens > max_position || start_pos < 0 || start_pos != p.core.next_position() ||
        start_pos > max_position - tokens || (start_pos && tokens != 1))
        throw std::invalid_argument("Source attention requires initial prefill then contiguous single-token decode");
    const int groups = (start_pos + tokens) / 2;
    if (groups > p.compressed.capacity() || groups > p.keys.capacity() || groups > p.indexer.capacity())
        throw std::invalid_argument("Source attention call exceeds cache capacity before any state advance");
    if (p.publication && tokens > p.shared_prefill_capacity)
        throw std::invalid_argument("Source attention call exceeds preallocated shared-index capacity");
}
void CudaSourceAttention::check_external(const void* pointer, size_t bytes) const {
    impl_->check_external(pointer, bytes);
}
void CudaSourceAttention::begin_forward(void* stream) {
    auto& p = *impl_;
    if (p.publication) { p.publication->valid = false; ++p.publication->publication; }
    p.core.impl_->bind(stream);
}
void CudaSourceAttention::compress_token(const float* input, int position, void* stream) {
    auto& p = *impl_;
    const auto batch = p.compressor.forward(input, p.latent, 1, position, stream);
    // Both caches consume the original latent before the next token overwrites it.
    p.keys.append(p.latent, batch, stream);
    p.compressed.append(p.latent, batch, stream);
}
void CudaSourceAttention::attend_token(const float* input, float* output, int token,
                                      int tokens, int start_pos, void* stream) {
    auto& p = *impl_;
    const int groups = (start_pos + tokens) / 2;
    const int width = start_pos ? 0 : std::min(tokens, window);
    const int offset = start_pos ? window : tokens;
    p.projections.project_qkv(input, stream);
    const auto chosen = p.indexer.forward(input, p.projections.qr, p.keys.data(), groups, start_pos + token, offset, stream);
    if (p.publication && chosen.count)
        ck(cudaMemcpyAsync(p.published_indices + size_t(token) * chosen.count, chosen.indices,
                           size_t(chosen.count) * 4, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)),
           "retain shared per-query indices");
    // Only the owned, validated indexer supplies IDs to the private core here.
    p.core.impl_->step(p.projections.q, p.projections.kv, p.projections.attn,
                      {p.compressed.data(), groups, chosen.indices, chosen.count, offset},
                      chosen.indices, start_pos + token, width);
    p.projections.project_output(output, stream);
}
void CudaSourceAttention::finish_forward(int tokens, int start_pos, void* stream) {
    auto& p = *impl_;
    ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait source attention");
    p.core.impl_->next += tokens;
    if (p.publication) {
        auto& pub = *p.publication;
        pub.start = start_pos; pub.tokens = tokens; pub.groups = (start_pos + tokens) / 2;
        pub.count = std::min(512, pub.groups); pub.offset = start_pos ? window : tokens;
        pub.stream = stream; pub.valid = true;
    }
}
void CudaSourceAttention::fail_forward(void* stream) noexcept {
    auto& p = *impl_; p.failed = true;
    if (p.publication) p.publication->valid = false;
    (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}
void CudaSourceAttention::forward(const float* input, float* output, int tokens, int start_pos, void* stream) {
    validate_forward(tokens, start_pos, stream);
    const size_t bytes = size_t(tokens) * 5120 * 4;
    check_external(input, bytes); check_external(output, bytes);
    if (overlap(input, bytes, output, bytes)) throw std::invalid_argument("Source attention input/output overlap");
    try {
        begin_forward(stream);
        // Bounded two passes retain the original whole-prefill layout.
        for (int t = 0; t < tokens; ++t) compress_token(input + size_t(t) * 5120, start_pos + t, stream);
        for (int t = 0; t < tokens; ++t)
            attend_token(input + size_t(t) * 5120, output + size_t(t) * 5120, t, tokens, start_pos, stream);
        finish_forward(tokens, start_pos, stream);
    } catch (...) { fail_forward(stream); throw; }
}
void CudaSourceAttention::reset(void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    if (p.publication) { p.publication->valid = false; ++p.publication->publication; ++p.publication->sequence; }
    try {
        p.compressor.reset(stream); p.compressed.reset(stream); p.keys.reset(stream); p.core.reset(stream);
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); throw; }
}
SharedAttentionBatch CudaSourceAttention::shared_batch() const {
    const auto& p = *impl_;
    if (p.failed || !p.publication || !p.publication->valid)
        throw std::invalid_argument("No completed source attention batch available");
    return {p.publication, p.publication->publication};
}
int CudaSourceAttention::next_position() const { return impl_->core.next_position(); }
int CudaSourceAttention::capacity() const { return impl_->compressed.capacity(); }
int CudaSourceAttention::compressed_groups() const { return impl_->compressed.size(); }
size_t CudaSourceAttention::device_bytes() const {
    const auto& p = *impl_;
    return p.projections.arena.capacity() + p.core.device_bytes() + p.compressor.device_bytes() +
           p.compressed.device_bytes() + p.keys.device_bytes() + p.indexer.device_bytes() + p.scratch.capacity();
}
const uint16_t* CudaSourceAttention::kv_cache() const { return impl_->core.kv_cache(); }
const uint16_t* CudaSourceAttention::compressed_kv_cache() const { return impl_->compressed.data(); }
const uint16_t* CudaSourceAttention::index_key_cache() const { return impl_->keys.data(); }
} // namespace strata::deepseek41

namespace strata::deepseek41 {
struct CudaSharedAttention::Impl {
    AttentionProjections projections;
    CudaCompressedAttention core;
    std::weak_ptr<SharedAttentionBatch::State> source;
    uint64_t source_sequence = 0;
    bool bound = false;
    const int source_layer;
    std::unique_ptr<CudaCandidateIndexQuery> indexer;
    std::unique_ptr<core::DeviceArena> published_storage;
    int32_t* published_indices = nullptr;
    int capacity = 0, prefill_capacity = 0;
    std::shared_ptr<SharedAttentionBatch::State> publication;
    Impl(const AttentionWeights& w, int device, const CandidateIndexQueryWeights* q = nullptr, int cap = 0, int prefill = 0)
        : projections(w, device), core(w.sink, device),
          source_layer(q ? 20 : w.layer < 8 ? 2 : w.layer < 14 ? 8 : w.layer < 20 ? 14 : w.layer < 24 ? 20 : w.layer / 4 * 4) {
        if (q) {
            capacity = cap; prefill_capacity = prefill;
            indexer = std::make_unique<CudaCandidateIndexQuery>(*q, cap, device);
            published_storage = std::make_unique<core::DeviceArena>(aligned(size_t(prefill) * std::min(512, cap) * 4), device);
            published_indices = static_cast<int32_t*>(published_storage->alloc(size_t(prefill) * std::min(512, cap) * 4));
            publication = std::make_shared<SharedAttentionBatch::State>();
            publication->source_layer = w.layer; publication->device = device; publication->borrows_kv = true;
            publication->indices = published_indices;
            const auto storage = indexer->storage();
            publication->source_ranges = {{{published_storage->base(), published_storage->capacity()},
                {projections.arena.base(), projections.arena.capacity()}, {core.impl_->arena.base(), core.impl_->arena.capacity()},
                {storage.first, storage.second}, {nullptr, 0}}};
        }
    }
    void check_external(const void* p, size_t bytes, const SharedAttentionBatch::State& shared) const {
        if (!p || uintptr_t(p) % 4 || bytes > UINTPTR_MAX - uintptr_t(p) ||
            overlap(p, bytes, projections.arena.base(), projections.arena.capacity()) ||
            overlap(p, bytes, core.impl_->arena.base(), core.impl_->arena.capacity()))
            throw std::invalid_argument("Shared attention needs aligned external buffers");
        shared.check_external(p, bytes);
        if (publication) for (const auto& range : publication->source_ranges)
            if (range.bytes && overlap(p, bytes, range.begin, range.bytes))
                throw std::invalid_argument("Shared attention buffer aliases owned index storage");
    }
};
CudaSharedAttention::CudaSharedAttention(const AttentionWeights& w, int device, int layer) {
    if (layer < 3 || layer > 39 || layer == 8 || layer == 14 || layer == 20 || layer == 24 || layer == 28 || layer == 32 || layer == 36 || w.layer != layer)
        throw std::invalid_argument("Shared attention needs matching explicit consumer weights for layers 3..7/9..13/15..19/21..23/25..27/29..31/33..35/37..39");
    validate_weights(w); impl_ = std::make_unique<Impl>(w, device);
}
CudaSharedAttention::CudaSharedAttention(const AttentionWeights& w, const CandidateIndexQueryWeights& query,
                                         int capacity, int prefill, int device) {
    if ((w.layer != 24 && w.layer != 28 && w.layer != 32 && w.layer != 36) || query.layer != w.layer)
        throw std::invalid_argument("Candidate attention requires matching original source-24/source-28/source-32/source-36 weights");
    (void)Source20Batch(0, 1, 0, capacity, prefill);
    validate_weights(w); impl_ = std::make_unique<Impl>(w, device, &query, capacity, prefill);
}
CudaSharedAttention::~CudaSharedAttention() = default;
SharedAttentionBatch CudaSharedAttention::shared_batch() const {
    const auto& pub = impl_->publication;
    if (!pub || !pub->current(pub->publication))
        throw std::invalid_argument("No completed candidate index source publication");
    return {pub, pub->publication};
}
void CudaSharedAttention::validate_forward(int tokens, int start_pos,
                                           const SharedAttentionBatch& batch, void* stream) const {
    auto& p = *impl_; auto& state = *p.core.impl_;
    state.validate_stream(stream);
    if (tokens <= 0 || tokens > max_position || start_pos < 0 || start_pos != state.next ||
        start_pos > max_position - tokens || (start_pos && tokens != 1))
        throw std::invalid_argument("Shared attention requires initial prefill then contiguous single-token decode");
    const auto shared = batch.state_.lock();
    if (!shared || !shared->current(batch.publication_) || shared->source_layer != p.source_layer ||
        shared->start != start_pos || shared->tokens != tokens || shared->device != state.device || shared->stream != stream)
        throw std::invalid_argument("Shared attention needs the current matching original source batch on the same stream/device");
    if (p.indexer && (tokens > p.prefill_capacity || shared->groups > p.capacity || !shared->keys || !shared->candidates))
        throw std::invalid_argument("Candidate attention needs bounded source20 keys/candidates");
    if (p.bound && (p.source.lock() != shared || p.source_sequence != shared->sequence))
        throw std::invalid_argument("Shared attention source or sequence changed; reset the consumer first");
}
void CudaSharedAttention::check_external(const void* pointer, size_t bytes, const SharedAttentionBatch& batch) const {
    const auto shared = batch.state_.lock();
    if (!shared) throw std::invalid_argument("Shared attention source expired");
    impl_->check_external(pointer, bytes, *shared);
}
void CudaSharedAttention::begin_forward(void* stream) {
    auto& p = *impl_; p.core.impl_->bind(stream);
    if (p.publication) { p.publication->valid = false; ++p.publication->publication; }
}
void CudaSharedAttention::attend_token(const float* input, float* output, int token, int tokens, int start_pos,
                                      const SharedAttentionBatch& batch, void* stream) {
    auto& p = *impl_; const auto shared = batch.state_.lock();
    if (!shared) throw std::invalid_argument("Shared attention source expired");
    const int width = start_pos ? 0 : std::min(tokens, window);
    const int32_t* selected = shared->count ? shared->indices + size_t(token) * shared->count : nullptr;
    p.projections.project_qkv(input, stream);
    if (p.indexer) {
        const Source20Batch call(start_pos, tokens, start_pos, p.capacity, p.prefill_capacity);
        const auto chosen = p.indexer->forward(input, p.projections.qr, shared->keys, call, token,
            {shared->candidates + size_t(token) * shared->candidate_stride, shared->candidate_stride}, shared->offset, stream);
        selected = chosen.indices;
        ck(cudaMemcpyAsync(p.published_indices + size_t(token) * chosen.count, selected, size_t(chosen.count) * 4,
                           cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "retain candidate-source per-query indices");
    }
    const CompressedAttentionSelection selection{shared->kv, shared->groups, selected, shared->count, shared->offset};
    p.core.impl_->step(p.projections.q, p.projections.kv, p.projections.attn, selection, selected, start_pos + token, width);
    p.projections.project_output(output, stream);
}
void CudaSharedAttention::finish_forward(int tokens, const SharedAttentionBatch& batch, void* stream) {
    auto& p = *impl_; const auto shared = batch.state_.lock();
    if (!shared) throw std::invalid_argument("Shared attention source expired");
    ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait shared attention");
    p.core.impl_->next += tokens; p.source = shared; p.source_sequence = shared->sequence; p.bound = true;
    if (p.publication) {
        auto& pub = *p.publication;
        pub.start = shared->start; pub.tokens = tokens; pub.groups = shared->groups;
        pub.count = shared->count; pub.offset = shared->offset; pub.kv = shared->kv;
        pub.stream = stream; pub.kv_source = shared; pub.kv_publication = batch.publication_; pub.valid = true;
    }
}
void CudaSharedAttention::fail_forward(void* stream) noexcept {
    impl_->core.impl_->poisoned = true;
    if (impl_->publication) impl_->publication->valid = false;
    (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}
void CudaSharedAttention::forward(const float* input, float* output, int tokens, int start_pos,
                                  const SharedAttentionBatch& batch, void* stream) {
    validate_forward(tokens, start_pos, batch, stream);
    const size_t bytes = size_t(tokens) * 5120 * 4;
    check_external(input, bytes, batch); check_external(output, bytes, batch);
    if (overlap(input, bytes, output, bytes)) throw std::invalid_argument("Shared attention input/output overlap");
    try {
        begin_forward(stream);
        for (int t = 0; t < tokens; ++t)
            attend_token(input + size_t(t) * 5120, output + size_t(t) * 5120, t, tokens, start_pos, batch, stream);
        finish_forward(tokens, batch, stream);
    } catch (...) { fail_forward(stream); throw; }
}
void CudaSharedAttention::forward_source20(const float* input, float* output, int tokens, int start_pos,
                                  const Source20AttentionBatch& batch, void* stream) {
    forward(input, output, tokens, start_pos, batch.attention_batch(), stream);
}
void CudaSharedAttention::reset(void* stream) {
    auto& p = *impl_; p.core.impl_->validate_stream(stream);
    if (p.publication) {
        p.publication->valid = false; ++p.publication->publication; ++p.publication->sequence;
        p.publication->kv_source.reset();
    }
    p.core.reset(stream); p.source.reset(); p.bound = false;
}
int CudaSharedAttention::next_position() const { return impl_->core.next_position(); }
size_t CudaSharedAttention::device_bytes() const {
    const auto& p = *impl_;
    return p.projections.arena.capacity() + p.core.device_bytes() +
           (p.indexer ? p.indexer->device_bytes() + p.published_storage->capacity() : 0);
}
const uint16_t* CudaSharedAttention::kv_cache() const { return impl_->core.kv_cache(); }
} // namespace strata::deepseek41

namespace strata::deepseek41 {
SharedAttentionBatch Source20AttentionBatch::attention_batch() const {
    return SharedAttentionBatch(state_.lock(), publication_);
}
Source20AttentionView Source20AttentionBatch::view(int start, int tokens, void* stream) const {
    const auto state = state_.lock();
    if (!state || !state->current(publication_))
        throw std::invalid_argument("No current source-20 attention publication");
    int device = -1; ck(cudaGetDevice(&device), "source-20 publication device");
    if (start != state->start || tokens != state->tokens || !stream ||
        stream != state->stream || device != state->device)
        throw std::invalid_argument("Source-20 publication position, shape, stream or device mismatch");
    return {state->start, state->tokens, state->groups, state->count, state->offset, state->candidate_stride,
            state->sequence, state->kv, state->keys, state->indices, state->candidates};
}
struct CudaSource20Attention::Impl {
    AttentionProjections projections;
    CudaCompressedAttention core;
    CudaSource20Compressor compressor;
    CudaSource20KvCache compressed;
    CudaSource20IndexKeyCache keys;
    CudaSource20IndexQuery indexer;
    core::DeviceArena scratch;
    float* latent;
    int32_t *indices, *candidates;
    int prefill_capacity;
    std::shared_ptr<SharedAttentionBatch::State> publication;
    bool failed = false;
    Impl(const AttentionWeights& a, const Source20CompressorWeights& c, const Source20IndexKeyWeights& k,
         const Source20IndexQueryWeights& q, int capacity, int prefill, int device)
        : projections(a, device), core(a.sink, device), compressor(c, capacity, 1, device),
          compressed(capacity, device), keys(k, capacity, device), indexer(q, capacity, device),
          scratch(dim * 4 + aligned(size_t(prefill) * std::min(512, capacity) * 4) +
                  aligned(size_t(prefill) * std::min(2048, (capacity + 7) / 8) * 4), device),
          prefill_capacity(prefill), publication(std::make_shared<SharedAttentionBatch::State>()) {
        latent = static_cast<float*>(scratch.alloc(dim * 4));
        indices = static_cast<int32_t*>(scratch.alloc(size_t(prefill) * std::min(512, capacity) * 4));
        candidates = static_cast<int32_t*>(scratch.alloc(size_t(prefill) * std::min(2048, (capacity + 7) / 8) * 4));
        publication->device = device; publication->source_layer = 20;
        publication->kv = compressed.data(); publication->keys = keys.data();
        publication->indices = indices; publication->candidates = candidates;
        publication->source_ranges = {{{scratch.base(), scratch.capacity()},
            {projections.arena.base(), projections.arena.capacity()}, {core.impl_->arena.base(), core.impl_->arena.capacity()},
            {compressed.data(), size_t(capacity) * dim * 2}, {keys.data(), size_t(capacity) * 128 * 2}}};
    }
    void check_stream(void* stream) const {
        if (failed) throw std::logic_error("Source-20 attention must be reconstructed after execution failure");
        core.impl_->validate_stream(stream);
    }
    void check_external(const void* p, size_t bytes) const {
        if (!p || uintptr_t(p) % 4 || bytes > UINTPTR_MAX - uintptr_t(p) ||
            overlap(p, bytes, scratch.base(), scratch.capacity()) ||
            overlap(p, bytes, projections.arena.base(), projections.arena.capacity()) ||
            overlap(p, bytes, core.impl_->arena.base(), core.impl_->arena.capacity()) ||
            overlap(p, bytes, compressed.data(), size_t(compressed.capacity()) * dim * 2) ||
            overlap(p, bytes, keys.data(), size_t(keys.capacity()) * 128 * 2))
            throw std::invalid_argument("Source-20 attention needs aligned external buffers");
    }
    void fail(void* stream) noexcept {
        failed = true; publication->valid = false;
        (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    }
};
CudaSource20Attention::CudaSource20Attention(const AttentionWeights& a, const Source20CompressorWeights& c,
                                             const Source20IndexKeyWeights& k, const Source20IndexQueryWeights& q,
                                             int capacity, int prefill, int device) {
    if (a.layer != 20) throw std::invalid_argument("Source-20 attention requires its original projection profile");
    // Validate both limits before allocating any component or publication storage.
    (void)Source20Batch(0, 1, 0, capacity, prefill);
    validate_weights(a);
    impl_ = std::make_unique<Impl>(a, c, k, q, capacity, prefill, device);
}
CudaSource20Attention::~CudaSource20Attention() = default;
void CudaSource20Attention::validate_forward(int tokens, int start, void* stream) const {
    const auto& p = *impl_; p.check_stream(stream);
    (void)Source20Batch(start, tokens, p.core.next_position(), p.compressed.capacity(), p.prefill_capacity);
}
void CudaSource20Attention::check_external(const void* pointer, size_t bytes) const {
    impl_->check_external(pointer, bytes);
}
void CudaSource20Attention::begin_forward(void* stream) {
    auto& p = *impl_;
    p.publication->valid = false; ++p.publication->publication; p.core.impl_->bind(stream);
}
void CudaSource20Attention::compress_token(const float* input, int position, void* stream) {
    auto& p = *impl_;
    const auto one = p.compressor.forward(input, p.latent, 1, position, stream);
    p.keys.append(p.latent, one, stream);
    p.compressed.append(p.latent, one, stream);
}
void CudaSource20Attention::attend_token(const float* input, float* output, int token,
                                         int tokens, int start, void* stream) {
    auto& p = *impl_;
    const Source20Batch batch(start, tokens, p.core.next_position(), p.compressed.capacity(), p.prefill_capacity);
    const int count = std::min(512, batch.key_count()), stride = batch.candidate_stride();
    const int offset = start ? window : tokens, width = start ? 0 : std::min(tokens, window);
    p.projections.project_qkv(input, stream);
    const auto chosen = p.indexer.forward(input, p.projections.qr, p.keys.data(), batch, token, offset, stream);
    ck(cudaMemcpyAsync(p.indices + size_t(token) * count, chosen.index.indices, size_t(count) * 4,
                       cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "retain source-20 indices");
    ck(cudaMemcpyAsync(p.candidates + size_t(token) * stride, chosen.candidates.blocks, size_t(stride) * 4,
                       cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "retain source-20 candidates");
    // Only owned typed ratio-1 selection enters this generic private core.
    p.core.impl_->step(p.projections.q, p.projections.kv, p.projections.attn,
                      {p.compressed.data(), batch.key_count(), chosen.index.indices, count, offset},
                      chosen.index.indices, start + token, width);
    p.projections.project_output(output, stream);
}
void CudaSource20Attention::finish_forward(int tokens, int start, void* stream) {
    auto& p = *impl_;
    ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait source-20 attention");
    p.core.impl_->next = start + tokens;
    auto& pub = *p.publication;
    pub.start = start; pub.tokens = tokens; pub.groups = start + tokens;
    pub.count = std::min(512, start + tokens); pub.offset = start ? window : tokens;
    pub.candidate_stride = std::min(2048, (start + tokens + 7) / 8);
    pub.stream = stream; pub.valid = true;
}
void CudaSource20Attention::fail_forward(void* stream) noexcept { impl_->fail(stream); }
void CudaSource20Attention::forward(const float* input, float* output, int tokens, int start, void* stream) {
    validate_forward(tokens, start, stream);
    const size_t bytes = size_t(tokens) * 5120 * 4;
    check_external(input, bytes); check_external(output, bytes);
    if (overlap(input, bytes, output, bytes)) throw std::invalid_argument("Source-20 attention input/output overlap");
    try {
        begin_forward(stream);
        for (int t = 0; t < tokens; ++t) compress_token(input + size_t(t) * 5120, start + t, stream);
        for (int t = 0; t < tokens; ++t)
            attend_token(input + size_t(t) * 5120, output + size_t(t) * 5120, t, tokens, start, stream);
        finish_forward(tokens, start, stream);
    } catch (...) { fail_forward(stream); throw; }
}
void CudaSource20Attention::reset(void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    p.publication->valid = false; ++p.publication->publication; ++p.publication->sequence;
    try {
        p.compressor.reset(stream); p.compressed.reset(stream); p.keys.reset(stream); p.core.reset(stream);
    } catch (...) { p.fail(stream); throw; }
}
Source20AttentionBatch CudaSource20Attention::shared_batch() const {
    const auto& p = *impl_;
    if (p.failed || !p.publication->valid) throw std::invalid_argument("No completed source-20 attention call");
    return Source20AttentionBatch(p.publication, p.publication->publication);
}
int CudaSource20Attention::next_position() const { return impl_->core.next_position(); }
int CudaSource20Attention::capacity() const { return impl_->compressed.capacity(); }
int CudaSource20Attention::size() const { return impl_->compressed.size(); }
size_t CudaSource20Attention::device_bytes() const {
    const auto& p = *impl_;
    return p.projections.arena.capacity() + p.core.device_bytes() + p.compressor.device_bytes() +
           p.compressed.device_bytes() + p.keys.device_bytes() + p.indexer.device_bytes() + p.scratch.capacity();
}
const uint16_t* CudaSource20Attention::kv_cache() const { return impl_->core.kv_cache(); }
const uint16_t* CudaSource20Attention::compressed_kv_cache() const { return impl_->compressed.data(); }
const uint16_t* CudaSource20Attention::index_key_cache() const { return impl_->keys.data(); }
} // namespace strata::deepseek41
