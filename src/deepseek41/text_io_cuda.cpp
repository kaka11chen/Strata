#include "strata/deepseek41/text_io_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "expert_ops.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace strata::deepseek41 {
namespace {
void ck(cudaError_t e, const char* label) {
    if (e != cudaSuccess) throw core::CudaError(std::string(label) + ": " + cudaGetErrorString(e), int(e));
}
size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    return an && bn && (uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn);
}
}
CudaTextIo::CudaTextIo(const TextIoWeights& w, int capacity, int device)
    : hidden_(w.hidden), vocab_(w.vocab_size), capacity_(capacity), device_(device), embedding_(w.embedding) {
    if (hidden_ <= 0 || hidden_ > 5120 || hidden_ % 32 || vocab_ <= 0 || vocab_ > 129280 ||
        capacity_ <= 0 || capacity_ > 1048576 || !w.embedding.data() || !w.norm.data() || !w.head.data() ||
        w.embedding.size() != size_t(hidden_) * vocab_ || w.head.size() != w.embedding.size() ||
        w.norm.size() != size_t(hidden_) || uintptr_t(w.embedding.data()) % 2 ||
        uintptr_t(w.norm.data()) % 2 || uintptr_t(w.head.data()) % 2)
        throw std::invalid_argument("DeepSeek text IO needs bounded dimensions and three matching BF16 tensors");
    for (auto bits : w.head)
        if ((bits & 0x7f80u) == 0x7f80u) throw std::invalid_argument("Nonfinite DeepSeek output head weight");
    std::vector<float> norm(size_t(hidden_), 0.f);
    for (int i = 0; i < hidden_; ++i) {
        norm[size_t(i)] = kernels::f32_from_bf16(w.norm[size_t(i)]);
        if (!std::isfinite(norm[size_t(i)])) throw std::invalid_argument("Nonfinite DeepSeek output norm weight");
    }
    rows_.resize(size_t(capacity_) * hidden_);
    initial_.resize(size_t(capacity_) * 4, 0.f);
    for (int t = 0; t < capacity_; ++t) initial_[size_t(t) * 4] = 1.f;
    const size_t n = size_t(hidden_) * sizeof(float);
    size_t bytes = 0;
    for (size_t part : {w.head.size_bytes(), n, 4 * n, n}) bytes = aligned(bytes) + part;
    arena_ = std::make_unique<core::DeviceArena>(bytes, device_);
    head_ = static_cast<uint16_t*>(arena_->alloc(w.head.size_bytes()));
    norm_ = static_cast<float*>(arena_->alloc(n));
    residual_ = static_cast<float*>(arena_->alloc(4 * n));
    normalized_ = static_cast<float*>(arena_->alloc(n));
    ck(cudaMemcpy(head_, w.head.data(), w.head.size_bytes(), cudaMemcpyHostToDevice), "upload DeepSeek output head");
    ck(cudaMemcpy(norm_, norm.data(), n, cudaMemcpyHostToDevice), "upload DeepSeek output norm");
    ck(cudaStreamSynchronize(nullptr), "wait for DeepSeek text IO upload");
}
void CudaTextIo::check_stream(void* stream) const {
    if (!stream || (stream_ && stream_ != stream)) throw std::invalid_argument("DeepSeek text IO needs one explicit stream");
    int device = -1; ck(cudaGetDevice(&device), "get DeepSeek text IO device");
    if (device != device_) throw std::invalid_argument("DeepSeek text IO called on wrong device");
    if (failed_) throw std::logic_error("DeepSeek text IO must be reconstructed after execution failure");
}
void CudaTextIo::check_external(const void* pointer, size_t bytes) const {
    if (!pointer || uintptr_t(pointer) % 4 || bytes > UINTPTR_MAX - uintptr_t(pointer) ||
        overlap(pointer, bytes, arena_->base(), arena_->capacity()))
        throw std::invalid_argument("DeepSeek text IO needs aligned external device buffers");
}
void CudaTextIo::embed(std::span<const int32_t> tokens, float* residual, float* initial, void* stream) {
    check_stream(stream);
    if (tokens.empty() || tokens.size() > size_t(capacity_)) throw std::invalid_argument("DeepSeek embedding token count exceeds capacity");
    const size_t width = size_t(hidden_) * sizeof(float), pre_bytes = tokens.size() * 4 * sizeof(float);
    check_external(residual, tokens.size() * 4 * width); check_external(initial, pre_bytes);
    if (overlap(residual, tokens.size() * 4 * width, initial, pre_bytes)) throw std::invalid_argument("DeepSeek embedding outputs overlap");
    for (int32_t token : tokens)
        if (token < 0 || token >= vocab_) throw std::invalid_argument("DeepSeek embedding original token ID out of range");
    for (size_t t = 0; t < tokens.size(); ++t)
        for (int d = 0; d < hidden_; ++d) {
            const float value = kernels::f32_from_bf16(embedding_[size_t(tokens[t]) * hidden_ + size_t(d)]);
            if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite DeepSeek embedding row");
            rows_[t * hidden_ + size_t(d)] = value;
        }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        stream_ = stream;
        for (int copy = 0; copy < 4; ++copy)
            ck(cudaMemcpy2DAsync(residual + size_t(copy) * hidden_, 4 * width, rows_.data(), width,
                                 width, tokens.size(), cudaMemcpyHostToDevice, s), "copy DeepSeek embedding streams");
        ck(cudaMemcpyAsync(initial, initial_.data(), pre_bytes, cudaMemcpyHostToDevice, s), "initialize DeepSeek pre mix");
        ck(cudaStreamSynchronize(s), "wait for DeepSeek embedding");
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaTextIo::head(const float* residual, const float* pre, int tokens, float* logits, void* stream) {
    check_stream(stream);
    if (tokens <= 0 || tokens > capacity_) throw std::invalid_argument("DeepSeek output head token count exceeds capacity");
    const size_t n = size_t(hidden_), residual_bytes = size_t(tokens) * 4 * n * sizeof(float);
    const size_t pre_bytes = size_t(tokens) * 4 * sizeof(float), logits_bytes = size_t(vocab_) * sizeof(float);
    const void* pointers[] = {residual, pre, logits};
    const size_t bytes[] = {residual_bytes, pre_bytes, logits_bytes};
    for (int i = 0; i < 3; ++i) {
        check_external(pointers[i], bytes[i]);
        for (int j = 0; j < i; ++j)
            if (overlap(pointers[i], bytes[i], pointers[j], bytes[j])) throw std::invalid_argument("DeepSeek output head buffers overlap");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        stream_ = stream;
        detail::round_bf16_values(residual + size_t(tokens - 1) * 4 * n, residual_, 4 * hidden_, stream);
        detail::collapse_mhc_values(residual_, pre + size_t(tokens - 1) * 4, normalized_, hidden_, stream);
        kernels::native_qsa_rms_norm_weighted(normalized_, norm_, normalized_, hidden_, 1, 1e-20f, stream);
        detail::round_bf16_values(normalized_, normalized_, hidden_, stream);
        kernels::bf16_gemv_fp32_mmvf(normalized_, head_, logits, hidden_, vocab_, stream);
        ck(cudaStreamSynchronize(s), "wait for DeepSeek output head");
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
} // namespace strata::deepseek41
