#include "strata/deepseek41/source20_compressor_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120, dim = 512;
constexpr size_t device_size = size_t(hidden) * dim * 2 + dim * 4 + hidden * 2 + dim * 4;
void ck(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(status), int(status));
}
bool valid(const void* pointer, size_t bytes) {
    return pointer && uintptr_t(pointer) % alignof(float) == 0 && bytes <= UINTPTR_MAX - uintptr_t(pointer);
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) < uintptr_t(b) + bn && uintptr_t(b) < uintptr_t(a) + an;
}
} // namespace

struct CudaSource20Compressor::Impl {
    int device, context, prefill, next = 0;
    bool failed = false;
    void* stream = nullptr;
    std::unique_ptr<core::DeviceArena> arena;
    uint16_t *weight, *input;
    float *norm, *projection;

    Impl(const Source20CompressorWeights& w, int capacity, int prefill_capacity, int device_id)
        : device(device_id), context(capacity), prefill(prefill_capacity) {
        // Validate capacities before any GPU allocation.
        (void)Source20Batch(0, 1, 0, context, prefill);
        if (w.kv.size() != size_t(hidden) * dim || w.norm.size() != dim || !w.kv.data() || !w.norm.data())
            throw std::invalid_argument("Source-20 compressor needs [512,5120] and [512] BF16 weights");
        for (auto view : {w.kv, w.norm})
            for (uint16_t bits : view)
                if (!std::isfinite(kernels::f32_from_bf16(bits)))
                    throw std::invalid_argument("Nonfinite source-20 compressor weight");
        std::vector<float> gamma(dim);
        for (int i = 0; i < dim; ++i) gamma[size_t(i)] = kernels::f32_from_bf16(w.norm[size_t(i)]);

        // Each size is a multiple of DeviceArena's 256-byte alignment.
        arena = std::make_unique<core::DeviceArena>(device_size, device);
        weight = static_cast<uint16_t*>(arena->alloc(w.kv.size_bytes()));
        norm = static_cast<float*>(arena->alloc(dim * 4));
        input = static_cast<uint16_t*>(arena->alloc(hidden * 2));
        projection = static_cast<float*>(arena->alloc(dim * 4));
        ck(cudaMemcpy(weight, w.kv.data(), w.kv.size_bytes(), cudaMemcpyHostToDevice), "upload source-20 projection");
        ck(cudaMemcpy(norm, gamma.data(), dim * 4, cudaMemcpyHostToDevice), "upload source-20 norm");
        ck(cudaStreamSynchronize(nullptr), "wait for source-20 weights");
    }
    void check_stream(void* s) const {
        if (!s || (stream && stream != s)) throw std::invalid_argument("Source-20 compressor needs one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get source-20 compressor device");
        if (current != device) throw std::invalid_argument("Source-20 compressor called on wrong device");
        if (failed) throw std::logic_error("Source-20 compressor must be reconstructed after an execution failure");
    }
    void check_external(const void* pointer, size_t bytes) const {
        if (!valid(pointer, bytes) || overlaps(pointer, bytes, arena->base(), arena->capacity()))
            throw std::invalid_argument("Source-20 compressor needs aligned external device buffers");
    }
};

CudaSource20Compressor::CudaSource20Compressor(const Source20CompressorWeights& weights,
                                               int context_capacity, int prefill_capacity, int device)
    : impl_(std::make_unique<Impl>(weights, context_capacity, prefill_capacity, device)) {}
CudaSource20Compressor::~CudaSource20Compressor() = default;

Source20Batch CudaSource20Compressor::forward(const float* input, float* output, int tokens, int start, void* stream) {
    auto& p = *impl_;
    p.check_stream(stream);
    const Source20Batch batch(start, tokens, p.next, p.context, p.prefill);
    const size_t input_bytes = size_t(tokens) * hidden * sizeof(float);
    const size_t output_bytes = size_t(tokens) * dim * sizeof(float);
    p.check_external(input, input_bytes); p.check_external(output, output_bytes);
    if (overlaps(input, input_bytes, output, output_bytes))
        throw std::invalid_argument("Source-20 compressor input and output overlap");
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        for (int token = 0; token < tokens; ++token) {
            kernels::f32_to_bf16_bulk(input + size_t(token) * hidden, p.input, hidden, stream);
            kernels::bf16_gemv(p.input, p.weight, p.projection, hidden, dim, stream);
            // Unlike ratio 2, projection rounds BEFORE normalization.
            detail::round_bf16_values(p.projection, p.projection, dim, stream);
            auto* destination = output + size_t(token) * dim;
            kernels::native_qsa_rms_norm_weighted(p.projection, p.norm, destination, dim, 1, 1e-20f, stream);
            detail::round_bf16_values(destination, destination, dim, stream);
        }
        ck(cudaStreamSynchronize(s), "wait for source-20 compressor");
        p.next = batch.key_count();
        return batch;
    } catch (...) {
        p.failed = true; (void)cudaStreamSynchronize(s); throw;
    }
}
void CudaSource20Compressor::reset(void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        ck(cudaStreamSynchronize(s), "wait for source-20 compressor reset");
        p.stream = stream; p.next = 0;
        // Scratch is overwritten in full on every token; there is no pair state.
    } catch (...) {
        p.failed = true; (void)cudaStreamSynchronize(s); throw;
    }
}
int CudaSource20Compressor::next_position() const { return impl_->next; }
size_t CudaSource20Compressor::device_bytes() const { return impl_->arena->capacity(); }
const float* CudaSource20Compressor::last_projection() const {
    return impl_->next && !impl_->failed ? impl_->projection : nullptr;
}
} // namespace strata::deepseek41
