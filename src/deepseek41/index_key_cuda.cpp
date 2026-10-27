#include "strata/deepseek41/source20_index_key_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int input_dim = 512, dim = 128;
constexpr size_t fixed_bytes = input_dim * dim * 2 + dim * 4 + input_dim * 2 + dim * 4;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
bool valid(const void* p, size_t bytes) {
    return p && uintptr_t(p) % 4 == 0 && bytes <= UINTPTR_MAX - uintptr_t(p);
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) < uintptr_t(b) + bn && uintptr_t(b) < uintptr_t(a) + an;
}
} // namespace
struct CudaIndexKeyCache::Impl {
    int device, capacity, position_step, count = 0;
    bool failed = false;
    void* stream = nullptr;
    std::unique_ptr<core::DeviceArena> arena;
    uint16_t *projection, *input, *cache;
    float *norm, *key;
    Impl(const IndexKeyWeights& w, int groups, int device_id, int step = 2)
        : device(device_id), capacity(groups), position_step(step) {
        if (groups <= 0 || groups > Source20Batch::max_context / position_step)
            throw std::invalid_argument("DeepSeek index key capacity exceeds its position range");
        if (w.projection.size() != input_dim * dim || w.norm.size() != dim ||
            !w.projection.data() || !w.norm.data())
            throw std::invalid_argument("DeepSeek index key needs [128,512] and [128] BF16 weights");
        for (auto view : {w.projection, w.norm})
            for (uint16_t bits : view)
                if (!std::isfinite(kernels::f32_from_bf16(bits)))
                    throw std::invalid_argument("Nonfinite DeepSeek index key weight");
        std::vector<float> gamma(dim);
        for (int i = 0; i < dim; ++i) gamma[i] = kernels::f32_from_bf16(w.norm[i]);
        // All sizes are multiples of DeviceArena's 256-byte alignment.
        arena = std::make_unique<core::DeviceArena>(fixed_bytes + size_t(groups) * dim * 2, device);
        projection = static_cast<uint16_t*>(arena->alloc(w.projection.size_bytes()));
        norm = static_cast<float*>(arena->alloc(dim * 4));
        input = static_cast<uint16_t*>(arena->alloc(input_dim * 2));
        key = static_cast<float*>(arena->alloc(dim * 4));
        cache = static_cast<uint16_t*>(arena->alloc(size_t(groups) * dim * 2));
        ck(cudaMemcpy(projection, w.projection.data(), w.projection.size_bytes(), cudaMemcpyHostToDevice), "upload index key projection");
        ck(cudaMemcpy(norm, gamma.data(), dim * 4, cudaMemcpyHostToDevice), "upload index key norm");
        ck(cudaMemset(cache, 0, size_t(groups) * dim * 2), "initialize index key cache");
        ck(cudaStreamSynchronize(nullptr), "wait for index key initialization");
    }
    void append(const float* latent, int rows, int first_position, void* stream);
    void reset(void* stream);
    void check_stream(void* s) const {
        if (!s || (stream && stream != s)) throw std::invalid_argument("DeepSeek index key cache needs one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get index key device");
        if (current != device) throw std::invalid_argument("DeepSeek index key called on wrong device");
        if (failed) throw std::logic_error("DeepSeek index key cache must be reconstructed after an execution failure");
    }
};
CudaIndexKeyCache::CudaIndexKeyCache(const IndexKeyWeights& weights, int capacity_groups, int device)
    : impl_(std::make_unique<Impl>(weights, capacity_groups, device)) {}
CudaIndexKeyCache::~CudaIndexKeyCache() = default;
void CudaIndexKeyCache::Impl::append(const float* latent, int rows, int first_position, void* stream) {
    auto& p = *this; p.check_stream(stream);
    if (rows < 0 || rows > p.capacity - p.count ||
        first_position != (rows ? p.position_step * p.count : -1))
        throw std::invalid_argument("DeepSeek index key append is noncontiguous, malformed or exceeds capacity");
    if (rows) {
        const size_t bytes = size_t(rows) * input_dim * 4;
        if (!valid(latent, bytes) || overlaps(latent, bytes, p.arena->base(), p.arena->capacity()))
            throw std::invalid_argument("DeepSeek index key cache requires an external aligned latent");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        for (int row = 0; row < rows; ++row) {
            kernels::f32_to_bf16_bulk(latent + size_t(row) * input_dim, p.input, input_dim, stream);
            kernels::bf16_gemv(p.input, p.projection, p.key, input_dim, dim, stream);
            detail::round_bf16_values(p.key, p.key, dim, stream);
            kernels::native_qsa_rms_norm_weighted(p.key, p.norm, p.key, dim, 1, 1e-20f, stream);
            detail::round_bf16_values(p.key, p.key, dim, stream);
            const auto postprocess = p.position_step == 1 ? postprocess_source20_index_keys : postprocess_index_keys;
            postprocess(p.key, p.cache + size_t(p.count + row) * dim, 1,
                        first_position + p.position_step * row, stream);
        }
        ck(cudaStreamSynchronize(s), "wait for index key append"); p.count += rows;
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaIndexKeyCache::Impl::reset(void* stream) {
    auto& p = *this; p.check_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        ck(cudaMemsetAsync(p.cache, 0, size_t(p.capacity) * dim * 2, s), "clear index key cache");
        ck(cudaStreamSynchronize(s), "wait for index key reset"); p.count = 0;
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaIndexKeyCache::append(const float* latent, CompressedBatch batch, void* stream) {
    impl_->append(latent, batch.count, batch.first_position, stream);
}
void CudaIndexKeyCache::reset(void* stream) { impl_->reset(stream); }
int CudaIndexKeyCache::size() const { return impl_->count; }
int CudaIndexKeyCache::capacity() const { return impl_->capacity; }
size_t CudaIndexKeyCache::device_bytes() const { return impl_->arena->capacity(); }
const uint16_t* CudaIndexKeyCache::data() const { return impl_->cache; }
CudaSource20IndexKeyCache::CudaSource20IndexKeyCache(const Source20IndexKeyWeights& weights, int capacity_tokens, int device)
    : impl_(std::make_unique<CudaIndexKeyCache::Impl>(IndexKeyWeights{weights.projection, weights.norm, 20},
                                                   capacity_tokens, device, 1)) {}
CudaSource20IndexKeyCache::~CudaSource20IndexKeyCache() = default;
void CudaSource20IndexKeyCache::append(const float* latent, const Source20Batch& batch, void* stream) {
    impl_->append(latent, batch.latent_count(), batch.first_position(), stream);
}
void CudaSource20IndexKeyCache::reset(void* stream) { impl_->reset(stream); }
int CudaSource20IndexKeyCache::size() const { return impl_->count; }
int CudaSource20IndexKeyCache::capacity() const { return impl_->capacity; }
size_t CudaSource20IndexKeyCache::device_bytes() const { return impl_->arena->capacity(); }
const uint16_t* CudaSource20IndexKeyCache::data() const { return impl_->cache; }
const float* CudaSource20IndexKeyCache::last_normalized() const { return impl_->count ? impl_->key : nullptr; }
} // namespace strata::deepseek41
