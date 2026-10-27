#include "strata/deepseek41/source20_kv_cuda.hpp"

#include <cuda_runtime.h>

namespace strata::deepseek41 {
namespace {
constexpr int dim = 512;
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

struct CudaSource20KvCache::Impl {
    int device, capacity, count = 0;
    bool failed = false;
    void* stream = nullptr;
    std::unique_ptr<core::DeviceArena> arena;
    uint16_t* cache;

    Impl(int tokens, int device_id) : device(device_id), capacity(tokens) {
        if (tokens < 1 || tokens > Source20Batch::max_context)
            throw std::invalid_argument("Source-20 KV capacity must be 1..1048576 tokens");
        arena = std::make_unique<core::DeviceArena>(size_t(tokens) * dim * sizeof(uint16_t), device);
        cache = static_cast<uint16_t*>(arena->alloc(arena->capacity()));
        ck(cudaMemset(cache, 0, arena->capacity()), "initialize source-20 KV cache");
        ck(cudaStreamSynchronize(nullptr), "wait for source-20 KV initialization");
    }
    void check_stream(void* s) const {
        if (!s || (stream && stream != s)) throw std::invalid_argument("Source-20 KV cache needs one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get source-20 KV device");
        if (current != device) throw std::invalid_argument("Source-20 KV called on wrong device");
        if (failed) throw std::logic_error("Source-20 KV cache must be reconstructed after an execution failure");
    }
};

CudaSource20KvCache::CudaSource20KvCache(int capacity_tokens, int device)
    : impl_(std::make_unique<Impl>(capacity_tokens, device)) {}
CudaSource20KvCache::~CudaSource20KvCache() = default;

void CudaSource20KvCache::append(const float* latent, const Source20Batch& batch, void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    if (batch.first_position() != p.count || batch.latent_count() > p.capacity - p.count)
        throw std::invalid_argument("Source-20 KV append must be contiguous and fit capacity");
    const size_t bytes = size_t(batch.latent_count()) * dim * sizeof(float);
    if (!valid(latent, bytes) || overlaps(latent, bytes, p.arena->base(), p.arena->capacity()))
        throw std::invalid_argument("Source-20 KV cache requires an aligned external latent");
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        postprocess_source20_kv(latent, p.cache + size_t(p.count) * dim,
                                batch.latent_count(), batch.first_position(), stream);
        ck(cudaStreamSynchronize(s), "wait for source-20 KV append");
        p.count = batch.key_count();
    } catch (...) {
        p.failed = true; (void)cudaStreamSynchronize(s); throw;
    }
}
void CudaSource20KvCache::reset(void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        ck(cudaMemsetAsync(p.cache, 0, p.arena->capacity(), s), "clear source-20 KV cache");
        ck(cudaStreamSynchronize(s), "wait for source-20 KV reset");
        p.count = 0;
    } catch (...) {
        p.failed = true; (void)cudaStreamSynchronize(s); throw;
    }
}
int CudaSource20KvCache::size() const { return impl_->count; }
int CudaSource20KvCache::capacity() const { return impl_->capacity; }
size_t CudaSource20KvCache::device_bytes() const { return impl_->arena->capacity(); }
const uint16_t* CudaSource20KvCache::data() const { return impl_->cache; }

} // namespace strata::deepseek41
