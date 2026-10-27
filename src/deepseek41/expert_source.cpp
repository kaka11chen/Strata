#include "strata/deepseek41/expert_source.hpp"
#include "strata/deepseek41/fp4_layout.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <utility>

namespace strata::deepseek41 {
Mxfp4ExpertSource::Mxfp4ExpertSource(const Geometry& g, const ExpertIndex& index)
    : geometry_(g), index_(index), staging_(nvfp4_expert_bytes(g)) {}
const uint8_t* Mxfp4ExpertSource::blob(int64_t layer, int64_t expert) {
    copy_blob(layer, expert, staging_.data());
    return staging_.data();
}
const uint8_t* Mxfp4ExpertSource::blob_stable(int64_t, int64_t) {
    throw std::logic_error("DeepSeek staging is transient; use copy_blob for retained host bytes");
}
bool Mxfp4ExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (!dst)
        return false;
    if (layer < 0 || layer >= geometry_.layers || expert < 0 || expert >= geometry_.experts)
        throw std::out_of_range("DeepSeek source layer/expert outside geometry");
    pack_nvfp4_expert(geometry_, index_.view(int(layer), int(expert)), {dst, staging_.size()});
    reads_.fetch_add(1, std::memory_order_relaxed);
    return true;
}
void Mxfp4ExpertSource::note_async_read(const uint8_t* blob, void* stream) {
    if (blob != staging_.data())
        return;
    // One staging buffer: complete the DMA before allowing it to be recycled.
    const auto status = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("DeepSeek source staging wait: ") + cudaGetErrorString(status));
}
const uint8_t* CachedExpertSource::load(int64_t layer, int64_t expert) {
    if (const auto* hit = cache_.find(layer, expert)) return hit;
    const auto* data = source_.blob(layer, expert);
    if (!data || !cache_.layout().slots) return data;
    return cache_.insert(layer, expert, {data, cache_.layout().blob_bytes});
}
const uint8_t* CachedExpertSource::blob(int64_t layer, int64_t expert) {
    std::lock_guard lock(mutex_); return load(layer, expert);
}
const uint8_t* CachedExpertSource::blob_stable(int64_t, int64_t) {
    throw std::logic_error("DeepSeek host cache entries may be evicted; use copy_blob for retained bytes");
}
bool CachedExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (!dst) return false;
    std::lock_guard lock(mutex_); const auto* data = load(layer, expert);
    if (!data) return false;
    std::memcpy(dst, data, cache_.layout().blob_bytes); return true;
}
void CachedExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t count) {
    std::lock_guard lock(mutex_); source_.begin_layer(layer, ids, count);
}
void CachedExpertSource::note_async_read(const uint8_t*, void* stream) {
    std::lock_guard lock(mutex_);
    const auto status = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) throw std::runtime_error(std::string("DeepSeek host cache wait: ") + cudaGetErrorString(status));
}
HostExpertCacheStats CachedExpertSource::stats() const {
    std::lock_guard lock(mutex_); return cache_.stats();
}
int64_t CachedExpertSource::reads() const { const auto s = stats(); return int64_t(s.hits + s.misses); }
size_t CachedExpertSource::resident_bytes() const { std::lock_guard lock(mutex_); return cache_.resident_bytes(); }
IndexedExpertSource::IndexedExpertSource(ExpertFileIndex index)
    : index_(std::move(index)), reader_(index_), native_(nvfp4_expert_bytes(index_.geometry())) {}
const uint8_t* IndexedExpertSource::blob(int64_t layer, int64_t expert) {
    std::lock_guard lock(mutex_); reader_.read(layer, expert, native_); return native_.data();
}
const uint8_t* IndexedExpertSource::blob_stable(int64_t, int64_t) {
    throw std::logic_error("Indexed DeepSeek source is transient; use copy_blob for retained bytes");
}
bool IndexedExpertSource::copy_blob(int64_t layer, int64_t expert, uint8_t* dst) {
    if (!dst) return false;
    std::lock_guard lock(mutex_); reader_.read(layer, expert, {dst, native_.size()}); return true;
}
void IndexedExpertSource::begin_layer(int64_t layer, const int32_t* ids, int64_t count) {
    if (!ids || count != index_.geometry().top_k) throw std::invalid_argument("Indexed source requires a complete original top-6 route");
    // Validate the whole route before any payload read or GPU cache fill.
    for (int64_t k = 0; k < count; ++k) (void)index_.at(layer, ids[k]);
}
void IndexedExpertSource::note_async_read(const uint8_t*, void* stream) {
    std::lock_guard lock(mutex_); const auto status = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) throw std::runtime_error(std::string("Indexed DeepSeek source wait: ") + cudaGetErrorString(status));
}
ExpertFileStats IndexedExpertSource::io_stats() const { std::lock_guard lock(mutex_); return reader_.stats(); }
int64_t IndexedExpertSource::reads() const { return int64_t(io_stats().converted); }
} // namespace strata::deepseek41
