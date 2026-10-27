#pragma once

#include "strata/core/expert_source.hpp"
#include "strata/deepseek41/artifact.hpp"

#include "strata/deepseek41/host_expert_cache.hpp"
#include "strata/deepseek41/expert_file.hpp"
#include <atomic>
#include <mutex>

namespace strata::deepseek41 {
// Adapter to Strata's existing source interface, with ONE reusable NVFP4 staging
// blob. The mapped model/index outlive this object. No full-model RAM copy.
// blob() has one caller; copy_blob() supports separate caller-owned destinations.
// Asynchronous consumers call note_async_read() or wait before the next blob().
class Mxfp4ExpertSource final : public core::ExpertSource {
  public:
    Mxfp4ExpertSource(const Geometry& geometry, const ExpertIndex& index);
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* blob_stable(int64_t, int64_t) override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    void note_async_read(const uint8_t* blob, void* stream) override;
    bool transient(int64_t, int64_t) const override { return true; }
    int64_t reads() const override { return reads_.load(std::memory_order_relaxed); }
    size_t staging_bytes() const { return staging_.size(); }

  private:
    Geometry geometry_;
    const ExpertIndex& index_;
    std::vector<uint8_t> staging_;
    std::atomic<int64_t> reads_{0};
};
// Adds a global bounded host tier to any existing native expert source. Upstream
// outlives this wrapper. blob() has one caller; copy_blob() serializes callers.
// Retained/asynchronous users must copy or finish/note the read before another
// blob() call. Entries become visible only after a successful upstream read.
class CachedExpertSource final : public core::ExpertSource {
  public:
    CachedExpertSource(const Geometry& g, core::ExpertSource& source, uint64_t budget)
        : source_(source), cache_(g, budget) {}
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* blob_stable(int64_t, int64_t) override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t count) override;
    void note_async_read(const uint8_t*, void* stream) override;
    bool transient(int64_t, int64_t) const override { return true; }
    int64_t reads() const override;
    HostExpertCacheStats stats() const;
    size_t allocated_bytes() const { return cache_.layout().allocated_bytes(); }
    size_t resident_bytes() const;
  private:
    const uint8_t* load(int64_t layer, int64_t expert);
    core::ExpertSource& source_;
    HostExpertCache cache_;
    mutable std::mutex mutex_;
};
// Takes ownership of an immutable original-ID index. GGUF mappings used to
// construct it may be closed before this source opens; only indexed demand reads
// supply weights. Single blob caller; copy_blob serializes callers. Same transient
// pointer/async completion contract as Mxfp4ExpertSource and CachedExpertSource.
class IndexedExpertSource final : public core::ExpertSource {
  public:
    explicit IndexedExpertSource(ExpertFileIndex index);
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    const uint8_t* blob_stable(int64_t, int64_t) override;
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override;
    void begin_layer(int64_t layer, const int32_t* ids, int64_t count) override;
    void note_async_read(const uint8_t*, void* stream) override;
    bool transient(int64_t, int64_t) const override { return true; }
    int64_t reads() const override;
    ExpertFileStats io_stats() const;
    const ExpertFileIndex& index() const { return index_; }
    size_t reader_staging_bytes() const { return reader_.staging_bytes(); }
    size_t staging_bytes() const { return reader_.staging_bytes() + native_.size(); }
    size_t host_upper_bytes() const { return staging_bytes() + index_.metadata_upper_bytes(); }
  private:
    const ExpertFileIndex index_;
    IndexedExpertFileReader reader_;
    std::vector<uint8_t> native_;
    mutable std::mutex mutex_;
};
} // namespace strata::deepseek41
