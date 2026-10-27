#include "strata/deepseek41/host_expert_cache.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek41 {
HostExpertCacheLayout host_expert_cache_layout(const Geometry& g, uint64_t budget) {
    if (g.layers < 1 || g.layers > 40 || g.experts != 384 || g.top_k != 6)
        throw std::invalid_argument("Host expert cache requires 1017 backbone geometry");
    HostExpertCacheLayout p; p.blob_bytes = nvfp4_expert_bytes(g);
    if (!budget) return p;
    const size_t pairs = size_t(g.layers) * size_t(g.experts), index_bytes = pairs * sizeof(int32_t);
    const size_t entry_bytes = p.blob_bytes + 3 * sizeof(int32_t);
    if (budget < index_bytes || budget - index_bytes < entry_bytes)
        throw std::invalid_argument("CPU expert cache budget cannot hold one expert and its metadata");
    p.slots = size_t(std::min<uint64_t>(pairs, (budget - index_bytes) / entry_bytes));
    p.payload_bytes = p.slots * p.blob_bytes;
    p.metadata_bytes = index_bytes + p.slots * 3 * sizeof(int32_t);
    return p;
}
HostExpertCache::HostExpertCache(const Geometry& g, uint64_t budget)
    : layers_(g.layers), experts_(g.experts), layout_(host_expert_cache_layout(g, budget)) {
    if (!layout_.slots) return;
    // Allocate once, without touching every payload page before its first fill.
    payload_.reset(new uint8_t[layout_.payload_bytes]);
    index_ = std::make_unique<int32_t[]>(size_t(layers_) * size_t(experts_));
    std::fill_n(index_.get(), size_t(layers_) * size_t(experts_), -1);
    slots_ = std::make_unique<Slot[]>(layout_.slots);
}
int32_t HostExpertCache::key(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= layers_ || expert < 0 || expert >= experts_)
        throw std::out_of_range("Host cache original layer/expert outside geometry");
    return int32_t(layer * experts_ + expert);
}
void HostExpertCache::unlink(int32_t i) {
    auto& s = slots_[size_t(i)];
    if (s.older < 0) oldest_ = s.newer; else slots_[size_t(s.older)].newer = s.newer;
    if (s.newer < 0) newest_ = s.older; else slots_[size_t(s.newer)].older = s.older;
}
void HostExpertCache::newest(int32_t i) {
    auto& s = slots_[size_t(i)]; s.older = newest_; s.newer = -1;
    if (newest_ < 0) oldest_ = i; else slots_[size_t(newest_)].newer = i;
    newest_ = i;
}
const uint8_t* HostExpertCache::find(int64_t layer, int64_t expert) {
    const auto k = key(layer, expert); const int32_t slot = layout_.slots ? index_[size_t(k)] : -1;
    if (slot < 0) { ++stats_.misses; return nullptr; }
    ++stats_.hits; unlink(slot); newest(slot);
    return payload_.get() + size_t(slot) * layout_.blob_bytes;
}
const uint8_t* HostExpertCache::insert(int64_t layer, int64_t expert, std::span<const uint8_t> blob) {
    const auto k = key(layer, expert);
    if (!layout_.slots || !blob.data() || blob.size() != layout_.blob_bytes)
        throw std::invalid_argument("Host cache insertion needs a slot and a complete native expert");
    int32_t slot = index_[size_t(k)];
    if (slot >= 0) unlink(slot);
    else if (filled_ < layout_.slots) slot = int32_t(filled_++);
    else {
        slot = oldest_; unlink(slot); index_[size_t(slots_[size_t(slot)].key)] = -1; ++stats_.evictions;
    }
    auto* destination = payload_.get() + size_t(slot) * layout_.blob_bytes;
    std::memmove(destination, blob.data(), blob.size());
    slots_[size_t(slot)].key = k; index_[size_t(k)] = slot; newest(slot);
    return destination;
}
} // namespace strata::deepseek41
