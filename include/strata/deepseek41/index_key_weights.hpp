#pragma once

#include "strata/deepseek41/artifact.hpp"

namespace strata::deepseek41 {
struct IndexKeyWeights {
    std::span<const uint16_t> projection; // original BF16, row-major [128,512]
    std::span<const uint16_t> norm;       // BF16 [128]
    int layer = 2; // original source identity; owner orchestration rejects mixed layers
};
// Fixed source layers 2, 8 and 14 only. Borrows the mapped partial model; caller keeps it alive.
IndexKeyWeights read_index_key_weights(const GgufModel& model, const Geometry& geometry, int layer);
// Source 20 owns the ratio-1 keys used through layer 39. Later index sources
// own queries, not these weights. Kept distinct from the ratio-2 reader above.
struct Source20IndexKeyWeights {
    std::span<const uint16_t> projection; // original BF16 row-major [128,512]
    std::span<const uint16_t> norm;       // original BF16 [128]
};
Source20IndexKeyWeights read_source20_index_key_weights(const GgufModel& model,
                                                       const Geometry& geometry);
} // namespace strata::deepseek41
