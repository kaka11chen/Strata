#pragma once

#include "strata/deepseek41/artifact.hpp"

namespace strata::deepseek41 {
// Original BF16 checkpoint weights. Ratio-2 projection math promotes them to F32.
struct CompressorWeights {
    std::span<const uint16_t> kv, gate; // row-major [512,5120]
    std::span<const uint16_t> norm;     // [512]
    int layer = 2; // original source identity; owner orchestration rejects mixed layers
};
// Fixed source layers 2, 8 and 14 only. Borrows the mapped partial model; caller keeps it alive.
CompressorWeights read_compressor_weights(const GgufModel& model, const Geometry& geometry, int layer);
} // namespace strata::deepseek41
