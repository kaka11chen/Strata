#pragma once

#include "strata/deepseek41/shared_weights.hpp"

namespace strata::deepseek41 {
struct EngramWeights {
    Fp8MatrixView wkv; // [25600,6144], E4M3FN + E8M0 per 32x32 block
    std::span<const uint16_t> q, k; // BF16 [4,5120]
    int layer = 1; // original 1017 layer ID; trailing default keeps layer-1 aggregates compatible
};
// Fixed 1017 layer 1 or 14. Borrowed views: model mappings must outlive them.
// Expected layer, profile and original tensor names must agree.
EngramWeights read_engram_weights(const GgufModel& model, int layer);
using Layer1EngramWeights = EngramWeights;
inline Layer1EngramWeights read_layer1_engram_weights(const GgufModel& model) {
    return read_engram_weights(model, 1);
}
} // namespace strata::deepseek41
