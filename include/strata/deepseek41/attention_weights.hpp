#pragma once

#include "strata/deepseek41/shared_weights.hpp"

namespace strata::deepseek41 {
// Fixed 1017 layer-0..20 projection layouts, with explicit layer profiles.
// No inferred geometry or format conversion.
struct AttentionWeights {
    std::array<Fp8MatrixView, 5> projections; // wq_a, wq_b, wkv, wo_a, wo_b
    std::span<const uint16_t> q_norm, kv_norm;
    std::span<const float> sink;
    int layer = 0; // must match the explicit consumer profile
};
// Borrows the mapped model; caller keeps it alive. This is a partial-layer loader.
AttentionWeights read_attention_weights(const GgufModel& model, const Geometry& geometry, int layer);
} // namespace strata::deepseek41
