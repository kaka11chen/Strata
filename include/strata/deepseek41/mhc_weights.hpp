#pragma once

#include "strata/deepseek41/artifact.hpp"

namespace strata::deepseek41 {
enum class MhcBranch { attention, ffn };
struct MhcWeights {
    std::span<const float> projection; // [24, 4*hidden], F32, row-major
    std::span<const float> scale;      // [3], F32
    std::span<const float> base;       // [24], F32
    std::span<const uint16_t> norm;    // [hidden], BF16
};
// Fixed 1017 profile: four streams, 20 Sinkhorn iterations, norm eps=1e-20,
// HC eps=1e-6 and delayed pre coefficients. Views borrow the mapped model.
MhcWeights read_mhc_weights(const GgufModel& model, const Geometry& geometry, int layer, MhcBranch branch);
} // namespace strata::deepseek41
