#pragma once

#include "strata/deepseek41/artifact.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace strata::deepseek41 {

// Lossless storage adapter for Strata's existing NVFP4 GPU projections.
// E8M0 bytes118..135 retain E2M1 codes and repeat each32-value scale for two16-value sub-blocks.
// Scale117 (2^-10) is also exact when every code can be halved: raise its scale
// to118 and remap codes, retaining signed zero. +/-0.5 and +/-1.5 are rejected.
// Other scales are rejected, never rounded. No global scale or blob layout change.
// Input must contain whole64-value pairs.
void mxfp4_to_nvfp4(std::span<const uint8_t> source, std::span<uint8_t> destination);
std::vector<uint8_t> mxfp4_to_nvfp4(std::span<const uint8_t> source);

// Strata's standard [gate | up | down | s_gate,s_up,s_down,0] NVFP4 blob.
// A projection keeps its old payload/scale1 when the direct adapter is exact.
// Otherwise scale117 codes may use exponent+1 and global0.5, provided every
// block is in117..134. Other inexact ranges are rejected. Apply the global
// scale to FP32 projection outputs BEFORE any BF16 rounding or activation.
inline constexpr size_t nvfp4_scale_tail_bytes = 4 * sizeof(float);
std::array<float, 3> nvfp4_expert_scales(std::span<const uint8_t> packed);
size_t nvfp4_expert_bytes(const Geometry& geometry);
void pack_nvfp4_expert(const Geometry& geometry, const ExpertView& weights, std::span<uint8_t> destination);

} // namespace strata::deepseek41
