#pragma once

#include "strata/deepseek41/artifact.hpp"

namespace strata::deepseek41 {
struct Fp8MatrixView {
    std::span<const uint8_t> values; // E4M3FN, row-major [rows, cols]
    std::span<const uint8_t> scales; // E8M0, row-major [rows/32, cols/32]
    int rows = 0, cols = 0;
};
struct SharedExpertWeights {
    std::array<Fp8MatrixView, 3> parts; // gate, up, down
};
// Lossless expansion for Strata's existing BF16 GEMV. Rejects NaNs, overflow
// and values below BF16 precision; never silently rounds a checkpoint weight.
// Whole 32x32 blocks, at most 32768 rows, 8192 columns and 41943040 elements
// (the largest fixed 1017 attention projection). Callers validate exact shapes.
std::vector<uint16_t> fp8_block32_to_bf16(const Fp8MatrixView& matrix);
// Raw I8 tensors with an explicit e4m3fn-e8m0-block32 encoding tag; no format guessing.
// Mapped model must outlive the returned views. This is not a complete-model loader.
SharedExpertWeights read_shared_expert(const GgufModel& model, const Geometry& geometry, int layer);
} // namespace strata::deepseek41
