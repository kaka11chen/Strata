#pragma once
#include "strata/deepseek41/text_io_weights.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cmath>

// Test-only CPU controls: BF16 storage, FP64 sums and independently indexed rows.
namespace text_io_reference {
using strata::deepseek41::TextIoWeights;
namespace kernels = strata::kernels;
inline float round_bf16(float value) { return kernels::f32_from_bf16(kernels::bf16_from_f32(value)); }
inline std::vector<float> embedding(const TextIoWeights& w, std::span<const int32_t> tokens) {
    std::vector<float> result(tokens.size() * 4 * size_t(w.hidden));
    for (size_t t = 0; t < tokens.size(); ++t)
        for (size_t stream = 0; stream < 4; ++stream)
            for (int d = 0; d < w.hidden; ++d)
                result[(t * 4 + stream) * w.hidden + size_t(d)] = kernels::f32_from_bf16(w.embedding[size_t(tokens[t]) * w.hidden + size_t(d)]);
    return result;
}
inline std::vector<float> head(const TextIoWeights& w, const std::vector<float>& residual, const std::vector<float>& pre) {
    const size_t count = pre.size() / 4, h = size_t(w.hidden); std::vector<float> normalized(h);
    double square = 0.;
    for (size_t d = 0; d < h; ++d) {
        double mixed = 0.;
        for (size_t s = 0; s < 4; ++s) mixed += double(pre[(count - 1) * 4 + s]) * round_bf16(residual[((count - 1) * 4 + s) * h + d]);
        normalized[d] = round_bf16(float(mixed)); square += double(normalized[d]) * normalized[d];
    }
    const float scale = 1.f / std::sqrt(float(square / double(h)) + 1e-20f);
    for (size_t d = 0; d < h; ++d) normalized[d] = round_bf16((normalized[d] * scale) * kernels::f32_from_bf16(w.norm[d]));
    std::vector<float> logits(size_t(w.vocab_size), 0.f);
    for (int v = 0; v < w.vocab_size; ++v) {
        double sum = 0.;
        for (size_t d = 0; d < h; ++d) sum += double(normalized[d]) * kernels::f32_from_bf16(w.head[size_t(v) * h + d]);
        logits[size_t(v)] = float(sum);
    }
    return logits;
}
} // namespace text_io_reference
