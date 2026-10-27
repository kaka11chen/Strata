#pragma once

#include "strata/deepseek41/artifact.hpp"

namespace strata::deepseek41 {
inline constexpr const char* text_io_profile = "1017-bf16-hc4-last-logits";
inline constexpr const char* text_io_revision = "1bae0e1088993b9b82de08f59de5be5a34e42170";
struct TextIoWeights {
    int hidden = 0, vocab_size = 0;
    std::span<const uint16_t> embedding; // [vocab,hidden], original BF16, untied
    std::span<const uint16_t> norm;      // [hidden], original BF16
    std::span<const uint16_t> head;      // [vocab,hidden], original BF16
};
// Views borrow the mapped model. Default dimensions are the pinned checkpoint;
// smaller dimensions are available for analytical controls, never inferred from a file.
TextIoWeights read_text_io_weights(const GgufModel& model, int hidden = 5120, int vocab_size = 129280);
} // namespace strata::deepseek41
