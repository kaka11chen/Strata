#pragma once

namespace strata::deepseek41::detail {
// Shared model-specific elementwise glue; matrix kernels remain in Strata.
// Existing four-stream BF16-valued residual collapse; incoming coefficients are F32.
void collapse_mhc_values(const float* residual, const float* incoming, float* output, int hidden, void* stream);
void round_bf16_values(const float* input, float* output, int n, void* stream);
void swiglu_1017_values(const float* gate, const float* up, float route, float* hidden, int n, void* stream);
} // namespace strata::deepseek41::detail
