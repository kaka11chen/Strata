#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace strata::deepseek41 {
struct Route {
    std::array<int32_t, 6> ids{};
    std::array<float, 6> weights{};
};
// 1017 text router: sqrt(softplus(logit)), temperature 1. Bias selects only;
// normalize the six unbiased scores with +1e-20, then scale by 1.5.
// Exactly 384 finite logits/biases. Ties select the lower ID (torch ties are unspecified).
Route route_top6(std::span<const float> logits, std::span<const float> bias);
} // namespace strata::deepseek41
