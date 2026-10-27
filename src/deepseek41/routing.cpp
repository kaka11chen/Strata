#include "strata/deepseek41/routing.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace strata::deepseek41 {
Route route_top6(std::span<const float> logits, std::span<const float> bias) {
    if (logits.size() != 384 || bias.size() != 384 || !logits.data() || !bias.data())
        throw std::invalid_argument("DeepSeek 1017 router requires 384 logits and biases");
    std::array<float, 384> scores{}, ranked{};
    std::array<int32_t, 384> order{};
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(logits[i]) || !std::isfinite(bias[i]))
            throw std::invalid_argument("DeepSeek router requires finite logits and bias");
        // Match torch softplus's threshold=20, avoiding exp overflow.
        scores[i] = std::sqrt(logits[i] > 20.f ? logits[i] : std::log1p(std::exp(logits[i])));
        ranked[i] = scores[i] + bias[i];
        if (!std::isfinite(ranked[i]))
            throw std::invalid_argument("DeepSeek router selection score overflow");
    }
    std::partial_sort(order.begin(), order.begin() + 6, order.end(), [&](int32_t a, int32_t b) {
        return ranked[a] > ranked[b] || (ranked[a] == ranked[b] && a < b);
    });
    Route route;
    float sum = 0.f;
    for (size_t k = 0; k < 6; ++k) {
        route.ids[k] = order[k];
        sum += scores[order[k]];
    }
    for (size_t k = 0; k < 6; ++k)
        route.weights[k] = (scores[order[k]] / (sum + 1e-20f)) * 1.5f;
    return route;
}
} // namespace strata::deepseek41
