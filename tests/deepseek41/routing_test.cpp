#include "strata/deepseek41/routing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

using namespace strata::deepseek41;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) {
    ++checks;
    failures += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void rejects(const std::string& name, const std::function<void()>& run) {
    try { run(); check(false, name); }
    catch (const std::invalid_argument&) { check(true, name); }
}
// FP64 reference: full stable sort and numerically stable softplus, separate
// from the production FP32 threshold/partial-sort implementation.
bool reference(const std::array<float, 384>& logits, const std::array<float, 384>& bias) {
    std::array<double, 384> scores{};
    std::array<int, 384> order{};
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < 384; ++i) {
        const double x = logits[i];
        scores[i] = std::sqrt(std::max(x, 0.) + std::log1p(std::exp(-std::abs(x))));
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] + bias[a] > scores[b] + bias[b];
    });
    const auto actual = route_top6(logits, bias);
    double sum = 1e-20;
    for (int k = 0; k < 6; ++k) sum += scores[order[k]];
    for (int k = 0; k < 6; ++k)
        if (actual.ids[k] != order[k] || std::abs(actual.weights[k] - scores[order[k]] / sum * 1.5) > 2e-6)
            return false;
    return true;
}
} // namespace
int main() {
    std::array<float, 384> logits{}, bias{};
    const auto tied = route_top6(logits, bias);
    check(tied.ids == std::array<int32_t, 6>{0, 1, 2, 3, 4, 5}, "equal scores have deterministic lower-ID ties");
    check(std::all_of(tied.weights.begin(), tied.weights.end(), [](float x) { return std::abs(x - .25f) <= 3e-8f; }),
          "equal-score weights are 0.25 within one FP32 rounding step");
    check(std::abs(std::accumulate(tied.weights.begin(), tied.weights.end(), 0.f) - 1.5f) <= 2e-7f,
          "normalized weights sum to 1.5 within FP32 rounding");
    bias[383] = 100.f;
    const auto corrected = route_top6(logits, bias);
    check(corrected.ids[0] == 383 && corrected.weights == tied.weights, "bias selects but never scales weights");
    bias.fill(0);
    for (int i = 0; i < 6; ++i) { logits[i] = float(i - 4); bias[i] = 100.f; }
    check(reference(logits, bias), "selected raw scores are normalized, not softmax or corrected scores");
    logits.fill(-1000.f);
    const auto underflow = route_top6(logits, bias);
    check(std::all_of(underflow.weights.begin(), underflow.weights.end(), [](float x) { return x == 0.f; }),
          "underflow scores use epsilon without NaN");
    logits.fill(1000.f);
    check(reference(logits, bias), "large logits avoid softplus overflow");
    std::mt19937 rng(1017);
    std::uniform_real_distribution<float> value(-15.f, 25.f), offset(-.5f, .5f);
    bool random_ok = true;
    for (int trial = 0; trial < 200; ++trial) {
        for (int i = 0; i < 384; ++i) { logits[i] = value(rng); bias[i] = offset(rng); }
        random_ok = random_ok && reference(logits, bias);
    }
    check(random_ok, "200 randomized routers match independent FP64 IDs and weights");
    rejects("short logits rejected", [&] { route_top6(std::span(logits).first(383), bias); });
    logits[0] = std::numeric_limits<float>::quiet_NaN();
    rejects("NaN logit rejected", [&] { route_top6(logits, bias); });
    logits[0] = std::numeric_limits<float>::infinity();
    rejects("infinite logit rejected", [&] { route_top6(logits, bias); });
    logits[0] = 0.f; bias[0] = std::numeric_limits<float>::infinity();
    rejects("infinite correction bias rejected", [&] { route_top6(logits, bias); });
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
