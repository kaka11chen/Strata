#include "strata/deepseek41/shared_weights.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); }
void rejects(const char* name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
} // namespace
int main() {
    std::vector<uint8_t> values(32 * 32, 0), scales(1, 127);
    Fp8MatrixView m{values, scales, 32, 32};
    bool codes_ok = true;
    for (int code = 0; code < 256; ++code) {
        values[0] = uint8_t(code);
        if ((code & 127) == 127) {
            bool refused = false;
            try { fp8_block32_to_bf16(m); } catch (const std::invalid_argument&) { refused = true; }
            codes_ok = codes_ok && refused;
            continue;
        }
        // Independent reconstruction via significand/exponent, including signed zero.
        const int exponent = (code >> 3) & 15;
        const float magnitude = exponent == 0 ? float(code & 7) / 512.f
                                               : (1.f + float(code & 7) / 8.f) * std::pow(2.f, exponent - 7);
        const float expected = code & 128 ? -magnitude : magnitude;
        const auto actual = fp8_block32_to_bf16(m);
        codes_ok = codes_ok && kernels::f32_from_bf16(actual[0]) == expected &&
                   bool(actual[0] & 0x8000) == bool(code & 128);
    }
    check(codes_ok, "all 256 E4M3 codes: exact finite values/signed zero, both NaNs refused");
    values[0] = 0x38; // 1.0
    bool scales_ok = true;
    for (int scale = 0; scale < 255; ++scale) {
        scales[0] = uint8_t(scale);
        const auto actual = fp8_block32_to_bf16(m);
        scales_ok = scales_ok && double(kernels::f32_from_bf16(actual[0])) == std::pow(2., scale - 127);
    }
    check(scales_ok, "all 255 finite E8M0 powers expand exactly for unit weight");
    scales[0] = 255;
    rejects("E8M0 NaN refused", [&] { fp8_block32_to_bf16(m); });
    scales[0] = 0; values[0] = 1;
    rejects("BF16 subnormal precision loss refused", [&] { fp8_block32_to_bf16(m); });
    scales[0] = 254; values[0] = 126;
    rejects("BF16 overflow refused", [&] { fp8_block32_to_bf16(m); });
    scales[0] = 127;
    auto broken = m; broken.cols = 31;
    rejects("non-block dimensions refused", [&] { fp8_block32_to_bf16(broken); });
    broken = m; broken.scales = {};
    rejects("missing scales refused", [&] { fp8_block32_to_bf16(broken); });
    broken = m; broken.values = std::span(values).first(100);
    rejects("short weight matrix refused", [&] { fp8_block32_to_bf16(broken); });
    // Distinct scales in both row and column axes catches accidental per-row FP8 decoding.
    values.assign(64 * 64, 0x38); scales = {126, 127, 128, 129};
    const auto block = fp8_block32_to_bf16({values, scales, 64, 64});
    bool blocks_ok = true;
    for (int row = 0; row < 64; ++row)
        for (int col = 0; col < 64; ++col)
            blocks_ok = blocks_ok && kernels::f32_from_bf16(block[size_t(row) * 64 + col]) ==
                                    std::ldexp(1.f, int(scales[size_t(row / 32) * 2 + col / 32]) - 127);
    check(blocks_ok, "32x32 scale addressing covers both row and column boundaries");
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
