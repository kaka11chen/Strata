#include "strata/deepseek41/fp4_layout.hpp"

#include <array>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek41 {
namespace {
void pack_blocks(std::span<const uint8_t> source, std::span<uint8_t> result, int shift) {
    if (source.empty() || !source.data() || source.size() % 34)
        throw std::invalid_argument("MXFP4 to NVFP4 requires whole 64-value pairs");
    if (!result.data() || result.size() != source.size() / 34 * 36)
        throw std::invalid_argument("NVFP4 destination has the wrong size");
    for (size_t b = 0; b < source.size() / 34; ++b) {
        auto* dst = result.data() + b * 36;
        for (size_t h = 0; h < 2; ++h) {
            const auto* src = source.data() + b * 34 + h * 17;
            int exponent = int(src[0]) - 127 + shift;
            const uint8_t* values = src + 1;
            std::array<uint8_t, 16> normalized;
            if (exponent == -10) {
                // Original layer39 has scale117 blocks with exactly halvable
                // E2M1 values. Raise the scale once and preserve every weight;
                // +/-0.5 and +/-1.5 cannot be halved exactly and must fail.
                constexpr std::array<uint8_t, 16> half{0, 255, 1, 255, 2, 3, 4, 5, 8, 255, 9, 255, 10, 11, 12, 13};
                for (size_t j = 0; j < normalized.size(); ++j) {
                    const auto low = half[values[j] & 15], high = half[values[j] >> 4];
                    if (low == 255 || high == 255)
                        throw std::invalid_argument("MXFP4 scale117 contains a code that cannot be halved exactly");
                    normalized[j] = uint8_t(low | (high << 4));
                }
                values = normalized.data();
                exponent = -9;
            }
            if (exponent < -9 || exponent > 8)
                throw std::invalid_argument("MXFP4 scale cannot be represented exactly by this NVFP4 adapter");
            const auto scale = uint8_t(exponent < -6 ? 1 << (exponent + 9) : (exponent + 7) << 3);
            dst[2 * h] = dst[2 * h + 1] = scale;
            for (size_t j = 0; j < 8; ++j) {
                dst[4 + h * 16 + j] = (values[j] & 15) | ((values[8 + j] & 15) << 4);
                dst[4 + h * 16 + 8 + j] = (values[j] >> 4) | (values[8 + j] & 240);
            }
        }
    }
}
// Decide before writing: the old exact remap remains preferred. A whole
// projection shift is needed only for117 codes0.5/1.5 (including signs).
int projection_shift(std::span<const uint8_t> source) {
    bool needs_half = false, has135 = false;
    for (size_t b = 0; b < source.size(); b += 17) {
        const uint8_t scale = source[b];
        if (scale < 117 || scale > 135)
            throw std::invalid_argument("MXFP4 projection scale is outside the exact adapter range");
        has135 |= scale == 135;
        if (scale == 117) for (size_t j = 1; j <= 16; ++j) {
            const int low = source[b+j] & 7, high = (source[b+j] >> 4) & 7;
            needs_half |= low == 1 || low == 3 || high == 1 || high == 3;
        }
    }
    if (needs_half && has135)
        throw std::invalid_argument("MXFP4 projection has no exact global scale for both117 and135");
    return needs_half ? 1 : 0;
}
} // namespace
void mxfp4_to_nvfp4(std::span<const uint8_t> source, std::span<uint8_t> result) {
    pack_blocks(source, result, 0);
}
std::vector<uint8_t> mxfp4_to_nvfp4(std::span<const uint8_t> source) {
    if (source.empty() || !source.data() || source.size() % 34)
        throw std::invalid_argument("MXFP4 to NVFP4 requires whole 64-value pairs");
    std::vector<uint8_t> result(source.size() / 34 * 36);
    mxfp4_to_nvfp4(source, result);
    return result;
}
size_t nvfp4_expert_bytes(const Geometry& g) {
    if (g.hidden <= 0 || g.hidden > 5120 || g.expert_width <= 0 || g.expert_width > 2304 || g.hidden % 64 ||
        g.expert_width % 64)
        throw std::invalid_argument("DeepSeek CUDA expert dimensions must be whole 64-value blocks up to 5120/2304");
    return size_t(g.hidden) * size_t(g.expert_width) / 64 * 36 * 3 + nvfp4_scale_tail_bytes;
}
void pack_nvfp4_expert(const Geometry& g, const ExpertView& weights, std::span<uint8_t> destination) {
    const size_t bytes = nvfp4_expert_bytes(g), part_bytes = (bytes - nvfp4_scale_tail_bytes) / 3;
    if (!destination.data() || destination.size() != bytes)
        throw std::invalid_argument("DeepSeek NVFP4 blob has the wrong size");
    for (const auto& part : weights.parts)
        if (part.type != 39 || !part.data.data() || part.data.size() != part_bytes / 36 * 34)
            throw std::invalid_argument("DeepSeek CUDA expert requires dimension-matched MXFP4 weights");
    std::array<int, 3> shifts;
    for (size_t role = 0; role < 3; ++role) shifts[role] = projection_shift(weights.parts[role].data);
    std::array<float, 4> scales{1.f, 1.f, 1.f, 0.f};
    for (size_t role = 0; role < 3; ++role) {
        pack_blocks(weights.parts[role].data, destination.subspan(role * part_bytes, part_bytes), shifts[role]);
        if (shifts[role]) scales[role] = .5f;
    }
    std::memcpy(destination.data() + 3 * part_bytes, scales.data(), nvfp4_scale_tail_bytes);
}
std::array<float, 3> nvfp4_expert_scales(std::span<const uint8_t> packed) {
    if (!packed.data() || packed.size() < nvfp4_scale_tail_bytes)
        throw std::invalid_argument("Missing DeepSeek NVFP4 scale tail");
    std::array<float, 4> tail;
    std::memcpy(tail.data(), packed.data() + packed.size() - nvfp4_scale_tail_bytes, nvfp4_scale_tail_bytes);
    for (size_t role = 0; role < 3; ++role)
        if (tail[role] != 1.f && tail[role] != .5f)
            throw std::invalid_argument("Unsupported DeepSeek NVFP4 global scale");
    if (tail[3] != 0.f) throw std::invalid_argument("Invalid DeepSeek NVFP4 reserved scale field");
    return {tail[0], tail[1], tail[2]};
}
} // namespace strata::deepseek41
