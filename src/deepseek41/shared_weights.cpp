#include "strata/deepseek41/shared_weights.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cmath>

namespace strata::deepseek41 {
std::vector<uint16_t> fp8_block32_to_bf16(const Fp8MatrixView& m) {
    if (m.rows <= 0 || m.cols <= 0 || m.rows > 32768 || m.cols > 8192 || size_t(m.rows) * size_t(m.cols) > 41943040 || m.rows % 32 || m.cols % 32 ||
        !m.values.data() || !m.scales.data() || m.values.size() != size_t(m.rows) * size_t(m.cols) ||
        m.scales.size() != size_t(m.rows / 32) * size_t(m.cols / 32))
        throw std::invalid_argument("DeepSeek FP8 matrix requires dimension-matched 32x32 blocks within 1017 matrix limits");
    for (uint8_t scale : m.scales)
        if (scale == 255)
            throw std::invalid_argument("Nonfinite DeepSeek E8M0 scale");
    std::vector<uint16_t> out(m.values.size());
    for (int row = 0; row < m.rows; ++row)
        for (int col = 0; col < m.cols; ++col) {
            const size_t at = size_t(row) * m.cols + col;
            const uint8_t code = m.values[at], magnitude = code & 127;
            if (magnitude == 127)
                throw std::invalid_argument("Nonfinite DeepSeek E4M3 weight");
            const int exp = magnitude >> 3, fraction = magnitude & 7;
            const double value = exp == 0 ? std::ldexp(double(fraction), -9)
                                         : std::ldexp(double(8 + fraction), exp - 10);
            const int power = int(m.scales[size_t(row / 32) * (m.cols / 32) + col / 32]) - 127;
            const double exact = std::ldexp(code & 128 ? -value : value, power);
            const uint16_t bits = kernels::bf16_from_f32(float(exact));
            if (!std::isfinite(exact) || double(kernels::f32_from_bf16(bits)) != exact)
                throw std::invalid_argument("DeepSeek FP8 weight is not exactly representable in BF16");
            out[at] = bits;
        }
    return out;
}

SharedExpertWeights read_shared_expert(const GgufModel& model, const Geometry& g, int layer) {
    const auto* encoding = model.meta().get("strata.deepseek41.shared_weight_encoding");
    if (!encoding || encoding->type != MetaType::STRING || encoding->s != "e4m3fn-e8m0-block32")
        throw std::invalid_argument("Missing DeepSeek shared weight encoding e4m3fn-e8m0-block32");
    if (layer < 0 || layer >= g.layers || g.hidden <= 0 || g.hidden > 5120 || g.hidden % 32 ||
        g.expert_width <= 0 || g.expert_width > 2304 || g.expert_width % 32)
        throw std::invalid_argument("Invalid DeepSeek shared expert geometry/layer");
    SharedExpertWeights result;
    for (int role = 0; role < 3; ++role) {
        const char* name = role == 0 ? "gate" : role == 1 ? "up" : "down";
        const std::string base = "blk." + std::to_string(layer) + ".ffn_" + name + "_shexp.";
        auto& part = result.parts[size_t(role)];
        part.rows = role == 2 ? g.hidden : g.expert_width;
        part.cols = role == 2 ? g.expert_width : g.hidden;
        auto read = [&](const std::string& suffix, int rows, int cols) {
            size_t shard = 0;
            const auto* t = model.find(base + suffix, &shard);
            if (!t || t->type != 24 || t->shape != std::vector<uint64_t>{uint64_t(cols), uint64_t(rows)} ||
                !model.in_bounds(*t, shard))
                throw std::invalid_argument("Malformed DeepSeek shared tensor " + base + suffix);
            return std::span<const uint8_t>(model.shard(shard).tensor_data(*t), size_t(rows) * cols);
        };
        part.values = read("weight", part.rows, part.cols);
        part.scales = read("scale", part.rows / 32, part.cols / 32);
    }
    return result;
}
} // namespace strata::deepseek41
