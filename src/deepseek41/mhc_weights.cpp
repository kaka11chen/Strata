#include "strata/deepseek41/mhc_weights.hpp"

namespace strata::deepseek41 {
MhcWeights read_mhc_weights(const GgufModel& model, const Geometry& g, int layer, MhcBranch branch) {
    const auto* profile = model.meta().get("strata.deepseek41.mhc_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != "1017-hc4-delayed")
        throw std::invalid_argument("Missing DeepSeek mHC profile 1017-hc4-delayed");
    if (g.hidden <= 0 || g.hidden > 5120 || g.hidden % 32 || g.layers <= 0 || g.layers > 40 ||
        layer < 0 || layer >= g.layers || (branch != MhcBranch::attention && branch != MhcBranch::ffn))
        throw std::invalid_argument("Invalid DeepSeek mHC geometry/layer/branch");
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const std::string kind = branch == MhcBranch::ffn ? "ffn" : "attn";
    auto read = [&](const std::string& name, uint32_t type, const std::vector<uint64_t>& shape) {
        size_t shard = 0;
        const auto* t = model.find(prefix + name, &shard);
        if (!t || t->type != type || t->shape != shape || !model.in_bounds(*t, shard))
            throw std::invalid_argument("Malformed DeepSeek mHC tensor " + prefix + name);
        const auto* data = model.shard(shard).tensor_data(*t);
        if (uintptr_t(data) % (type == 0 ? 4 : 2))
            throw std::invalid_argument("Misaligned DeepSeek mHC tensor " + prefix + name);
        return data;
    };
    const size_t n = size_t(g.hidden);
    return {{reinterpret_cast<const float*>(read("hc_" + kind + "_fn", 0, {4 * n, 24})), 96 * n},
            {reinterpret_cast<const float*>(read("hc_" + kind + "_scale", 0, {3})), 3},
            {reinterpret_cast<const float*>(read("hc_" + kind + "_base", 0, {24})), 24},
            {reinterpret_cast<const uint16_t*>(read(kind + "_norm.weight", 30, {n})), n}};
}
} // namespace strata::deepseek41
