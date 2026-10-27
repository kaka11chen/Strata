#include "strata/deepseek41/compressor_weights.hpp"

namespace strata::deepseek41 {
CompressorWeights read_compressor_weights(const GgufModel& model, const Geometry& g, int layer) {
    const std::string expected = "1017-layer" + std::to_string(layer) + "-ratio2-bf16";
    const auto* profile = model.meta().get("strata.deepseek41.compressor_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != expected)
        throw std::invalid_argument("Missing or mismatched DeepSeek source profile");
    if ((layer != 2 && layer != 8 && layer != 14) || g.layers != 40 || g.hidden != 5120 ||
        g.compress_ratios.size() != 40 || g.compress_ratios[size_t(layer)] != 2)
        throw std::invalid_argument("Only fixed 1017 source-2/source-8/source-14 ratio-2 compression is supported");
    if ((layer == 8 || layer == 14) && (g.sources.size() != 40 || g.sources[size_t(layer)].kv != layer ||
        g.sources[size_t(layer)].index != layer || g.sources[size_t(layer)].candidates != -1))
        throw std::invalid_argument("The original source layer must own its KV and index without candidate blocks");
    auto read = [&](const std::string& name, const std::vector<uint64_t>& shape, size_t count) {
        size_t shard = 0;
        const auto* t = model.find("blk." + std::to_string(layer) + ".attn_compressor_" + name + ".weight", &shard);
        if (!t || t->type != 30 || t->shape != shape || !model.in_bounds(*t, shard))
            throw std::invalid_argument("Malformed DeepSeek compressor tensor " + name);
        const auto* data = model.shard(shard).tensor_data(*t);
        if (uintptr_t(data) % 2) throw std::invalid_argument("Misaligned DeepSeek compressor tensor " + name);
        return std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(data), count);
    };
    return {read("wkv", {5120, 512}, 5120 * 512), read("wgate", {5120, 512}, 5120 * 512),
            read("norm", {512}, 512), layer};
}
} // namespace strata::deepseek41
