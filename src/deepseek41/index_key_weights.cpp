#include "strata/deepseek41/index_key_weights.hpp"

namespace strata::deepseek41 {
namespace {
IndexKeyWeights read_weights(const GgufModel& model, int layer) {
    auto read = [&](const std::string& name, const std::vector<uint64_t>& shape, size_t count) {
        size_t shard = 0;
        const auto* t = model.find("blk." + std::to_string(layer) + ".attn_indexer_" + name + ".weight", &shard);
        if (!t || t->type != 30 || t->shape != shape || !model.in_bounds(*t, shard))
            throw std::invalid_argument("Malformed DeepSeek index key tensor " + name);
        const auto* data = model.shard(shard).tensor_data(*t);
        if (uintptr_t(data) % 2) throw std::invalid_argument("Misaligned DeepSeek index key tensor " + name);
        return std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(data), count);
    };
    return {read("wk", {512, 128}, 512 * 128), read("k_norm", {128}, 128), layer};
}
} // namespace
IndexKeyWeights read_index_key_weights(const GgufModel& model, const Geometry& g, int layer) {
    const std::string expected = "1017-layer" + std::to_string(layer) + "-index-key128-bf16";
    const auto* profile = model.meta().get("strata.deepseek41.index_key_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != expected)
        throw std::invalid_argument("Missing or mismatched DeepSeek source profile");
    if ((layer != 2 && layer != 8 && layer != 14) || g.layers != 40 || g.hidden != 5120 ||
        g.compress_ratios.size() != 40 || g.compress_ratios[size_t(layer)] != 2)
        throw std::invalid_argument("Only fixed 1017 source-2/source-8/source-14 index keys are supported");
    if ((layer == 8 || layer == 14) && (g.sources.size() != 40 || g.sources[size_t(layer)].kv != layer ||
        g.sources[size_t(layer)].index != layer || g.sources[size_t(layer)].candidates != -1))
        throw std::invalid_argument("The original source layer must own its KV and index without candidate blocks");
    return read_weights(model, layer);
}
Source20IndexKeyWeights read_source20_index_key_weights(const GgufModel& model, const Geometry& g) {
    const auto* profile = model.meta().get("strata.deepseek41.index_key_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != "1017-layer20-index-key128-bf16")
        throw std::invalid_argument("Missing or mismatched DeepSeek source-20 index key profile");
    if (g.layers != 40 || g.hidden != 5120 || g.compress_ratios.size() != 40 ||
        g.compress_ratios[20] != 1 || g.sources.size() != 40 || g.sources[20].kv != 20 ||
        g.sources[20].index != 20 || g.sources[20].candidates != -1)
        throw std::invalid_argument("Source 20 must own ratio-1 KV/index and publish candidates without consuming them");
    const auto weights = read_weights(model, 20);
    return {weights.projection, weights.norm};
}
} // namespace strata::deepseek41
