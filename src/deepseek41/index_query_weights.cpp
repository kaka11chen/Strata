#include "strata/deepseek41/index_query_weights.hpp"

namespace strata::deepseek41 {
namespace {
IndexQueryWeights read_weights(const GgufModel& model, int layer) {
    auto read = [&](const std::string& name, uint32_t type, const std::vector<uint64_t>& shape) {
        size_t shard = 0;
        const auto* t = model.find("blk." + std::to_string(layer) + ".attn_indexer_" + name, &shard);
        if (!t || t->type != type || t->shape != shape || !model.in_bounds(*t, shard))
            throw std::invalid_argument("Malformed DeepSeek index query tensor " + name);
        const auto* data = model.shard(shard).tensor_data(*t);
        if (type == 30 && uintptr_t(data) % 2) throw std::invalid_argument("Misaligned index head weights");
        return data;
    };
    return {{{read("wq_b.weight", 24, {1280, 4096}), 1280 * 4096},
             {read("wq_b.scale", 24, {40, 128}), 40 * 128}, 4096, 1280},
            {reinterpret_cast<const uint16_t*>(read("weights_proj.weight", 30, {5120, 32})), 5120 * 32}, layer};
}
} // namespace
IndexQueryWeights read_index_query_weights(const GgufModel& model, const Geometry& g, int layer) {
    const std::string expected = "1017-layer" + std::to_string(layer) + "-index32x128-fp8-block32";
    const auto* profile = model.meta().get("strata.deepseek41.index_query_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != expected)
        throw std::invalid_argument("Missing or mismatched DeepSeek source profile");
    if ((layer != 2 && layer != 8 && layer != 14) || g.layers != 40 || g.hidden != 5120 ||
        g.compress_ratios.size() != 40 || g.compress_ratios[size_t(layer)] != 2)
        throw std::invalid_argument("Only fixed 1017 source-2/source-8/source-14 index queries are supported");
    if ((layer == 8 || layer == 14) && (g.sources.size() != 40 || g.sources[size_t(layer)].kv != layer ||
        g.sources[size_t(layer)].index != layer || g.sources[size_t(layer)].candidates != -1))
        throw std::invalid_argument("The original source layer must own its KV and index without candidate blocks");
    return read_weights(model, layer);
}
Source20IndexQueryWeights read_source20_index_query_weights(const GgufModel& model, const Geometry& g) {
    const auto* profile = model.meta().get("strata.deepseek41.index_query_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != "1017-layer20-index32x128-fp8-block32")
        throw std::invalid_argument("Missing or mismatched DeepSeek source-20 index query profile");
    if (g.layers != 40 || g.hidden != 5120 || g.compress_ratios.size() != 40 ||
        g.compress_ratios[20] != 1 || g.sources.size() != 40 || g.sources[20].kv != 20 ||
        g.sources[20].index != 20 || g.sources[20].candidates != -1)
        throw std::invalid_argument("Source 20 must own ratio-1 KV/index and publish candidates without consuming them");
    const auto weights = read_weights(model, 20);
    return {weights.query, weights.head_weights};
}
CandidateIndexQueryWeights read_candidate_index_query_weights(const GgufModel& model, const Geometry& g, int layer) {
    const auto* profile = model.meta().get("strata.deepseek41.index_query_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != "1017-layer" + std::to_string(layer) + "-index32x128-fp8-block32")
        throw std::invalid_argument("Missing or mismatched candidate index query profile");
    if ((layer != 24 && layer != 28 && layer != 32 && layer != 36) || g.layers != 40 || g.hidden != 5120 || g.compress_ratios.size() != 40 ||
        g.compress_ratios[20] != 1 || g.compress_ratios[size_t(layer)] != 1 || g.sources.size() != 40 ||
        g.sources[20].kv != 20 || g.sources[20].index != 20 || g.sources[20].candidates != -1 ||
        g.sources[size_t(layer)].kv != 20 || g.sources[size_t(layer)].index != layer || g.sources[size_t(layer)].candidates != 20)
        throw std::invalid_argument("Index sources 24/28/32/36 must borrow source-20 keys and candidates");
    const auto weights = read_weights(model, layer);
    return {weights.query, weights.head_weights, layer};
}
} // namespace strata::deepseek41
