#include "strata/deepseek41/attention_weights.hpp"

namespace strata::deepseek41 {
AttentionWeights read_attention_weights(const GgufModel& model, const Geometry& g, int layer) {
    if (layer < 0 || layer > 39)
        throw std::invalid_argument("Only fixed 1017 layer-0..39 attention weights are supported");
    const int index_source = layer / 4 * 4;
    const std::string expected = (layer == 24 || layer == 28 || layer == 32 || layer == 36) ? "1017-layer" + std::to_string(layer) + "-candidates20-fp8-block32" : layer == 20 ? "1017-layer20-ratio1-fp8-block32" :
                           layer == 0 ? "1017-layer0-swa128-fp8-block32" :
                           layer == 1 ? "1017-layer1-swa128-fp8-block32" :
                           (layer == 2 || layer == 8 || layer == 14) ? "1017-layer" + std::to_string(layer) + "-ratio2-fp8-block32" : "1017-layer" + std::to_string(layer) + "-shared" + std::to_string(layer < 8 ? 2 : layer < 14 ? 8 : layer < 20 ? 14 : layer < 24 ? 20 : index_source) + "-fp8-block32";
    const auto* profile = model.meta().get("strata.deepseek41.attention_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != expected)
        throw std::invalid_argument(std::string("Missing DeepSeek attention profile ") + expected);
    if (g.hidden != 5120 || (layer == 0 ?
        (g.layers <= 0 || g.layers > 40 || g.compress_ratios.empty() || g.compress_ratios[0] != 0) :
        (g.layers != 40 || g.compress_ratios.size() != 40 || g.compress_ratios[size_t(layer)] != (layer == 1 ? 0 : layer >= 20 ? 1 : 2))))
        throw std::invalid_argument("Attention weights do not match the fixed 1017 layer geometry");
    const int source = layer < 8 ? 2 : layer < 14 ? 8 : layer < 20 ? 14 : 20;
    if (layer >= 3 && layer < 24 && layer != 8 && layer != 14 && layer != 20 && (g.sources.size() != 40 || g.compress_ratios[size_t(source)] != (source == 20 ? 1 : 2) ||
        g.sources[size_t(source)].kv != source || g.sources[size_t(source)].index != source || g.sources[size_t(source)].candidates != -1 ||
        g.sources[size_t(layer)].kv != source || g.sources[size_t(layer)].index != source || g.sources[size_t(layer)].candidates != -1))
        throw std::invalid_argument("Shared consumers must reuse their original source's KV/index without candidate blocks");
    if ((layer == 8 || layer == 14 || layer == 20) && (g.sources.size() != 40 || g.sources[size_t(layer)].kv != layer ||
        g.sources[size_t(layer)].index != layer || g.sources[size_t(layer)].candidates != -1))
        throw std::invalid_argument("The original source layer must own its KV and index without consuming candidate blocks");
    if (layer >= 24 && (g.sources.size() != 40 || g.sources[20].kv != 20 || g.sources[20].index != 20 ||
        g.sources[20].candidates != -1 || g.compress_ratios[20] != 1 || g.sources[size_t(index_source)].kv != 20 ||
        g.sources[size_t(index_source)].index != index_source || g.sources[size_t(index_source)].candidates != 20 || g.compress_ratios[size_t(index_source)] != 1 ||
        g.sources[size_t(layer)].kv != 20 || g.sources[size_t(layer)].index != index_source ||
        g.sources[size_t(layer)].candidates != (layer == index_source ? 20 : -1)))
        throw std::invalid_argument("Layers 24..39 must share source20 KV and their original candidate-filtered index source");
    const std::string prefix = "blk." + std::to_string(layer) + ".attn_";
    auto read = [&](const std::string& name, uint32_t type, const std::vector<uint64_t>& shape) {
        size_t shard = 0;
        const auto* t = model.find(prefix + name, &shard);
        if (!t || t->type != type || t->shape != shape || !model.in_bounds(*t, shard))
            throw std::invalid_argument("Malformed DeepSeek attention tensor " + name);
        const auto* data = model.shard(shard).tensor_data(*t);
        if (uintptr_t(data) % (type == 0 ? 4 : type == 30 ? 2 : 1))
            throw std::invalid_argument("Misaligned DeepSeek attention tensor " + name);
        return data;
    };
    AttentionWeights result; result.layer = layer;
    constexpr std::array<const char*, 5> names{"wq_a", "wq_b", "wkv", "wo_a", "wo_b"};
    constexpr std::array<int, 5> rows{1280, 32768, 512, 8192, 5120}, cols{5120, 1280, 5120, 4096, 8192};
    for (size_t i = 0; i < 5; ++i) {
        auto& m = result.projections[i]; m.rows = rows[i]; m.cols = cols[i];
        m.values = {read(std::string(names[i]) + ".weight", 24, {uint64_t(m.cols), uint64_t(m.rows)}), size_t(m.rows) * m.cols};
        m.scales = {read(std::string(names[i]) + ".scale", 24, {uint64_t(m.cols / 32), uint64_t(m.rows / 32)}), size_t(m.rows / 32) * (m.cols / 32)};
    }
    result.q_norm = {reinterpret_cast<const uint16_t*>(read("q_norm.weight", 30, {1280})), 1280};
    result.kv_norm = {reinterpret_cast<const uint16_t*>(read("kv_norm.weight", 30, {512})), 512};
    result.sink = {reinterpret_cast<const float*>(read("sink", 0, {64})), 64};
    return result;
}
} // namespace strata::deepseek41
