#include "strata/deepseek41/source20_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace strata::deepseek41 {

Source20CompressorWeights read_source20_compressor_weights(const GgufModel& model, const Geometry& g) {
    const auto* profile = model.meta().get("strata.deepseek41.compressor_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != "1017-layer20-ratio1-bf16")
        throw std::invalid_argument("Missing or mismatched DeepSeek source-20 profile");
    if (g.layers != 40 || g.hidden != 5120 || g.compress_ratios.size() != 40 ||
        g.compress_ratios[20] != 1 || g.sources.size() != 40 || g.sources[20].kv != 20 ||
        g.sources[20].index != 20 || g.sources[20].candidates != -1)
        throw std::invalid_argument("Source 20 must own ratio-1 KV/index and publish candidates without consuming them");
    if (model.find("blk.20.attn_compressor_wgate.weight"))
        throw std::invalid_argument("Source-20 ratio-1 compressor has no gate");
    auto read = [&](const std::string& part, const std::vector<uint64_t>& shape, size_t count) {
        size_t shard = 0;
        const auto* tensor = model.find("blk.20.attn_compressor_" + part + ".weight", &shard);
        if (!tensor || tensor->type != 30 || tensor->shape != shape || !model.in_bounds(*tensor, shard))
            throw std::invalid_argument("Malformed DeepSeek source-20 compressor tensor " + part);
        const auto* data = model.shard(shard).tensor_data(*tensor);
        if (uintptr_t(data) % alignof(uint16_t))
            throw std::invalid_argument("Misaligned DeepSeek source-20 compressor tensor " + part);
        return std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(data), count);
    };
    return {read("wkv", {5120, 512}, 5120 * 512), read("norm", {512}, 512)};
}

Source20Batch::Source20Batch(int start, int tokens, int expected_next, int context_capacity, int prefill_capacity)
    : start_(start), tokens_(tokens) {
    if (context_capacity < 1 || context_capacity > max_context ||
        prefill_capacity < 1 || prefill_capacity > context_capacity)
        throw std::invalid_argument("Invalid source-20 context/prefill capacity");
    if (start < 0 || start != expected_next || start >= context_capacity || tokens < 1 ||
        tokens > context_capacity - start)
        throw std::invalid_argument("Source-20 call must be contiguous and fit its context");
    if ((start == 0 && tokens > prefill_capacity) || (start != 0 && tokens != 1))
        throw std::invalid_argument("Source-20 requires bounded initial prefill then single-token decode");
    if (size_t(tokens) > std::numeric_limits<size_t>::max() / size_t(candidate_stride()))
        throw std::invalid_argument("Source-20 candidate layout exceeds addressable size");
}

int Source20Batch::position(int query) const {
    if (query < 0 || query >= tokens_) throw std::invalid_argument("Source-20 query is outside this call");
    return start_ + query;
}

int Source20Batch::candidate_stride() const {
    return std::min(topk_blocks, (key_count() + block_size - 1) / block_size);
}

void validate_source20_candidates(const Source20Batch& batch, int query,
                                  std::span<const float> logits, std::span<const int32_t> ids) {
    const int visible = batch.visible_keys(query);
    if (logits.size() != size_t(batch.key_count()) || ids.size() != size_t(batch.candidate_stride()))
        throw std::invalid_argument("Source-20 candidate row has the wrong width");
    int selected = 0, previous = -1;
    bool padding = false, newest_selected = false;
    const int newest = batch.newest_block(query);
    for (int32_t id : ids) {
        if (id == -1) {
            padding = true;
        } else {
            if (padding || id < 0 || id <= previous || id > newest)
                throw std::invalid_argument("Source-20 candidates require ordered unique reachable block IDs then -1");
            previous = id;
            newest_selected |= id == newest;
            ++selected;
        }
    }
    if (!newest_selected) throw std::invalid_argument("Source-20 candidates must include the newest block");

    const float inf = std::numeric_limits<float>::infinity();
    float min_selected = inf, max_omitted = -inf;
    int reachable = 0, selected_index = 0;
    const int blocks = (batch.key_count() + Source20Batch::block_size - 1) / Source20Batch::block_size;
    for (int block = 0; block < blocks; ++block) {
        float score = -inf;
        const int begin = block * Source20Batch::block_size;
        const int end = std::min(begin + Source20Batch::block_size, batch.key_count());
        for (int position = begin; position < end; ++position) {
            const float value = logits[size_t(position)];
            if (std::isnan(value) || value == inf || (position >= visible && value != -inf))
                throw std::invalid_argument("Source-20 scores must be finite or -inf and already causally masked");
            score = std::max(score, value);
        }
        if (block == newest) score = inf; // Pin even if its original score was -inf.
        reachable += score > -inf;
        if (selected_index < selected && ids[size_t(selected_index)] == block) {
            if (score == -inf) throw std::invalid_argument("Source-20 candidate includes an unreachable block");
            min_selected = std::min(min_selected, score);
            ++selected_index;
        } else {
            max_omitted = std::max(max_omitted, score);
        }
    }
    if (selected != std::min(Source20Batch::topk_blocks, reachable) || min_selected < max_omitted)
        throw std::invalid_argument("Source-20 candidates do not satisfy reference block top-k");
}

} // namespace strata::deepseek41
