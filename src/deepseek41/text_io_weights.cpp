#include "strata/deepseek41/text_io_weights.hpp"

namespace strata::deepseek41 {
TextIoWeights read_text_io_weights(const GgufModel& model, int hidden, int vocab) {
    for (const auto& entry : {std::pair{"strata.deepseek41.text_io_profile", text_io_profile},
                              std::pair{"strata.deepseek41.model_revision", text_io_revision}}) {
        const auto* value = model.meta().get(entry.first);
        if (!value || value->type != MetaType::STRING || value->s != entry.second)
            throw std::invalid_argument(std::string("Wrong DeepSeek text IO identity: ") + entry.first);
    }
    if (hidden <= 0 || hidden > 5120 || hidden % 32 || vocab <= 0 || vocab > 129280)
        throw std::invalid_argument("Invalid DeepSeek text IO geometry");
    auto read = [&](const char* name, const std::vector<uint64_t>& shape, size_t count) {
        size_t shard = 0;
        const auto* tensor = model.find(name, &shard);
        if (!tensor || tensor->type != 30 || tensor->shape != shape || !model.in_bounds(*tensor, shard))
            throw std::invalid_argument(std::string("Malformed DeepSeek text IO tensor: ") + name);
        const auto* data = model.shard(shard).tensor_data(*tensor);
        if (uintptr_t(data) % 2) throw std::invalid_argument("Misaligned DeepSeek text IO weight");
        return std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(data), count);
    };
    const size_t count = size_t(hidden) * size_t(vocab);
    return {hidden, vocab, read("token_embd.weight", {uint64_t(hidden), uint64_t(vocab)}, count),
            read("output_norm.weight", {uint64_t(hidden)}, size_t(hidden)),
            read("output.weight", {uint64_t(hidden), uint64_t(vocab)}, count)};
}
} // namespace strata::deepseek41
