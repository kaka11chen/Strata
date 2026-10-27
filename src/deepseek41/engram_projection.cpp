#include "strata/deepseek41/engram_projection.hpp"

namespace strata::deepseek41 {
EngramWeights read_engram_weights(const GgufModel& model, int layer) {
    if (layer != 1 && layer != 14) throw std::invalid_argument("Engram projection requires original layer 1 or 14");
    const auto* profile = model.meta().get("strata.deepseek41.engram_projection_profile");
    if (!profile || profile->type != MetaType::STRING || profile->s != "1017-layer" + std::to_string(layer) + "-wkv-fp8-block32")
        throw std::invalid_argument("Missing/incorrect fixed Engram projection profile");
    auto read = [&](const char* suffix, uint32_t type, std::vector<uint64_t> shape) {
        const std::string name = "blk." + std::to_string(layer) + ".engram." + suffix;
        size_t shard = 0; const auto* t = model.find(name, &shard);
        if (!t || t->type != type || t->shape != shape || !model.in_bounds(*t, shard))
            throw std::invalid_argument("Malformed Engram projection tensor " + name);
        const auto* data = model.shard(shard).tensor_data(*t);
        if (type == 30 && uintptr_t(data) % 2) throw std::invalid_argument("Misaligned Engram BF16 tensor " + name);
        return data;
    };
    EngramWeights w; w.layer = layer;
    w.wkv = {{read("wkv.weight", 24, {6144,25600}), size_t(25600)*6144},
             {read("wkv.scale", 24, {192,800}), size_t(800)*192}, 25600, 6144};
    w.q = {reinterpret_cast<const uint16_t*>(read("q_weight", 30, {5120,4})), 20480};
    w.k = {reinterpret_cast<const uint16_t*>(read("k_weight", 30, {5120,4})), 20480};
    return w;
}
} // namespace strata::deepseek41
