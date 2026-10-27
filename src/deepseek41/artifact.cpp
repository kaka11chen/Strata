// Core profile and expert tensor addressing; see NOTICE.md for the source contract.
#include "strata/deepseek41/artifact.hpp"

#include <cctype>
#include <limits>
#include <numeric>
#include <set>

namespace strata::deepseek41 {
namespace {
const MetaValue& required(const GgufFile& f, const std::string& key) {
    const auto* v = f.get(key);
    if (!v)
        throw std::runtime_error("DeepSeek: missing " + key);
    return *v;
}
uint64_t integer(const MetaValue& v, const std::string& key) {
    switch (v.type) {
    case MetaType::U8:
    case MetaType::I8:
    case MetaType::U16:
    case MetaType::I16:
    case MetaType::U32:
    case MetaType::I32:
    case MetaType::U64:
    case MetaType::I64:
        if (v.u <= uint64_t((std::numeric_limits<int64_t>::max)()))
            return v.u;
        break;
    default:
        break;
    }
    throw std::runtime_error("DeepSeek: " + key + " must be a nonnegative integer");
}
void expect_integer(const GgufFile& f, const std::string& key, uint64_t expected) {
    if (integer(required(f, key), key) != expected)
        throw std::runtime_error("DeepSeek: " + key + " must be " + std::to_string(expected) + " for 1017");
}
void expect_string(const GgufFile& f, const std::string& key, const std::string& expected) {
    const auto& v = required(f, key);
    if (v.type != MetaType::STRING || v.s != expected)
        throw std::runtime_error("DeepSeek: " + key + " must be '" + expected + "'");
}
void expect_float(const GgufFile& f, const std::string& key, float expected) {
    const auto& v = required(f, key);
    if ((v.type != MetaType::F32 && v.type != MetaType::F64) || !std::isfinite(v.f) ||
        static_cast<float>(v.f) != expected)
        throw std::runtime_error("DeepSeek: " + key + " differs from the 1017 profile");
}
void expect_array(const GgufFile& f, const std::string& key, const std::vector<int>& expected) {
    const auto& v = required(f, key);
    if (v.type != MetaType::ARRAY || v.count != expected.size() || v.items.size() != expected.size())
        throw std::runtime_error("DeepSeek: " + key + " has the wrong array length for 1017");
    for (size_t i = 0; i < expected.size(); ++i)
        if (integer(v.items[i], key) != uint64_t(expected[i]))
            throw std::runtime_error("DeepSeek: " + key + " differs at layer/list index " + std::to_string(i));
}
bool expert_type(uint32_t type) {
    // Raw addressing only; this list does not claim compute support for these formats.
    return type == 0 || type == 1 || type == 8 || type == 11 || type == 30 || type == 39 || type == 42 || type == 143;
}
ExpertPart slice(const GgufModel& m, const std::string& name, uint32_t type, int input, int output, int count,
                 int slot) {
    size_t shard = 0;
    const auto* t = m.find(name, &shard);
    if (!t)
        throw std::runtime_error("DeepSeek: missing tensor " + name);
    const std::vector<uint64_t> expected = {uint64_t(input), uint64_t(output), uint64_t(count)};
    if (t->shape != expected || t->type != type)
        throw std::runtime_error("DeepSeek: " + name + " has the wrong shape/type");
    if (!expert_type(type) || !m.in_bounds(*t, shard))
        throw std::runtime_error("DeepSeek: " + name + " has an unsupported format or invalid payload bounds");
    const uint64_t bytes = tensor_payload_bytes(*t) / uint64_t(count);
    if (bytes > (std::numeric_limits<size_t>::max)())
        throw std::runtime_error("DeepSeek: expert slice exceeds this process's address range");
    const uint64_t offset = bytes * uint64_t(slot);
    const auto& f = m.shard(shard);
    return {{f.tensor_data(*t) + offset, size_t(bytes)}, type, shard, f.data_start() + t->offset + offset};
}
std::vector<int> type_map(const GgufModel& m, const TensorInfo& t, size_t shard, int experts) {
    // I32 (ggml type 26) is not a compute format in Strata's artifact reader. Check its raw
    // span here before reading it, without widening the default Qwen reader's type policy.
    if (t.type != 26 || t.shape != std::vector<uint64_t>{uint64_t(experts)})
        throw std::runtime_error("DeepSeek: " + t.name + " must be an I32 expert type map");
    const auto& f = m.shard(shard);
    const uint64_t payload = f.file_size() - f.data_start();
    const uint64_t bytes = uint64_t(experts) * sizeof(int32_t);
    if (t.offset > payload || bytes > payload - t.offset)
        throw std::runtime_error("DeepSeek: truncated type map " + t.name);
    std::vector<int> result(size_t(experts), 0);
    for (int e = 0; e < experts; ++e) {
        int32_t type;
        std::memcpy(&type, f.tensor_data(t) + size_t(e) * sizeof(type), sizeof(type));
        if (type != 11 && type != 42 && type != 143)
            throw std::runtime_error("DeepSeek: " + t.name + " contains unsupported expert tier " +
                                     std::to_string(type));
        result[size_t(e)] = type;
    }
    return result;
}
} // namespace

Geometry read_1017_geometry(const GgufFile& f) {
    expect_string(f, "general.architecture", "deepseek41");
    const std::string p = "deepseek41.";
    // Official DeepSeek-V4.1-Flash-1017 text_config, serialized as Xolotl GGUF metadata.
    for (const auto& [key, value] :
         std::initializer_list<std::pair<const char*, uint64_t>>{{"hidden_size", 5120},
                                                                 {"moe_intermediate_size", 2304},
                                                                 {"num_hidden_layers", 40},
                                                                 {"n_routed_experts", 384},
                                                                 {"num_experts_per_tok", 6},
                                                                 {"n_shared_experts", 1},
                                                                 {"num_attention_heads", 64},
                                                                 {"num_key_value_heads", 1},
                                                                 {"head_dim", 512},
                                                                 {"qk_rope_head_dim", 64},
                                                                 {"q_lora_rank", 1280},
                                                                 {"o_lora_rank", 1024},
                                                                 {"o_groups", 8},
                                                                 {"sliding_window", 128},
                                                                 {"hc_mult", 4},
                                                                 {"hc_sinkhorn_iters", 20},
                                                                 {"index_n_heads", 32},
                                                                 {"index_head_dim", 128},
                                                                 {"index_topk", 512},
                                                                 {"candidate_source_layer_id", 20},
                                                                 {"candidate_topk_blocks", 2048},
                                                                 {"candidate_block_size", 8},
                                                                 {"vocab_size", 129280},
                                                                 {"max_position_embeddings", 1048576},
                                                                 {"rope_theta", 10000},
                                                                 {"compress_rope_theta", 160000}})
        expect_integer(f, p + key, value);
    for (const auto& [key, value] : std::initializer_list<std::pair<const char*, float>>{
             {"rms_norm_eps", 1e-20f},
             {"hc_eps", 1e-6f},
             {"swiglu_limit", 10.f},
             {"routed_scaling_factor", 1.5f},
             {"rope_scaling.factor", 16.f},
             {"rope_scaling.beta_fast", 32.f},
             {"rope_scaling.beta_slow", 1.f},
             {"rope_scaling.original_max_position_embeddings", 65536.f}})
        expect_float(f, p + key, value);
    expect_string(f, p + "scoring_func", "sqrtsoftplus");
    expect_string(f, p + "hidden_act", "silu");
    expect_string(f, p + "topk_method", "noaux_tc");
    const std::string norm_key = p + "norm_topk_prob";
    const auto& norm = required(f, norm_key);
    if (norm.type != MetaType::BOOL || !norm.u)
        throw std::runtime_error("DeepSeek: deepseek41.norm_topk_prob must be true");

    Geometry g;
    g.layers = 40;
    g.hidden = 5120;
    g.expert_width = 2304;
    g.experts = 384;
    g.top_k = 6;
    g.compress_ratios.assign(40, 1);
    std::fill(g.compress_ratios.begin(), g.compress_ratios.begin() + 2, 0);
    std::fill(g.compress_ratios.begin() + 2, g.compress_ratios.begin() + 20, 2);
    const std::vector<int> kv = {2, 8, 14, 20}, index = {2, 8, 14, 20, 24, 28, 32, 36};
    expect_array(f, p + "compress_ratios", g.compress_ratios);
    expect_array(f, p + "kv_source_layer_ids", kv);
    expect_array(f, p + "index_source_layer_ids", index);
    for (int l = 0; l < g.layers; ++l) {
        LayerSources s;
        if (g.compress_ratios[size_t(l)]) {
            for (int source : kv)
                if (source <= l)
                    s.kv = source;
            for (int source : index)
                if (source <= l)
                    s.index = source;
            if (l > 20 && std::find(index.begin(), index.end(), l) != index.end())
                s.candidates = 20;
        }
        g.sources.push_back(s);
    }
    return g;
}

size_t ExpertView::bytes() const {
    size_t total = 0;
    for (const auto& part : parts) {
        if (part.data.size() > (std::numeric_limits<size_t>::max)() - total)
            throw std::runtime_error("DeepSeek: expert byte count overflow");
        total += part.data.size();
    }
    return total;
}

ExpertIndex::ExpertIndex(const GgufModel& m, const Geometry& g, std::vector<int> layers) {
    if (g.layers <= 0 || g.layers > 40 || g.hidden <= 0 || g.expert_width <= 0 || g.experts <= 0 || g.experts > 384)
        throw std::runtime_error("DeepSeek: invalid expert geometry");
    if (layers.empty()) {
        layers.resize(size_t(g.layers));
        std::iota(layers.begin(), layers.end(), 0);
    }
    std::set<int> seen;
    for (int l : layers) {
        if (l < 0 || l >= g.layers || !seen.insert(l).second)
            throw std::runtime_error("DeepSeek: invalid or repeated layer " + std::to_string(l));
        auto& views = layers_[l];
        views.resize(size_t(g.experts));
        const std::string b = "blk." + std::to_string(l) + ".";
        for (int down = 0; down < 2; ++down) {
            const std::string map_name = b + (down ? "ffn_exps_down_type" : "ffn_exps_gu_type");
            size_t shard = 0;
            const auto* map_tensor = m.find(map_name, &shard);
            std::vector<int> types;
            if (map_tensor)
                types = type_map(m, *map_tensor, shard, g.experts);
            else {
                const auto* t = m.find(b + (down ? "ffn_down_exps.weight" : "ffn_gate_exps.weight"));
                if (!t)
                    throw std::runtime_error("DeepSeek: missing " + b +
                                             (down ? "ffn_down_exps.weight" : "ffn_gate_exps.weight"));
                if (!expert_type(t->type))
                    throw std::runtime_error("DeepSeek: unsupported expert tensor type " + std::to_string(t->type));
                types.assign(size_t(g.experts), int(t->type));
            }
            std::map<int, int> counts, next;
            for (int t : types)
                ++counts[t];
            for (int e = 0; e < g.experts; ++e) {
                const int type = types[size_t(e)], slot = next[type]++;
                std::string suffix = ".weight";
                if (map_tensor) {
                    std::string name = ggml_type_name(uint32_t(type));
                    for (char& c : name)
                        c = char(std::tolower(static_cast<unsigned char>(c)));
                    suffix = "." + name + suffix;
                }
                for (int role = down ? 2 : 0; role < (down ? 3 : 2); ++role) {
                    const char* role_name = role == 0 ? "gate" : role == 1 ? "up" : "down";
                    const std::string stem = b + "ffn_" + role_name + "_exps";
                    if (map_tensor && m.find(stem + ".weight"))
                        throw std::runtime_error("DeepSeek: ambiguous uniform and mixed tensors for " + stem);
                    views[size_t(e)].parts[size_t(role)] =
                        slice(m, stem + suffix, uint32_t(type), down ? g.expert_width : g.hidden,
                              down ? g.hidden : g.expert_width, counts[type], slot);
                }
            }
        }
        for (const auto& v : views) {
            const auto bytes = v.bytes();
            if (bytes > (std::numeric_limits<uint64_t>::max)() - total_bytes_)
                throw std::runtime_error("DeepSeek: total expert byte count overflow");
            total_bytes_ += bytes;
        }
    }
}

const ExpertView& ExpertIndex::view(int layer, int expert) const {
    const auto it = layers_.find(layer);
    if (it == layers_.end() || expert < 0 || size_t(expert) >= it->second.size())
        throw std::out_of_range("DeepSeek: layer/expert is outside the checked index");
    return it->second[size_t(expert)];
}
void ExpertIndex::copy(int layer, int expert, std::span<uint8_t> dst) const {
    const auto& v = view(layer, expert);
    if (dst.size() != v.bytes())
        throw std::invalid_argument("DeepSeek: destination must fit exactly one expert");
    size_t offset = 0;
    for (const auto& part : v.parts) {
        std::memcpy(dst.data() + offset, part.data.data(), part.data.size());
        offset += part.data.size();
    }
}
} // namespace strata::deepseek41
