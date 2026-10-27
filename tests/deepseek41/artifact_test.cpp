#include "strata/deepseek41/artifact.hpp"
#include "../core/gguf_fixture.hpp"

#include <chrono>
#include <functional>
#include <iterator>

namespace fs = std::filesystem;
using namespace strata;
using namespace strata::deepseek41;
namespace {
int failures = 0, checks = 0;
void check(bool ok, const std::string& name) {
    ++checks;
    if (!ok)
        ++failures;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void rejects(const std::string& name, const std::string& message, const std::function<void()>& run) {
    try {
        run();
        check(false, name + " (accepted)");
    } catch (const std::exception& e) {
        check(std::string(e.what()).find(message) != std::string::npos, name + ": " + e.what());
    }
}
struct TempDir {
    fs::path path;
    TempDir()
        : path(fs::temp_directory_path() /
               ("strata-ds41-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

// Copied as data from official 1017 text_config, not obtained from the adapter's expected-value table.
// Model revision 1bae0e1088993b9b82de08f59de5be5a34e42170; GGUF export truncates the three DSpark ratios.
std::vector<fixture::Kv> profile() {
    using namespace fixture;
    std::vector<Kv> v = {str("general.architecture", "deepseek41")};
    const std::pair<const char*, uint64_t> ints[] = {{"vocab_size", 129280},
                                                     {"hidden_size", 5120},
                                                     {"moe_intermediate_size", 2304},
                                                     {"num_hidden_layers", 40},
                                                     {"num_attention_heads", 64},
                                                     {"num_key_value_heads", 1},
                                                     {"head_dim", 512},
                                                     {"qk_rope_head_dim", 64},
                                                     {"q_lora_rank", 1280},
                                                     {"o_lora_rank", 1024},
                                                     {"o_groups", 8},
                                                     {"n_routed_experts", 384},
                                                     {"n_shared_experts", 1},
                                                     {"num_experts_per_tok", 6},
                                                     {"max_position_embeddings", 1048576},
                                                     {"sliding_window", 128},
                                                     {"index_n_heads", 32},
                                                     {"index_head_dim", 128},
                                                     {"index_topk", 512},
                                                     {"candidate_source_layer_id", 20},
                                                     {"candidate_topk_blocks", 2048},
                                                     {"candidate_block_size", 8},
                                                     {"hc_mult", 4},
                                                     {"hc_sinkhorn_iters", 20},
                                                     {"rope_theta", 10000},
                                                     {"compress_rope_theta", 160000}};
    for (const auto& [key, value] : ints)
        v.push_back(u32(std::string("deepseek41.") + key, value));
    const std::pair<const char*, float> floats[] = {
        {"rms_norm_eps", 1e-20f},        {"hc_eps", 1e-6f},
        {"swiglu_limit", 10.f},          {"routed_scaling_factor", 1.5f},
        {"rope_scaling.factor", 16.f},   {"rope_scaling.beta_fast", 32.f},
        {"rope_scaling.beta_slow", 1.f}, {"rope_scaling.original_max_position_embeddings", 65536.f}};
    for (const auto& [key, value] : floats)
        v.push_back(f32(std::string("deepseek41.") + key, value));
    v.push_back(str("deepseek41.scoring_func", "sqrtsoftplus"));
    v.push_back(str("deepseek41.hidden_act", "silu"));
    v.push_back(str("deepseek41.topk_method", "noaux_tc"));
    // The fixture writer supports u32; write_artifact() below converts this field to a real GGUF BOOL.
    v.push_back(u32("deepseek41.norm_topk_prob", 1));
    v.push_back(u32s("deepseek41.compress_ratios", {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    v.push_back(u32s("deepseek41.kv_source_layer_ids", {2, 8, 14, 20}));
    v.push_back(u32s("deepseek41.index_source_layer_ids", {2, 8, 14, 20, 24, 28, 32, 36}));
    return v;
}

// Reuse the existing fixture writer. I32 expert maps have the same payload size as F32;
// patch their directory type, then put known integer bytes into the recorded data span.
// BOOL metadata is encoded separately in profile_file() below.
fixture::Written write_artifact(const fs::path& path, const std::vector<fixture::Kv>& kv,
                                std::vector<fixture::Tensor> ts,
                                const std::map<std::string, std::vector<int32_t>>& maps = {}) {
    for (auto& t : ts)
        if (maps.count(t.name))
            t.type = 0;
    const auto result = fixture::write(path, kv, ts);
    if (maps.empty())
        return result;
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    in.close();
    Cursor cursor(bytes.data(), bytes.size());
    cursor.read<uint32_t>();
    cursor.read<uint32_t>();
    const auto nt = cursor.read<uint64_t>(), nkv = cursor.read<uint64_t>();
    for (uint64_t i = 0; i < nkv; ++i) {
        cursor.str();
        read_value(cursor, MetaType(cursor.read<uint32_t>()));
    }
    std::fstream out(path, std::ios::in | std::ios::out | std::ios::binary);
    for (uint64_t i = 0; i < nt; ++i) {
        const auto name = cursor.str();
        const auto nd = cursor.read<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d)
            cursor.read<uint64_t>();
        const auto type_at = cursor.pos();
        cursor.read<uint32_t>();
        cursor.read<uint64_t>();
        if (const auto it = maps.find(name); it != maps.end()) {
            const uint32_t i32 = 26;
            out.seekp(std::streamoff(type_at));
            out.write(reinterpret_cast<const char*>(&i32), 4);
            out.seekp(std::streamoff(result.data_start + result.offsets[size_t(i)]));
            out.write(reinterpret_cast<const char*>(it->second.data()), std::streamsize(it->second.size() * 4));
        }
    }
    if (!out)
        throw std::runtime_error("cannot patch fixture");
    return result;
}

// A metadata-only GGUF. Keep the common fixture writer unchanged: its output header is
// re-encoded for one BOOL, with the 32-byte header alignment recomputed after shrinking it.
void profile_file(const fs::path& path, const std::vector<fixture::Kv>& kv, bool bool_norm = true) {
    fixture::write(path, kv, {});
    if (!bool_norm)
        return;
    std::ifstream in(path, std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
    in.close();
    Cursor cursor(bytes.data(), bytes.size());
    cursor.read<uint32_t>();
    cursor.read<uint32_t>();
    cursor.read<uint64_t>();
    const auto nkv = cursor.read<uint64_t>();
    size_t norm_at = 0;
    for (uint64_t i = 0; i < nkv; ++i) {
        const auto key = cursor.str();
        const auto at = cursor.pos();
        const auto type = MetaType(cursor.read<uint32_t>());
        read_value(cursor, type);
        if (key == "deepseek41.norm_topk_prob" && type == MetaType::U32)
            norm_at = at;
    }
    bytes.resize(cursor.pos());
    if (norm_at) {
        const uint32_t type = 7;
        std::memcpy(bytes.data() + norm_at, &type, 4);
        bytes.erase(bytes.begin() + std::ptrdiff_t(norm_at + 5), bytes.begin() + std::ptrdiff_t(norm_at + 8));
    }
    bytes.resize((bytes.size() + 31) / 32 * 32, 0);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    if (!out)
        throw std::runtime_error("cannot write profile");
}
fixture::Kv& field(std::vector<fixture::Kv>& kv, const std::string& key) {
    for (auto& v : kv)
        if (v.key == key)
            return v;
    throw std::runtime_error("missing test field");
}
Geometry tiny() {
    Geometry g;
    g.layers = 2;
    g.hidden = 256;
    g.expert_width = 512;
    g.experts = 5;
    g.top_k = 2;
    return g;
}
std::vector<fixture::Tensor> uniform(const Geometry& g, int layer, uint32_t type = 39) {
    const std::string b = "blk." + std::to_string(layer) + ".ffn_";
    return {{b + "gate_exps.weight", {uint64_t(g.hidden), uint64_t(g.expert_width), uint64_t(g.experts)}, type, 1},
            {b + "up_exps.weight", {uint64_t(g.hidden), uint64_t(g.expert_width), uint64_t(g.experts)}, type, 2},
            {b + "down_exps.weight", {uint64_t(g.expert_width), uint64_t(g.hidden), uint64_t(g.experts)}, type, 3}};
}
void profile_tests(const fs::path& dir) {
    const auto p = dir / "profile.gguf";
    profile_file(p, profile());
    const GgufFile f(p.string());
    const auto g = read_1017_geometry(f);
    check(g.layers == 40 && g.hidden == 5120 && g.experts == 384 && g.top_k == 6, "1017 core metadata");
    check(g.sources[0].kv == -1 && g.sources[1].index == -1, "pure SWA layers have no compressed source");
    check(g.sources[7].kv == 2 && g.sources[13].kv == 8 && g.sources[19].kv == 14,
          "ratio-2 layers reuse the latest KV source");
    check(g.sources[39].kv == 20 && g.sources[39].index == 36, "YOCO tail shares KV 20 and index 36");
    check(g.sources[24].candidates == 20 && g.sources[25].candidates == -1,
          "candidate source is consumed by later index owners");
    int mutation = 0;
    auto bad = [&](const std::string& key, const std::function<void(std::vector<fixture::Kv>&)>& change,
                   bool bool_norm = true) {
        auto kv = profile();
        change(kv);
        const auto path = dir / ("bad-profile-" + std::to_string(mutation++) + ".gguf");
        profile_file(path, kv, bool_norm);
        rejects("reject wrong " + key, key, [&] { read_1017_geometry(GgufFile(path.string())); });
    };
    bad("general.architecture", [](auto& kv) { field(kv, "general.architecture").s = "qwen4exp"; });
    bad("hidden_size", [](auto& kv) { field(kv, "deepseek41.hidden_size").u = 2560; });
    bad("hidden_size",
        [](auto& kv) { field(kv, "deepseek41.hidden_size") = fixture::f32("deepseek41.hidden_size", 5120.f); });
    bad("hidden_size",
        [](auto& kv) { field(kv, "deepseek41.hidden_size") = fixture::i32("deepseek41.hidden_size", uint64_t(-1)); });
    bad("head_dim", [](auto& kv) { std::erase_if(kv, [](const auto& v) { return v.key == "deepseek41.head_dim"; }); });
    bad("rms_norm_eps", [](auto& kv) { field(kv, "deepseek41.rms_norm_eps").f = 1e-6f; });
    bad("rms_norm_eps",
        [](auto& kv) { field(kv, "deepseek41.rms_norm_eps").f = std::numeric_limits<float>::quiet_NaN(); });
    bad("norm_topk_prob", [](auto& kv) { field(kv, "deepseek41.norm_topk_prob").u = 0; });
    bad(
        "norm_topk_prob", [](auto&) {}, false);
    bad("compress_ratios", [](auto& kv) { field(kv, "deepseek41.compress_ratios").arr.push_back(0); });
    bad("kv_source_layer_ids", [](auto& kv) {
        std::swap(field(kv, "deepseek41.kv_source_layer_ids").arr[0],
                  field(kv, "deepseek41.kv_source_layer_ids").arr[1]);
    });
    bad("index_source_layer_ids", [](auto& kv) {
        field(kv, "deepseek41.index_source_layer_ids").elem = 8;
        field(kv, "deepseek41.index_source_layer_ids").sarr.assign(8, "2");
    });
}

void expert_tests(const fs::path& dir) {
    auto g = tiny();
    auto ts = uniform(g, 0);
    const auto p = dir / "uniform.gguf";
    const auto written = write_artifact(p, {}, ts);
    const GgufModel m({p.string()});
    const ExpertIndex index(m, g, {0});
    const size_t one_role = size_t(g.hidden) * size_t(g.expert_width) / 32 * 17;
    check(index.total_bytes() == one_role * 3 * size_t(g.experts),
          "MXFP4 byte budget includes one scale per 32 values");
    for (int e : {0, 2, 4}) {
        const auto& v = index.view(0, e);
        std::vector<uint8_t> dst(v.bytes());
        index.copy(0, e, dst);
        bool same = true;
        for (size_t role = 0; role < 3; ++role) {
            same &= v.parts[role].file_offset == written.data_start + written.offsets[role] + size_t(e) * one_role;
            for (size_t j = 0; j < one_role; ++j)
                if (dst[role * one_role + j] != fixture::pattern(ts[role].seed, size_t(e) * one_role + j)) {
                    same = false;
                    break;
                }
        }
        check(same, "exact first/middle/last expert bytes: " + std::to_string(e));
    }
    rejects("partial file is not a complete backbone", "blk.1.", [&] { ExpertIndex full(m, g); });
    rejects("invalid expert", "outside", [&] { index.view(0, 5); });
    rejects("negative expert", "outside", [&] { index.view(0, -1); });
    rejects("unchecked layer", "outside", [&] { index.view(1, 0); });
    rejects("repeated layer", "repeated", [&] { ExpertIndex bad(m, g, {0, 0}); });
    rejects("invalid layer", "layer", [&] { ExpertIndex bad(m, g, {2}); });
    std::vector<uint8_t> short_buffer(index.view(0, 0).bytes() - 1, 77);
    rejects("bounded copy refuses short destination", "exactly", [&] { index.copy(0, 0, short_buffer); });
    check(std::all_of(short_buffer.begin(), short_buffer.end(), [](auto b) { return b == 77; }),
          "failed copy does not alter destination");
    for (auto type : {0u, 1u, 8u, 11u, 30u, 39u, 42u, 143u}) {
        auto path = dir / ("type-" + std::to_string(type) + ".gguf");
        write_artifact(path, {}, uniform(g, 0, type));
        const GgufModel model({path.string()});
        const ExpertIndex idx(model, g, {0});
        check(idx.view(0, 4).parts[0].type == type, "supported raw expert layout type " + std::to_string(type));
    }
    auto bad_shape = uniform(g, 0);
    bad_shape[2].shape[0] = 256;
    auto shape_path = dir / "wrong-shape.gguf";
    write_artifact(shape_path, {}, bad_shape);
    rejects("transposed or short matrix rejected", "shape/type", [&] {
        const GgufModel model({shape_path.string()});
        ExpertIndex idx(model, g, {0});
    });
    auto mismatch = uniform(g, 0);
    mismatch[1].type = 42;
    auto mismatch_path = dir / "mismatch.gguf";
    write_artifact(mismatch_path, {}, mismatch);
    rejects("gate and up format mismatch", "shape/type", [&] {
        const GgufModel model({mismatch_path.string()});
        ExpertIndex idx(model, g, {0});
    });
    auto truncated = dir / "truncated.gguf";
    write_artifact(truncated, {}, uniform(g, 0));
    fs::resize_file(truncated, fs::file_size(truncated) - 1);
    rejects("truncated expert payload", "bounds", [&] {
        const GgufModel model({truncated.string()});
        ExpertIndex idx(model, g, {0});
    });
    // Different roles can live in different shards; the index must preserve each role's shard and offset.
    const auto a = dir / "split-00001-of-00002.gguf", b = dir / "split-00002-of-00002.gguf";
    auto metadata = fixture::split_keys(0, 2, 3);
    metadata.push_back(fixture::str("general.architecture", "deepseek41"));
    write_artifact(a, metadata, {ts[0], ts[2]});
    write_artifact(b, fixture::split_keys(1, 2, 3), {ts[1]});
    const auto split = GgufModel::open(b.string());
    const ExpertIndex si(split, g, {0});
    check(si.view(0, 4).parts[0].shard == 0 && si.view(0, 4).parts[1].shard == 1 && si.view(0, 4).parts[2].shard == 0,
          "expert roles across GGUF shards");
    std::vector<uint8_t> x(index.view(0, 4).bytes()), y(si.view(0, 4).bytes());
    index.copy(0, 4, x);
    si.copy(0, 4, y);
    check(x == y, "split and single-file expert copies are byte-identical");
}

void mixed_tests(const fs::path& dir) {
    const auto g = tiny();
    const std::vector<int32_t> gu = {42, 11, 42, 143, 11}, down = {11, 42, 143, 11, 42};
    const std::map<std::string, std::vector<int32_t>> maps = {{"blk.0.ffn_exps_gu_type", gu},
                                                              {"blk.0.ffn_exps_down_type", down}};
    std::vector<fixture::Tensor> ts;
    for (const auto& [name, types] : maps)
        ts.push_back({name, {uint64_t(types.size())}, 0, 0});
    uint8_t seed = 10;
    for (int role = 0; role < 3; ++role) {
        const auto& types = role == 2 ? down : gu;
        std::map<int, int> counts;
        for (int t : types)
            ++counts[t];
        for (const auto& [t, n] : counts) {
            std::string tn = ggml_type_name(uint32_t(t));
            std::transform(tn.begin(), tn.end(), tn.begin(), [](unsigned char c) { return char(std::tolower(c)); });
            const std::string name = std::string("blk.0.ffn_") +
                                     (role == 0   ? "gate"
                                      : role == 1 ? "up"
                                                  : "down") +
                                     "_exps." + tn + ".weight";
            ts.push_back({name,
                          {uint64_t(role == 2 ? g.expert_width : g.hidden),
                           uint64_t(role == 2 ? g.hidden : g.expert_width), uint64_t(n)},
                          uint32_t(t),
                          seed++});
        }
    }
    const auto p = dir / "mixed.gguf";
    write_artifact(p, {}, ts, maps);
    const GgufModel m({p.string()});
    const ExpertIndex index(m, g, {0});
    for (int e = 0; e < g.experts; ++e) {
        bool same = true;
        const auto& v = index.view(0, e);
        std::vector<uint8_t> copied(v.bytes());
        index.copy(0, e, copied);
        size_t copied_offset = 0;
        for (size_t role = 0; role < 3; ++role) {
            const auto& types = role == 2 ? down : gu;
            const int slot = int(std::count(types.begin(), types.begin() + e, types[size_t(e)]));
            const auto& part = v.parts[role];
            same &= part.type == uint32_t(types[size_t(e)]);
            const auto tensor = std::find_if(ts.begin(), ts.end(), [&](const auto& t) {
                return t.type == part.type && t.name.find(role == 0   ? "gate_exps."
                                                          : role == 1 ? "up_exps."
                                                                      : "down_exps.") != std::string::npos;
            });
            if (tensor == ts.end()) {
                same = false;
                continue;
            }
            for (size_t j = 0; j < part.data.size(); ++j)
                if (part.data[j] != fixture::pattern(tensor->seed, size_t(slot) * part.data.size() + j) ||
                    copied[copied_offset + j] != fixture::pattern(tensor->seed, size_t(slot) * part.data.size() + j)) {
                    same = false;
                    break;
                }
            copied_offset += part.data.size();
        }
        check(same, "mixed types keep original expert id and type-local slot: " + std::to_string(e));
    }
    auto bad_maps = maps;
    bad_maps.begin()->second[2] = 999;
    const auto bad = dir / "bad-tier.gguf";
    write_artifact(bad, {}, ts, bad_maps);
    rejects("unknown mixed tier", "unsupported expert tier", [&] {
        const GgufModel model({bad.string()});
        ExpertIndex i(model, g, {0});
    });
    auto ambiguous = ts;
    const auto ut = uniform(g, 0);
    ambiguous.push_back(ut[0]);
    const auto amb = dir / "ambiguous.gguf";
    write_artifact(amb, {}, ambiguous, maps);
    rejects("mixed and uniform forms cannot silently conflict", "ambiguous", [&] {
        const GgufModel model({amb.string()});
        ExpertIndex i(model, g, {0});
    });
    const auto short_map = dir / "short-map.gguf";
    write_artifact(short_map, {}, {{"blk.0.ffn_exps_gu_type", {5}, 0, 0}}, {{"blk.0.ffn_exps_gu_type", gu}});
    fs::resize_file(short_map, fs::file_size(short_map) - 1);
    rejects("truncated type map is checked before dereferencing", "truncated type map", [&] {
        const GgufModel model({short_map.string()});
        ExpertIndex i(model, g, {0});
    });
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-profile") {
            profile_file(argv[2], profile());
            return 0;
        }
        TempDir dir;
        profile_tests(dir.path);
        expert_tests(dir.path);
        mixed_tests(dir.path);
        std::printf("%d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "unexpected error: %s\n", e.what());
        return 1;
    }
}
