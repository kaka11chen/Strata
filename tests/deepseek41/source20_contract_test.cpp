#include "strata/deepseek41/source20_contract.hpp"
#include "strata/deepseek41/compressor_weights.hpp"
#include "../core/gguf_fixture.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <numeric>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) {
    ++checks; failures += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void rejects(const std::string& name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::exception&) { check(true, name); }
}
Geometry geometry() {
    Geometry g; g.layers = 40; g.hidden = 5120;
    g.compress_ratios.resize(40); g.compress_ratios[20] = 1;
    g.sources.resize(40); g.sources[20] = {20, 20, -1};
    return g;
}
void weights(const fs::path& dir) {
    const auto g = geometry();
    const std::vector<fixture::Kv> meta{
        fixture::str("strata.deepseek41.compressor_profile", "1017-layer20-ratio1-bf16")};
    const std::vector<fixture::Tensor> tensors{
        {"blk.20.attn_compressor_wkv.weight", {5120, 512}, 30, 1},
        {"blk.20.attn_compressor_norm.weight", {512}, 30, 2}};
    int file = 0;
    auto open = [&](const std::vector<fixture::Kv>& metadata, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
        const auto path = dir / (std::to_string(file++) + ".gguf");
        fixture::write(path, metadata, ts, cut);
        return GgufModel::open(path.string());
    };
    const auto model = open(meta, tensors);
    const auto w = read_source20_compressor_weights(model, g);
    check(w.kv.size() == 5120 * 512 && w.norm.size() == 512, "source-20 mapped BF16 weight dimensions");
    bool exact = true; uint8_t seed = 0;
    for (auto view : {w.kv, w.norm}) {
        ++seed;
        const auto* bytes = reinterpret_cast<const uint8_t*>(view.data());
        for (size_t i = 0; i < view.size_bytes(); ++i) exact &= bytes[i] == fixture::pattern(seed, i);
    }
    check(exact, "source-20 preserves every original BF16 weight byte");
    rejects("missing profile rejected", [&] { const auto m = open({}, tensors); read_source20_compressor_weights(m, g); });
    rejects("wrong profile type rejected", [&] {
        const auto m = open({fixture::u32("strata.deepseek41.compressor_profile", 20)}, tensors);
        read_source20_compressor_weights(m, g);
    });
    rejects("ratio-2 profile rejected", [&] {
        const auto m = open({fixture::str("strata.deepseek41.compressor_profile", "1017-layer14-ratio2-bf16")}, tensors);
        read_source20_compressor_weights(m, g);
    });
    for (size_t i = 0; i < tensors.size(); ++i) {
        rejects("missing source tensor rejected", [&] {
            auto ts = tensors; ts.erase(ts.begin() + i); const auto m = open(meta, ts);
            read_source20_compressor_weights(m, g);
        });
        rejects("non-BF16 source rejected", [&] {
            auto ts = tensors; ts[i].type = 0; const auto m = open(meta, ts);
            read_source20_compressor_weights(m, g);
        });
    }
    rejects("transposed source-20 projection rejected", [&] {
        auto ts = tensors; ts[0].shape = {512, 5120}; const auto m = open(meta, ts);
        read_source20_compressor_weights(m, g);
    });
    rejects("wrong source-20 norm width rejected", [&] {
        auto ts = tensors; ts[1].shape = {511}; const auto m = open(meta, ts);
        read_source20_compressor_weights(m, g);
    });
    rejects("truncated source-20 norm rejected", [&] {
        const auto m = open(meta, tensors, 1); read_source20_compressor_weights(m, g);
    });
    rejects("ratio-1 cannot have a gate", [&] {
        auto ts = tensors; ts.push_back({"blk.20.attn_compressor_wgate.weight", {1}, 30, 3});
        const auto m = open(meta, ts); read_source20_compressor_weights(m, g);
    });
    for (int kind = 0; kind < 8; ++kind) {
        auto bad = g;
        if (kind == 0) bad.layers = 39;
        if (kind == 1) bad.hidden = 4096;
        if (kind == 2) bad.compress_ratios.clear();
        if (kind == 3) bad.compress_ratios[20] = 2;
        if (kind == 4) bad.sources.clear();
        if (kind == 5) bad.sources[20].kv = 14;
        if (kind == 6) bad.sources[20].index = 14;
        if (kind == 7) bad.sources[20].candidates = 20;
        rejects("wrong source-20 topology " + std::to_string(kind), [&] { read_source20_compressor_weights(model, bad); });
    }
    for (int layer : {2, 8, 14, 20})
        rejects("ratio-2 reader cannot consume source-20 weights", [&] { read_compressor_weights(model, g, layer); });
}
void batches() {
    for (int tokens : {1, 7, 8, 9, 17}) {
        const Source20Batch batch(0, tokens, 0, 32, 17);
        bool positions = batch.first_position() == 0 && batch.latent_count() == tokens && batch.key_count() == tokens;
        for (int query = 0; query < tokens; ++query)
            positions &= batch.position(query) == query && batch.visible_keys(query) == query + 1 &&
                         batch.newest_block(query) == query / 8;
        check(positions, "ratio-1 emits every prefill token including an odd tail");
        check(batch.candidate_entries() == size_t(tokens) * size_t((tokens + 7) / 8), "compact candidate row stride");
        for (int next = tokens; next < 20; ++next) {
            const Source20Batch decode(next, 1, next, 32, 17);
            check(decode.position(0) == next && decode.visible_keys(0) == next + 1 &&
                  decode.key_count() == next + 1, "decode includes the current token without pair delay");
        }
    }
    auto bad = [](int start, int tokens, int expected, int context, int prefill) {
        Source20Batch batch(start, tokens, expected, context, prefill);
        (void)batch;
    };
    rejects("zero context", [&] { bad(0, 1, 0, 0, 1); });
    rejects("context above model limit", [&] { bad(0, 1, 0, 1048577, 1); });
    rejects("negative prefill capacity", [&] { bad(0, 1, 0, 32, -1); });
    rejects("prefill capacity above context", [&] { bad(0, 1, 0, 32, 33); });
    rejects("empty batch", [&] { bad(0, 0, 0, 32, 8); });
    rejects("negative token count", [&] { bad(0, -1, 0, 32, 8); });
    rejects("negative start", [&] { bad(-1, 1, -1, 32, 8); });
    rejects("stale start", [&] { bad(7, 1, 8, 32, 8); });
    rejects("skipped start", [&] { bad(9, 1, 8, 32, 8); });
    rejects("unreset history", [&] { bad(0, 1, 8, 32, 8); });
    rejects("initial prefill above capacity", [&] { bad(0, 9, 0, 32, 8); });
    rejects("multi-token decode", [&] { bad(8, 2, 8, 32, 8); });
    rejects("full cache", [&] { bad(32, 1, 32, 32, 8); });
    rejects("oversized count without overflow", [&] { bad(1, std::numeric_limits<int>::max(), 1, 32, 8); });
    rejects("oversized start without overflow", [&] { bad(std::numeric_limits<int>::max(), 1, std::numeric_limits<int>::max(), 32, 8); });
    const Source20Batch batch(0, 9, 0, 32, 9);
    rejects("negative query", [&] { batch.position(-1); });
    rejects("query beyond emitted latents", [&] { batch.visible_keys(9); });
    for (int width : {16384, 16385, 1048576}) {
        const Source20Batch last(width - 1, 1, width - 1, width, 1);
        check(last.key_count() == width && last.candidate_stride() == 2048 &&
              last.candidate_entries() == 2048, "candidate stride capped at 2048 blocks");
    }
    const Source20Batch maximum(0, 1048576, 0, 1048576, 1048576);
    check(maximum.candidate_entries() == size_t(1048576) * 2048,
          "maximum plan reports size without allocating a quadratic mask");
}
void candidates() {
    const float inf = std::numeric_limits<float>::infinity();
    const Source20Batch batch(0, 17, 0, 17, 17);
    std::vector<float> logits(17, -inf); logits[0] = -3;
    const std::vector<int32_t> first{0, -1, -1};
    validate_source20_candidates(batch, 0, logits, first);
    check(true, "first query keeps its whole partial block with -1 row padding");
    auto reject_row = [&](const std::string& name, const std::vector<int32_t>& ids) {
        rejects(name, [&] { validate_source20_candidates(batch, 0, logits, ids); });
    };
    reject_row("missing newest block", {-1, -1, -1});
    reject_row("duplicate block", {0, 0, -1});
    reject_row("future block", {0, 1, -1});
    reject_row("invalid negative block", {-2, 0, -1});
    reject_row("ID after padding", {-1, 0, -1});
    reject_row("wrong candidate stride", {0});
    rejects("wrong logits width", [&] { validate_source20_candidates(batch, 0, std::span(logits).first(16), first); });
    for (float wrong : {0.f, inf, std::numeric_limits<float>::quiet_NaN()}) {
        logits[1] = wrong;
        rejects("future score must already be -inf", [&] { validate_source20_candidates(batch, 0, logits, first); });
    }
    logits[1] = -inf;
    for (float wrong : {inf, std::numeric_limits<float>::quiet_NaN()}) {
        logits[0] = wrong;
        rejects("nonfinite reachable score rejected", [&] { validate_source20_candidates(batch, 0, logits, first); });
    }
    logits[0] = -inf;
    validate_source20_candidates(batch, 0, logits, first);
    check(true, "newest block pinned even when all original scores are -inf");
    logits[0] = 1.f;
    const std::vector<int32_t> holes{0, 2, -1};
    validate_source20_candidates(batch, 16, logits, holes);
    check(true, "unreachable interior block excluded and empty newest block pinned");
    const std::vector<int32_t> unreachable{0, 1, 2};
    rejects("cannot select an unreachable interior block", [&] {
        validate_source20_candidates(batch, 16, logits, unreachable);
    });
    const std::vector<int32_t> unordered{2, 0, -1};
    rejects("candidate IDs must be sorted by original position", [&] {
        validate_source20_candidates(batch, 16, logits, unordered);
    });
    const Source20Batch large(16384, 1, 16384, 16385, 1);
    logits.resize(16385);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = float(logits.size() - i);
    std::vector<int32_t> top(2048); std::iota(top.begin(), top.end(), 0); top.back() = 2048;
    validate_source20_candidates(large, 0, logits, top);
    check(true, "2049 blocks prune one older block and retain the lowest-scoring newest block");
    std::iota(top.begin(), top.end(), 1);
    rejects("higher omitted score invalidates selection", [&] { validate_source20_candidates(large, 0, logits, top); });
    std::fill(logits.begin(), logits.end(), 0.f);
    validate_source20_candidates(large, 0, logits, top);
    check(true, "cutoff ties accept any valid subset rather than torch-specific tie IDs");
}
uint32_t number(const GgufModel& model, const std::string& name) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::U32) throw std::runtime_error("Missing fixture integer " + name);
    return uint32_t(value->u);
}
void string_is(const GgufModel& model, const std::string& name, const std::string& expected) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::STRING || value->s != expected)
        throw std::runtime_error("Wrong fixture identity " + name);
}
template<class T>
std::span<const T> tensor(const GgufModel& model, const std::string& name,
                          uint32_t type, const std::vector<uint64_t>& shape) {
    size_t shard = 0;
    const auto* info = model.find(name, &shard);
    if (!info || info->type != type || info->shape != shape || !model.in_bounds(*info, shard))
        throw std::runtime_error("Malformed fixture tensor " + name);
    const uint8_t* data = model.shard(shard).tensor_data(*info);
    if (uintptr_t(data) % alignof(T)) throw std::runtime_error("Misaligned fixture tensor");
    return {reinterpret_cast<const T*>(data), size_t(tensor_payload_bytes(*info) / sizeof(T))};
}
bool finite_bf16(std::span<const uint16_t> values) {
    return std::all_of(values.begin(), values.end(), [](uint16_t value) {
        return std::isfinite(std::bit_cast<float>(uint32_t(value) << 16));
    });
}
void real(const std::string& path) {
    const auto model = GgufModel::open(path);
    string_is(model, "general.architecture", "deepseek41-source20-contract-fixture");
    string_is(model, "strata.deepseek41.revision", "1bae0e1088993b9b82de08f59de5be5a34e42170");
    string_is(model, "strata.deepseek41.official_source_sha256", "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d");
    string_is(model, "fixture.candidate_encoding", "sorted-i32-little-endian-bytes-minus-one-padding");
    const auto w = read_source20_compressor_weights(model, geometry());
    check(finite_bf16(w.kv) && finite_bf16(w.norm), "real source-20 BF16 weights finite and mapped");
    const int plans = int(number(model, "fixture.plans"));
    if (plans != 9) throw std::runtime_error("Incomplete compressor reference plans");
    int calls = 0;
    bool norm_changes_projection = false;
    for (int plan = 0; plan < plans; ++plan) {
        const std::string prefix = "plan." + std::to_string(plan) + ".";
        const int tokens = int(number(model, prefix + "tokens")), prefill = int(number(model, prefix + "prefill"));
        if (tokens < 1 || tokens > 17 || prefill < 1 || prefill > tokens)
            throw std::runtime_error("Invalid reference plan bounds");
        check(finite_bf16(tensor<uint16_t>(model, prefix + "input", 30, {5120, uint64_t(tokens), 1})),
              "reference input has exact BF16 shape and finite values");
        for (int start = 0; start < tokens;) {
            const int count = start ? 1 : prefill;
            const Source20Batch batch(start, count, start, tokens, prefill);
            const std::string call = prefix + "at." + std::to_string(start) + ".";
            const std::vector<uint64_t> shape{512, uint64_t(batch.latent_count()), 1};
            const auto projection = tensor<uint16_t>(model, call + "projection", 30, shape);
            const auto output = tensor<uint16_t>(model, call + "output", 30, shape);
            check(finite_bf16(projection) && finite_bf16(output) && batch.position(count - 1) == start + count - 1,
                  "official BF16 projection/output obey one-latent-per-token interface");
            norm_changes_projection |= !std::equal(projection.begin(), projection.end(), output.begin());
            ++calls; start += count;
        }
    }
    check(norm_changes_projection, "real ratio-1 output includes RMSNorm rather than returning projection unchanged");
    check(calls == 58, "all pinned compressor calls consumed");
    const int rows = int(number(model, "fixture.candidate_rows"));
    if (rows != 20) throw std::runtime_error("Incomplete candidate reference rows");
    int future_cases = 0, pruned_cases = 0, max_context_cases = 0;
    for (int row = 0; row < rows; ++row) {
        const std::string prefix = "candidate." + std::to_string(row) + ".";
        const int width = int(number(model, prefix + "width")), visible = int(number(model, prefix + "visible"));
        const bool decode = number(model, prefix + "decode") != 0;
        if (width < 1 || width > Source20Batch::max_context || visible < 1 || visible > width || (decode && visible != width))
            throw std::runtime_error("Invalid reference candidate bounds");
        const int start = decode ? width - 1 : 0;
        const Source20Batch batch(start, decode ? 1 : width, start, width, width);
        const int query = decode ? 0 : visible - 1;
        const auto scores = tensor<float>(model, prefix + "scores", 0, {uint64_t(width)});
        const auto mask = tensor<uint8_t>(model, prefix + "mask", 24, {uint64_t(width)});
        const auto bytes = tensor<uint8_t>(model, prefix + "blocks", 24, {uint64_t(batch.candidate_stride()) * 4});
        std::vector<int32_t> ids(size_t(batch.candidate_stride()));
        for (size_t i = 0; i < ids.size(); ++i) {
            const size_t p = 4 * i;
            ids[i] = std::bit_cast<int32_t>(uint32_t(bytes[p]) | (uint32_t(bytes[p + 1]) << 8) |
                                           (uint32_t(bytes[p + 2]) << 16) | (uint32_t(bytes[p + 3]) << 24));
        }
        validate_source20_candidates(batch, query, scores, ids);
        check(true, "official candidate row satisfies independent block-max/cutoff contract " + std::to_string(row));
        const auto end = std::find(ids.begin(), ids.end(), -1);
        bool exact = true, future = false;
        for (int key = 0; key < width; ++key) {
            const bool selected = std::binary_search(ids.begin(), end, key / 8);
            exact &= mask[size_t(key)] <= 1 && selected == (mask[size_t(key)] != 0);
            future |= key >= visible && selected;
        }
        check(exact, "packed IDs reconstruct every official mask bit including partial future positions");
        future_cases += future;
        pruned_cases += (width + 7) / 8 > 2048 && end - ids.begin() == 2048;
        max_context_cases += width == Source20Batch::max_context;
    }
    check(future_cases > 0, "candidate mask and causal mask remain separate contracts");
    check(pruned_cases >= 4, "real 2048-block cap exercised with actual pruning");
    check(max_context_cases == 1, "maximum context tested one row at a time without quadratic allocation");
}
} // namespace

int main(int argc, char** argv) {
    const auto dir = fs::temp_directory_path() /
        ("strata-source20-contract-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        if (argc == 1) {
            fs::create_directories(dir);
            weights(dir); batches(); candidates();
        } else if (argc == 3 && std::string(argv[1]) == "--real") {
            real(argv[2]);
        } else {
            throw std::invalid_argument("usage: deepseek41_source20_contract_test [--real FIXTURE]");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what()); ++failures;
    }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
