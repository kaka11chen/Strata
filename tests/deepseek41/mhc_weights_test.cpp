#include "strata/deepseek41/mhc_weights.hpp"
#include "../core/gguf_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
int checks = 0, failures = 0;
void check(bool ok, const char* name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); }
void rejects(const char* name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::exception&) { check(true, name); }
}
int main() {
    const auto dir = fs::temp_directory_path() / ("strata-mhc-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        Geometry g; g.layers = 1; g.hidden = 64;
        const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.mhc_profile", "1017-hc4-delayed")};
        const std::vector<fixture::Tensor> tensors{{"blk.0.hc_ffn_fn", {256, 24}, 0, 1},
            {"blk.0.hc_ffn_scale", {3}, 0, 2}, {"blk.0.hc_ffn_base", {24}, 0, 3},
            {"blk.0.ffn_norm.weight", {64}, 30, 4}};
        auto read = [&](const std::vector<fixture::Kv>& meta, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
            const auto path = dir / (std::to_string(checks) + ".gguf");
            fixture::write(path, meta, ts, cut);
            return GgufModel::open(path.string());
        };
        const auto model = read(metadata, tensors);
        const auto w = read_mhc_weights(model, g, 0, MhcBranch::ffn);
        check(w.projection.size() == 6144 && w.scale.size() == 3 && w.base.size() == 24 && w.norm.size() == 64,
              "typed mapped views have exact 1017 mHC shapes");
        const auto* raw = reinterpret_cast<const uint8_t*>(w.projection.data());
        bool exact = true;
        for (size_t i = 0; i < w.projection.size_bytes(); ++i) exact &= raw[i] == fixture::pattern(1, i);
        check(exact, "F32 projection bytes preserved without BF16 conversion");
        rejects("missing profile rejected", [&] { const auto m = read({}, tensors); read_mhc_weights(m, g, 0, MhcBranch::ffn); });
        rejects("wrong profile rejected", [&] { const auto m = read({fixture::str("strata.deepseek41.mhc_profile", "other")}, tensors); read_mhc_weights(m, g, 0, MhcBranch::ffn); });
        rejects("wrong projection dtype rejected", [&] { auto ts = tensors; ts[0].type = 30; const auto m = read(metadata, ts); read_mhc_weights(m, g, 0, MhcBranch::ffn); });
        rejects("wrong projection orientation rejected", [&] { auto ts = tensors; ts[0].shape = {24, 256}; const auto m = read(metadata, ts); read_mhc_weights(m, g, 0, MhcBranch::ffn); });
        rejects("missing norm rejected", [&] { auto ts = tensors; ts.pop_back(); const auto m = read(metadata, ts); read_mhc_weights(m, g, 0, MhcBranch::ffn); });
        rejects("truncated norm rejected", [&] { const auto m = read(metadata, tensors, 1); read_mhc_weights(m, g, 0, MhcBranch::ffn); });
        rejects("absent attention weights rejected", [&] { read_mhc_weights(model, g, 0, MhcBranch::attention); });
        rejects("invalid branch rejected", [&] { read_mhc_weights(model, g, 0, static_cast<MhcBranch>(99)); });
        rejects("invalid layer rejected", [&] { read_mhc_weights(model, g, 1, MhcBranch::ffn); });
        rejects("invalid geometry rejected", [&] { auto bad = g; bad.hidden = 0; read_mhc_weights(model, bad, 0, MhcBranch::ffn); });
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
