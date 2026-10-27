#include "strata/deepseek41/compressor_weights.hpp"
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
    const auto dir = fs::temp_directory_path() / ("strata-compressor-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
        const std::vector<fixture::Kv> meta{fixture::str("strata.deepseek41.compressor_profile", "1017-layer2-ratio2-bf16")};
        const std::vector<fixture::Tensor> tensors{{"blk.2.attn_compressor_wkv.weight", {5120, 512}, 30, 1},
            {"blk.2.attn_compressor_wgate.weight", {5120, 512}, 30, 2}, {"blk.2.attn_compressor_norm.weight", {512}, 30, 3}};
        auto read = [&](const std::vector<fixture::Kv>& metadata, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
            const auto path = dir / (std::to_string(checks) + ".gguf");
            fixture::write(path, metadata, ts, cut); return GgufModel::open(path.string());
        };
        const auto model = read(meta, tensors);
        const auto w = read_compressor_weights(model, g, 2);
        check(w.kv.size() == 5120 * 512 && w.gate.size() == 5120 * 512 && w.norm.size() == 512,
              "typed mapped views preserve fixed ratio-2 geometry");
        bool exact = true; int seed = 0;
        for (auto view : {w.kv, w.gate, w.norm}) {
            ++seed; const auto* bytes = reinterpret_cast<const uint8_t*>(view.data());
            for (size_t i = 0; i < view.size_bytes(); ++i) exact &= bytes[i] == fixture::pattern(uint8_t(seed), i);
        }
        check(exact, "all original BF16 weight bytes preserved");
        rejects("missing profile rejected", [&] { const auto m = read({}, tensors); read_compressor_weights(m, g, 2); });
        rejects("wrong profile rejected", [&] { const auto m = read({fixture::str("strata.deepseek41.compressor_profile", "other")}, tensors); read_compressor_weights(m, g, 2); });
        rejects("wrong profile type rejected", [&] { const auto m = read({fixture::u32("strata.deepseek41.compressor_profile", 2)}, tensors); read_compressor_weights(m, g, 2); });
        for (size_t i = 0; i < tensors.size(); ++i) {
            rejects("missing tensor rejected", [&] { auto ts = tensors; ts.erase(ts.begin() + i); const auto m = read(meta, ts); read_compressor_weights(m, g, 2); });
            rejects("wrong tensor dtype rejected", [&] { auto ts = tensors; ts[i].type = 0; const auto m = read(meta, ts); read_compressor_weights(m, g, 2); });
        }
        rejects("transposed projection rejected", [&] { auto ts = tensors; ts[0].shape = {512, 5120}; const auto m = read(meta, ts); read_compressor_weights(m, g, 2); });
        rejects("wrong norm width rejected", [&] { auto ts = tensors; ts[2].shape = {511}; const auto m = read(meta, ts); read_compressor_weights(m, g, 2); });
        rejects("truncated norm rejected", [&] { const auto m = read(meta, tensors, 1); read_compressor_weights(m, g, 2); });
        rejects("other layer rejected", [&] { read_compressor_weights(model, g, 8); });
        rejects("wrong hidden width rejected", [&] { auto bad = g; bad.hidden = 4096; read_compressor_weights(model, bad, 2); });
        rejects("wrong layer count rejected", [&] { auto bad = g; bad.layers = 3; read_compressor_weights(model, bad, 2); });
        rejects("missing ratio list rejected", [&] { auto bad = g; bad.compress_ratios.clear(); read_compressor_weights(model, bad, 2); });
        rejects("ratio 1 is not ratio 2", [&] { auto bad = g; bad.compress_ratios[2] = 1; read_compressor_weights(model, bad, 2); });
        for (int layer : {8,14}) {
            auto source_geometry=g;source_geometry.compress_ratios[size_t(layer)]=2;source_geometry.sources.resize(40);source_geometry.sources[size_t(layer)]={layer,layer,-1};
            const std::vector<fixture::Kv> source_meta{fixture::str("strata.deepseek41.compressor_profile","1017-layer"+std::to_string(layer)+"-ratio2-bf16")};
            auto source_tensors=tensors;for(auto& t:source_tensors)t.name.replace(0,5,"blk."+std::to_string(layer));
            const auto source_model=read(source_meta,source_tensors);const auto v=read_compressor_weights(source_model,source_geometry,layer);
            check(v.layer == layer && v.kv.size() == 5120*512 && v.norm.size() == 512,"original source identity and fixed shapes");
            rejects("original source tensor names must match",[&]{const auto bad=read(source_meta,tensors);read_compressor_weights(bad,source_geometry,layer);});
            rejects("original source ratio must be two",[&]{auto bad=source_geometry;bad.compress_ratios[size_t(layer)]=1;read_compressor_weights(source_model,bad,layer);});
            rejects("original source must own KV",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].kv=2;read_compressor_weights(source_model,bad,layer);});
            rejects("original source must own index",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].index=2;read_compressor_weights(source_model,bad,layer);});
            rejects("original source cannot use candidates",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].candidates=2;read_compressor_weights(source_model,bad,layer);});
            rejects("original source missing topology",[&]{auto bad=source_geometry;bad.sources.clear();read_compressor_weights(source_model,bad,layer);});
            rejects("original source missing ratios",[&]{auto bad=source_geometry;bad.compress_ratios.clear();read_compressor_weights(source_model,bad,layer);});
            rejects("original source truncated payload",[&]{const auto bad=read(source_meta,source_tensors,1);read_compressor_weights(bad,source_geometry,layer);});
            for (int other : {2,8,14}) if (other != layer)
                rejects("source profile cannot be borrowed by another owner",[&]{read_compressor_weights(source_model,source_geometry,other);});
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
