#include "strata/deepseek41/index_key_weights.hpp"
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
    const auto dir = fs::temp_directory_path() / ("strata-index-key-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
        const std::vector<fixture::Kv> meta{fixture::str("strata.deepseek41.index_key_profile", "1017-layer2-index-key128-bf16")};
        const std::vector<fixture::Tensor> tensors{{"blk.2.attn_indexer_wk.weight", {512, 128}, 30, 1},
            {"blk.2.attn_indexer_k_norm.weight", {128}, 30, 2}};
        auto read = [&](const std::vector<fixture::Kv>& metadata, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
            const auto path = dir / (std::to_string(checks) + ".gguf");
            fixture::write(path, metadata, ts, cut); return GgufModel::open(path.string());
        };
        const auto model = read(meta, tensors);
        const auto w = read_index_key_weights(model, g, 2);
        check(w.projection.size() == 512 * 128 && w.norm.size() == 128,
              "typed mapped views preserve fixed index key geometry");
        bool exact = true; int seed = 0;
        for (auto view : {w.projection, w.norm}) {
            ++seed; const auto* bytes = reinterpret_cast<const uint8_t*>(view.data());
            for (size_t i = 0; i < view.size_bytes(); ++i) exact &= bytes[i] == fixture::pattern(uint8_t(seed), i);
        }
        check(exact, "all original BF16 weight bytes preserved");
        rejects("missing profile rejected", [&] { const auto m = read({}, tensors); read_index_key_weights(m, g, 2); });
        rejects("wrong profile rejected", [&] { const auto m = read({fixture::str("strata.deepseek41.index_key_profile", "other")}, tensors); read_index_key_weights(m, g, 2); });
        rejects("wrong profile type rejected", [&] { const auto m = read({fixture::u32("strata.deepseek41.index_key_profile", 2)}, tensors); read_index_key_weights(m, g, 2); });
        for (size_t i = 0; i < tensors.size(); ++i) {
            rejects("missing tensor rejected", [&] { auto ts = tensors; ts.erase(ts.begin() + i); const auto m = read(meta, ts); read_index_key_weights(m, g, 2); });
            rejects("wrong tensor dtype rejected", [&] { auto ts = tensors; ts[i].type = 0; const auto m = read(meta, ts); read_index_key_weights(m, g, 2); });
        }
        rejects("transposed projection rejected", [&] { auto ts = tensors; ts[0].shape = {128, 512}; const auto m = read(meta, ts); read_index_key_weights(m, g, 2); });
        rejects("wrong norm width rejected", [&] { auto ts = tensors; ts[1].shape = {127}; const auto m = read(meta, ts); read_index_key_weights(m, g, 2); });
        rejects("truncated norm rejected", [&] { const auto m = read(meta, tensors, 1); read_index_key_weights(m, g, 2); });
        rejects("other layer rejected", [&] { read_index_key_weights(model, g, 8); });
        rejects("wrong hidden width rejected", [&] { auto bad = g; bad.hidden = 4096; read_index_key_weights(model, bad, 2); });
        rejects("wrong layer count rejected", [&] { auto bad = g; bad.layers = 3; read_index_key_weights(model, bad, 2); });
        rejects("missing ratio list rejected", [&] { auto bad = g; bad.compress_ratios.clear(); read_index_key_weights(model, bad, 2); });
        rejects("ratio 1 is not ratio 2", [&] { auto bad = g; bad.compress_ratios[2] = 1; read_index_key_weights(model, bad, 2); });
        for (int layer : {8,14}) {
            auto source_geometry=g;source_geometry.compress_ratios[size_t(layer)]=2;source_geometry.sources.resize(40);source_geometry.sources[size_t(layer)]={layer,layer,-1};
            const std::vector<fixture::Kv> source_meta{fixture::str("strata.deepseek41.index_key_profile","1017-layer"+std::to_string(layer)+"-index-key128-bf16")};
            auto source_tensors=tensors;for(auto& t:source_tensors)t.name.replace(0,5,"blk."+std::to_string(layer));
            const auto source_model=read(source_meta,source_tensors);const auto v=read_index_key_weights(source_model,source_geometry,layer);
            check(v.layer == layer && v.projection.size() == 512*128 && v.norm.size() == 128,"original source identity and fixed shapes");
            rejects("original source tensor names must match",[&]{const auto bad=read(source_meta,tensors);read_index_key_weights(bad,source_geometry,layer);});
            rejects("original source ratio must be two",[&]{auto bad=source_geometry;bad.compress_ratios[size_t(layer)]=1;read_index_key_weights(source_model,bad,layer);});
            rejects("original source must own KV",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].kv=2;read_index_key_weights(source_model,bad,layer);});
            rejects("original source must own index",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].index=2;read_index_key_weights(source_model,bad,layer);});
            rejects("original source cannot use candidates",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].candidates=2;read_index_key_weights(source_model,bad,layer);});
            rejects("original source missing topology",[&]{auto bad=source_geometry;bad.sources.clear();read_index_key_weights(source_model,bad,layer);});
            rejects("original source missing ratios",[&]{auto bad=source_geometry;bad.compress_ratios.clear();read_index_key_weights(source_model,bad,layer);});
            rejects("original source truncated payload",[&]{const auto bad=read(source_meta,source_tensors,1);read_index_key_weights(bad,source_geometry,layer);});
            for (int other : {2,8,14}) if (other != layer)
                rejects("source profile cannot be borrowed by another owner",[&]{read_index_key_weights(source_model,source_geometry,other);});
        }
        {
            auto source = g; source.compress_ratios[20] = 1; source.sources.resize(40); source.sources[20] = {20,20,-1};
            const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.index_key_profile", "1017-layer20-index-key128-bf16")};
            auto ts = tensors;
            for (auto& t : ts) t.name.replace(0, 5, "blk.20");
            const auto m = read(metadata, ts);
            const auto key = read_source20_index_key_weights(m, source);
            check(key.projection.size() == 512*128 && key.norm.size() == 128, "source-20 typed original key geometry");
            bool preserved = true; int original_seed = 0;
            for (auto view : {key.projection, key.norm}) {
                ++original_seed; const auto* bytes = reinterpret_cast<const uint8_t*>(view.data());
                for (size_t i = 0; i < view.size_bytes(); ++i) preserved &= bytes[i] == fixture::pattern(uint8_t(original_seed), i);
            }
            check(preserved, "source-20 all mapped BF16 bytes unchanged");
            rejects("source-20 missing profile", [&] { const auto bad=read({},ts); read_source20_index_key_weights(bad,source); });
            rejects("source-20 wrong profile type", [&] { const auto bad=read({fixture::u32("strata.deepseek41.index_key_profile",20)},ts); read_source20_index_key_weights(bad,source); });
            rejects("source-20 cannot borrow ratio-2 profile", [&] { const auto bad=read(meta,ts); read_source20_index_key_weights(bad,source); });
            rejects("source-20 cannot borrow ratio-2 tensors", [&] { const auto bad=read(metadata,tensors); read_source20_index_key_weights(bad,source); });
            rejects("ratio-2 reader cannot consume source-20", [&] { read_index_key_weights(m,source,20); });
            rejects("source-20 wrong hidden width", [&] { auto bad=source;bad.hidden=4096;read_source20_index_key_weights(m,bad); });
            rejects("source-20 wrong layer count", [&] { auto bad=source;bad.layers=39;read_source20_index_key_weights(m,bad); });
            rejects("source-20 missing ratios", [&] { auto bad=source;bad.compress_ratios.clear();read_source20_index_key_weights(m,bad); });
            rejects("source-20 cannot use ratio 2", [&] { auto bad=source;bad.compress_ratios[20]=2;read_source20_index_key_weights(m,bad); });
            rejects("source-20 missing topology", [&] { auto bad=source;bad.sources.clear();read_source20_index_key_weights(m,bad); });
            rejects("source-20 must own KV", [&] { auto bad=source;bad.sources[20].kv=14;read_source20_index_key_weights(m,bad); });
            rejects("source-20 must own index", [&] { auto bad=source;bad.sources[20].index=14;read_source20_index_key_weights(m,bad); });
            rejects("source-20 cannot consume candidates", [&] { auto bad=source;bad.sources[20].candidates=20;read_source20_index_key_weights(m,bad); });
            for (size_t i=0;i<ts.size();++i) {
                rejects("source-20 missing tensor", [&] { auto bad=ts;bad.erase(bad.begin()+i);const auto file=read(metadata,bad);read_source20_index_key_weights(file,source); });
                rejects("source-20 wrong dtype", [&] { auto bad=ts;bad[i].type=0;const auto file=read(metadata,bad);read_source20_index_key_weights(file,source); });
            }
            rejects("source-20 transposed projection", [&] { auto bad=ts;bad[0].shape={128,512};const auto file=read(metadata,bad);read_source20_index_key_weights(file,source); });
            rejects("source-20 wrong norm width", [&] { auto bad=ts;bad[1].shape={127};const auto file=read(metadata,bad);read_source20_index_key_weights(file,source); });
            rejects("source-20 truncated payload", [&] { const auto bad=read(metadata,ts,1);read_source20_index_key_weights(bad,source); });
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
