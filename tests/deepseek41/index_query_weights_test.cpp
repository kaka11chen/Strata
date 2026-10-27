#include "strata/deepseek41/index_query_weights.hpp"
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
    const auto dir = fs::temp_directory_path() / ("strata-index-query-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
        const std::vector<fixture::Kv> meta{fixture::str("strata.deepseek41.index_query_profile", "1017-layer2-index32x128-fp8-block32")};
        const std::vector<fixture::Tensor> tensors{{"blk.2.attn_indexer_wq_b.weight", {1280, 4096}, 24, 1},
            {"blk.2.attn_indexer_wq_b.scale", {40, 128}, 24, 2},
            {"blk.2.attn_indexer_weights_proj.weight", {5120, 32}, 30, 3}};
        auto read = [&](const std::vector<fixture::Kv>& metadata, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
            const auto path = dir / (std::to_string(checks) + ".gguf");
            fixture::write(path, metadata, ts, cut); return GgufModel::open(path.string());
        };
        const auto model = read(meta, tensors);
        const auto w = read_index_query_weights(model, g, 2);
        check(w.query.rows == 4096 && w.query.cols == 1280 && w.query.values.size() == 4096 * 1280 && w.query.scales.size() == 128 * 40 && w.head_weights.size() == 32 * 5120,
              "typed mapped views preserve fixed index query geometry");
        bool exact = true; int seed = 0;
        for (auto view : {w.query.values, w.query.scales, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(w.head_weights.data()), w.head_weights.size_bytes())}) {
            ++seed; const auto* bytes = reinterpret_cast<const uint8_t*>(view.data());
            for (size_t i = 0; i < view.size_bytes(); ++i) exact &= bytes[i] == fixture::pattern(uint8_t(seed), i);
        }
        check(exact, "all original FP8/scale/BF16 weight bytes preserved");
        rejects("missing profile rejected", [&] { const auto m = read({}, tensors); read_index_query_weights(m, g, 2); });
        rejects("wrong profile rejected", [&] { const auto m = read({fixture::str("strata.deepseek41.index_query_profile", "other")}, tensors); read_index_query_weights(m, g, 2); });
        rejects("wrong profile type rejected", [&] { const auto m = read({fixture::u32("strata.deepseek41.index_query_profile", 2)}, tensors); read_index_query_weights(m, g, 2); });
        for (size_t i = 0; i < tensors.size(); ++i) {
            rejects("missing tensor rejected", [&] { auto ts = tensors; ts.erase(ts.begin() + i); const auto m = read(meta, ts); read_index_query_weights(m, g, 2); });
            rejects("wrong tensor dtype rejected", [&] { auto ts = tensors; ts[i].type = 0; const auto m = read(meta, ts); read_index_query_weights(m, g, 2); });
        }
        rejects("transposed projection rejected", [&] { auto ts = tensors; ts[0].shape = {4096, 1280}; const auto m = read(meta, ts); read_index_query_weights(m, g, 2); });
        rejects("wrong scale shape rejected", [&] { auto ts = tensors; ts[1].shape = {128, 40}; const auto m = read(meta, ts); read_index_query_weights(m, g, 2); });
        rejects("truncated head weights rejected", [&] { const auto m = read(meta, tensors, 1); read_index_query_weights(m, g, 2); });
        rejects("other layer rejected", [&] { read_index_query_weights(model, g, 8); });
        rejects("wrong hidden width rejected", [&] { auto bad = g; bad.hidden = 4096; read_index_query_weights(model, bad, 2); });
        rejects("wrong layer count rejected", [&] { auto bad = g; bad.layers = 3; read_index_query_weights(model, bad, 2); });
        rejects("missing ratio list rejected", [&] { auto bad = g; bad.compress_ratios.clear(); read_index_query_weights(model, bad, 2); });
        rejects("ratio 1 is not ratio 2", [&] { auto bad = g; bad.compress_ratios[2] = 1; read_index_query_weights(model, bad, 2); });
        for (int layer : {8,14}) {
            auto source_geometry=g;source_geometry.compress_ratios[size_t(layer)]=2;source_geometry.sources.resize(40);source_geometry.sources[size_t(layer)]={layer,layer,-1};
            const std::vector<fixture::Kv> source_meta{fixture::str("strata.deepseek41.index_query_profile","1017-layer"+std::to_string(layer)+"-index32x128-fp8-block32")};
            auto source_tensors=tensors;for(auto& t:source_tensors)t.name.replace(0,5,"blk."+std::to_string(layer));
            const auto source_model=read(source_meta,source_tensors);const auto v=read_index_query_weights(source_model,source_geometry,layer);
            check(v.layer == layer && v.query.rows == 4096 && v.head_weights.size() == 32*5120,"original source identity and fixed shapes");
            rejects("original source tensor names must match",[&]{const auto bad=read(source_meta,tensors);read_index_query_weights(bad,source_geometry,layer);});
            rejects("original source ratio must be two",[&]{auto bad=source_geometry;bad.compress_ratios[size_t(layer)]=1;read_index_query_weights(source_model,bad,layer);});
            rejects("original source must own KV",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].kv=2;read_index_query_weights(source_model,bad,layer);});
            rejects("original source must own index",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].index=2;read_index_query_weights(source_model,bad,layer);});
            rejects("original source cannot use candidates",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].candidates=2;read_index_query_weights(source_model,bad,layer);});
            rejects("original source missing topology",[&]{auto bad=source_geometry;bad.sources.clear();read_index_query_weights(source_model,bad,layer);});
            rejects("original source missing ratios",[&]{auto bad=source_geometry;bad.compress_ratios.clear();read_index_query_weights(source_model,bad,layer);});
            rejects("original source truncated payload",[&]{const auto bad=read(source_meta,source_tensors,1);read_index_query_weights(bad,source_geometry,layer);});
            for (int other : {2,8,14}) if (other != layer)
                rejects("source profile cannot be borrowed by another owner",[&]{read_index_query_weights(source_model,source_geometry,other);});
        }
        {
            auto source = g; source.compress_ratios[20] = 1; source.sources.resize(40); source.sources[20] = {20,20,-1};
            const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.index_query_profile", "1017-layer20-index32x128-fp8-block32")};
            auto ts = tensors;
            for (auto& t : ts) t.name.replace(0, 5, "blk.20");
            const auto m = read(metadata, ts);
            const auto key = read_source20_index_query_weights(m, source);
            check(key.query.rows == 4096 && key.query.cols == 1280 && key.query.values.size() == 4096*1280 && key.query.scales.size() == 128*40 && key.head_weights.size() == 32*5120, "source-20 typed original query geometry");
            bool preserved = true; int original_seed = 0;
            for (auto view : {key.query.values, key.query.scales, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(key.head_weights.data()), key.head_weights.size_bytes())}) {
                ++original_seed; const auto* bytes = reinterpret_cast<const uint8_t*>(view.data());
                for (size_t i = 0; i < view.size_bytes(); ++i) preserved &= bytes[i] == fixture::pattern(uint8_t(original_seed), i);
            }
            check(preserved, "source-20 original FP8/scale/BF16 bytes unchanged");
            rejects("source-20 missing profile", [&] { const auto bad=read({},ts); read_source20_index_query_weights(bad,source); });
            rejects("source-20 wrong profile type", [&] { const auto bad=read({fixture::u32("strata.deepseek41.index_query_profile",20)},ts); read_source20_index_query_weights(bad,source); });
            rejects("source-20 cannot borrow ratio-2 profile", [&] { const auto bad=read(meta,ts); read_source20_index_query_weights(bad,source); });
            rejects("source-20 cannot borrow ratio-2 tensors", [&] { const auto bad=read(metadata,tensors); read_source20_index_query_weights(bad,source); });
            rejects("ratio-2 reader cannot consume source-20", [&] { read_index_query_weights(m,source,20); });
            rejects("source-20 wrong hidden width", [&] { auto bad=source;bad.hidden=4096;read_source20_index_query_weights(m,bad); });
            rejects("source-20 wrong layer count", [&] { auto bad=source;bad.layers=39;read_source20_index_query_weights(m,bad); });
            rejects("source-20 missing ratios", [&] { auto bad=source;bad.compress_ratios.clear();read_source20_index_query_weights(m,bad); });
            rejects("source-20 cannot use ratio 2", [&] { auto bad=source;bad.compress_ratios[20]=2;read_source20_index_query_weights(m,bad); });
            rejects("source-20 missing topology", [&] { auto bad=source;bad.sources.clear();read_source20_index_query_weights(m,bad); });
            rejects("source-20 must own KV", [&] { auto bad=source;bad.sources[20].kv=14;read_source20_index_query_weights(m,bad); });
            rejects("source-20 must own index", [&] { auto bad=source;bad.sources[20].index=14;read_source20_index_query_weights(m,bad); });
            rejects("source-20 cannot consume candidates", [&] { auto bad=source;bad.sources[20].candidates=20;read_source20_index_query_weights(m,bad); });
            for (size_t i=0;i<ts.size();++i) {
                rejects("source-20 missing tensor", [&] { auto bad=ts;bad.erase(bad.begin()+i);const auto file=read(metadata,bad);read_source20_index_query_weights(file,source); });
                rejects("source-20 wrong dtype", [&] { auto bad=ts;bad[i].type=0;const auto file=read(metadata,bad);read_source20_index_query_weights(file,source); });
            }
            rejects("source-20 transposed projection", [&] { auto bad=ts;bad[0].shape={4096,1280};const auto file=read(metadata,bad);read_source20_index_query_weights(file,source); });
            rejects("source-20 wrong scale shape", [&] { auto bad=ts;bad[1].shape={128,40};const auto file=read(metadata,bad);read_source20_index_query_weights(file,source); });
            rejects("source-20 truncated payload", [&] { const auto bad=read(metadata,ts,1);read_source20_index_query_weights(bad,source); });
        }
        for(int owner:{24,28,32,36}) {
            auto source=g;source.compress_ratios[20]=source.compress_ratios[size_t(owner)]=1;source.sources.resize(40);
            source.sources[20]={20,20,-1};source.sources[size_t(owner)]={20,owner,20};
            const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.index_query_profile","1017-layer"+std::to_string(owner)+"-index32x128-fp8-block32")};
            auto ts=tensors;for(auto& t:ts)t.name.replace(0,5,"blk."+std::to_string(owner));
            const auto m=read(metadata,ts);const auto q=read_candidate_index_query_weights(m,source,owner);
            check(q.layer==owner && q.query.rows==4096 && q.query.cols==1280 && q.head_weights.size()==32*5120,"candidate source original query geometry");
            bool preserved=true;int seed=0;
            for(auto view:{q.query.values,q.query.scales,std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(q.head_weights.data()),q.head_weights.size_bytes())}) {
                ++seed;for(size_t i=0;i<view.size();++i)preserved &= view[i]==fixture::pattern(uint8_t(seed),i);
            }
            check(preserved,"candidate source original FP8/scale/BF16 bytes preserved");
            for(int layer:{20,25,40,-1})rejects("unsupported original candidate source is rejected",[&]{read_candidate_index_query_weights(m,source,layer);});
            for(int other:{24,28,32,36})if(other!=owner)rejects("only matching original candidate source is accepted",[&]{read_candidate_index_query_weights(m,source,other);});
            rejects("source20 reader rejects candidate source",[&]{read_source20_index_query_weights(m,source);});
            rejects("ratio2 reader rejects candidate source",[&]{read_index_query_weights(m,source,owner);});
            rejects("missing candidate profile",[&]{const auto bad=read({},ts);read_candidate_index_query_weights(bad,source,owner);});
            rejects("wrong profile type",[&]{const auto bad=read({fixture::u32("strata.deepseek41.index_query_profile",24)},ts);read_candidate_index_query_weights(bad,source,owner);});
            rejects("wrong original tensor names",[&]{const auto bad=read(metadata,tensors);read_candidate_index_query_weights(bad,source,owner);});
            for(int layer:{20,owner}) {
                rejects("candidate source ratio must be one",[&]{auto bad=source;bad.compress_ratios[size_t(layer)]=2;read_candidate_index_query_weights(m,bad,owner);});
                rejects("candidate source KV topology",[&]{auto bad=source;bad.sources[size_t(layer)].kv=14;read_candidate_index_query_weights(m,bad,owner);});
                rejects("candidate source index topology",[&]{auto bad=source;bad.sources[size_t(layer)].index=14;read_candidate_index_query_weights(m,bad,owner);});
                rejects("candidate source candidate topology",[&]{auto bad=source;bad.sources[size_t(layer)].candidates=14;read_candidate_index_query_weights(m,bad,owner);});
            }
            rejects("missing topology",[&]{auto bad=source;bad.sources.clear();read_candidate_index_query_weights(m,bad,owner);});
            rejects("missing ratios",[&]{auto bad=source;bad.compress_ratios.clear();read_candidate_index_query_weights(m,bad,owner);});
            for(size_t i=0;i<ts.size();++i){
                rejects("candidate missing tensor",[&]{auto bad=ts;bad.erase(bad.begin()+i);const auto file=read(metadata,bad);read_candidate_index_query_weights(file,source,owner);});
                rejects("candidate wrong dtype",[&]{auto bad=ts;bad[i].type=0;const auto file=read(metadata,bad);read_candidate_index_query_weights(file,source,owner);});
            }
            rejects("candidate transposed projection",[&]{auto bad=ts;bad[0].shape={4096,1280};const auto file=read(metadata,bad);read_candidate_index_query_weights(file,source,owner);});
            rejects("candidate wrong scale shape",[&]{auto bad=ts;bad[1].shape={128,40};const auto file=read(metadata,bad);read_candidate_index_query_weights(file,source,owner);});
            rejects("candidate truncated payload",[&]{const auto bad=read(metadata,ts,1);read_candidate_index_query_weights(bad,source,owner);});
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
