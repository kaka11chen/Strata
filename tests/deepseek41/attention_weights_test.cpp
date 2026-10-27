#include "strata/deepseek41/attention_weights.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "../core/gguf_fixture.hpp"

#include <chrono>
#include <cstdio>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
int checks = 0, failures = 0;
void check(bool ok, const char* name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); }
void rejects(const char* name, const std::function<void()>& fn) {
    try { fn(); check(false, name); } catch (const std::exception&) { check(true, name); }
}
int main() {
    const auto dir = fs::temp_directory_path() / ("strata-attention-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        Geometry g; g.layers = 40; g.hidden = 5120; g.compress_ratios = {0};
        const std::vector<fixture::Kv> meta{fixture::str("strata.deepseek41.attention_profile", "1017-layer0-swa128-fp8-block32")};
        std::vector<fixture::Tensor> tensors;
        const std::array<const char*, 5> names{"wq_a", "wq_b", "wkv", "wo_a", "wo_b"};
        const std::array<uint64_t, 5> rows{1280, 32768, 512, 8192, 5120}, cols{5120, 1280, 5120, 4096, 8192};
        for (size_t i = 0; i < 5; ++i) {
            tensors.push_back({std::string("blk.0.attn_") + names[i] + ".weight", {cols[i], rows[i]}, 24, uint8_t(i + 1)});
            tensors.push_back({std::string("blk.0.attn_") + names[i] + ".scale", {cols[i] / 32, rows[i] / 32}, 24, 6});
        }
        tensors.push_back({"blk.0.attn_q_norm.weight", {1280}, 30, 7});
        tensors.push_back({"blk.0.attn_kv_norm.weight", {512}, 30, 8});
        tensors.push_back({"blk.0.attn_sink", {64}, 0, 9});
        auto read = [&](const std::vector<fixture::Kv>& metadata, const std::vector<fixture::Tensor>& ts, uint64_t cut = 0) {
            const auto file = dir / (std::to_string(checks) + ".gguf");
            fixture::write(file, metadata, ts, cut); return GgufModel::open(file.string());
        };
        const auto model = read(meta, tensors);
        const auto w = read_attention_weights(model, g, 0);
        check(w.layer == 0 && w.projections[1].values.size() == 32768 * 1280 && w.projections[4].cols == 8192 &&
              w.q_norm.size() == 1280 && w.kv_norm.size() == 512 && w.sink.size() == 64,
              "exact fixed 1017 projection/norm/sink shapes");
        check(w.projections[3].values[4096] == fixture::pattern(4, 4096), "grouped wo_a bytes preserved");
        rejects("other layer refused", [&] { read_attention_weights(model, g, 1); });
        rejects("compressed geometry refused", [&] { auto bad = g; bad.compress_ratios = {2}; read_attention_weights(model, bad, 0); });
        rejects("wrong hidden size refused", [&] { auto bad = g; bad.hidden = 4096; read_attention_weights(model, bad, 0); });
        rejects("missing profile refused", [&] { const auto m = read({}, {}); read_attention_weights(m, g, 0); });
        rejects("wrong profile refused", [&] { const auto m = read({fixture::str("strata.deepseek41.attention_profile", "other")}, {}); read_attention_weights(m, g, 0); });
        rejects("wrong projection dtype refused", [&] { auto t = tensors[0]; t.type = 30; const auto m = read(meta, {t}); read_attention_weights(m, g, 0); });
        rejects("transposed projection refused", [&] { auto t = tensors[0]; t.shape = {1280, 5120}; const auto m = read(meta, {t}); read_attention_weights(m, g, 0); });
        rejects("missing scale refused", [&] { const auto m = read(meta, {tensors[0]}); read_attention_weights(m, g, 0); });
        rejects("truncated scale refused", [&] { const auto m = read(meta, {tensors[0], tensors[1]}, 1); read_attention_weights(m, g, 0); });
        auto g1 = g; g1.compress_ratios.assign(40, 0);
        const std::vector<fixture::Kv> meta1{fixture::str("strata.deepseek41.attention_profile", "1017-layer1-swa128-fp8-block32")};
        auto tensors1 = tensors;
        for (auto& t : tensors1) t.name.replace(0, 5, "blk.1");
        const auto model1 = read(meta1, tensors1);
        const auto w1 = read_attention_weights(model1, g1, 1);
        check(w1.layer == 1 && w1.projections[1].rows == 32768 && w1.q_norm.size() == 1280 &&
              w1.projections[3].values[4096] == fixture::pattern(4, 4096), "layer-1 pure-SWA profile preserves exact shapes and bytes");
        rejects("layer-0 profile cannot load layer 1", [&] { read_attention_weights(model, g1, 1); });
        rejects("layer-1 profile cannot load layer 0", [&] { read_attention_weights(model1, g, 0); });
        rejects("layer-1 wrong ratio", [&] { auto bad = g1; bad.compress_ratios[1] = 2; read_attention_weights(model1, bad, 1); });
        rejects("layer-1 missing ratio", [&] { auto bad = g1; bad.compress_ratios.resize(1); read_attention_weights(model1, bad, 1); });
        rejects("layer-1 wrong layer count", [&] { auto bad = g1; bad.layers = 39; read_attention_weights(model1, bad, 1); });
        rejects("layer-1 names cannot use layer 0", [&] { const auto m = read(meta1, {tensors[0]}); read_attention_weights(m, g1, 1); });
        auto g2 = g; g2.compress_ratios.assign(40, 0); g2.compress_ratios[2] = 2;
        const std::vector<fixture::Kv> meta2{fixture::str("strata.deepseek41.attention_profile", "1017-layer2-ratio2-fp8-block32")};
        auto tensors2 = tensors;
        for (auto& t : tensors2) t.name.replace(0, 5, "blk.2");
        const auto model2 = read(meta2, tensors2);
        const auto w2 = read_attention_weights(model2, g2, 2);
        check(w2.layer == 2 && w2.projections[1].rows == 32768 && w2.projections[4].cols == 8192 &&
              w2.q_norm.size() == 1280 && w2.kv_norm.size() == 512 && w2.sink.size() == 64 &&
              w2.projections[3].values[4096] == fixture::pattern(4, 4096), "layer-2 profile preserves fixed shapes and bytes");
        rejects("layer-0 profile cannot load layer 2", [&] { read_attention_weights(model, g2, 2); });
        rejects("layer-2 profile cannot load layer 0", [&] { read_attention_weights(model2, g, 0); });
        rejects("layer-2 wrong ratio", [&] { auto bad = g2; bad.compress_ratios[2] = 4; read_attention_weights(model2, bad, 2); });
        rejects("layer-2 missing ratio", [&] { auto bad = g2; bad.compress_ratios.resize(2); read_attention_weights(model2, bad, 2); });
        rejects("layer-2 wrong layer count", [&] { auto bad = g2; bad.layers = 39; read_attention_weights(model2, bad, 2); });
        rejects("layer-2 wrong hidden size", [&] { auto bad = g2; bad.hidden = 4096; read_attention_weights(model2, bad, 2); });
        rejects("layer-2 names cannot use layer 0", [&] { const auto m = read(meta2, {tensors[0]}); read_attention_weights(m, g2, 2); });
        rejects("layer-2 projection dtype", [&] { auto t = tensors2[0]; t.type = 30; const auto m = read(meta2, {t}); read_attention_weights(m, g2, 2); });
        rejects("layer-2 truncated scale", [&] { const auto m = read(meta2, {tensors2[0], tensors2[1]}, 1); read_attention_weights(m, g2, 2); });
        rejects("layer-2 invalid norm shape", [&] { auto ts = tensors2; ts[10].shape = {1279}; const auto m = read(meta2, ts); read_attention_weights(m, g2, 2); });
        auto g3 = g2; g3.compress_ratios[3] = 2; g3.sources.resize(40);
        g3.sources[2] = g3.sources[3] = {2, 2, -1};
        const std::vector<fixture::Kv> meta3{fixture::str("strata.deepseek41.attention_profile", "1017-layer3-shared2-fp8-block32")};
        auto tensors3 = tensors;
        for (auto& t : tensors3) t.name.replace(0, 5, "blk.3");
        const auto model3 = read(meta3, tensors3);
        const auto w3 = read_attention_weights(model3, g3, 3);
        check(w3.layer == 3 && w3.projections[1].rows == 32768 && w3.q_norm.size() == 1280 &&
              w3.projections[3].values[4096] == fixture::pattern(4, 4096), "layer-3 consumer profile preserves projection bytes");
        rejects("layer-2 profile cannot load layer 3", [&] { read_attention_weights(model2, g3, 3); });
        rejects("layer-3 profile cannot load layer 2", [&] { read_attention_weights(model3, g3, 2); });
        rejects("layer-3 missing sharing topology", [&] { auto bad = g3; bad.sources.clear(); read_attention_weights(model3, bad, 3); });
        rejects("layer-3 wrong KV source", [&] { auto bad = g3; bad.sources[3].kv = 3; read_attention_weights(model3, bad, 3); });
        rejects("layer-3 wrong index source", [&] { auto bad = g3; bad.sources[3].index = 3; read_attention_weights(model3, bad, 3); });
        rejects("layer-3 candidate source forbidden", [&] { auto bad = g3; bad.sources[3].candidates = 2; read_attention_weights(model3, bad, 3); });
        rejects("layer-2 must own shared source", [&] { auto bad = g3; bad.sources[2].kv = 1; read_attention_weights(model3, bad, 3); });
        rejects("layer-3 ratio must be two", [&] { auto bad = g3; bad.compress_ratios[3] = 1; read_attention_weights(model3, bad, 3); });
        rejects("layer-2 source ratio must be two", [&] { auto bad = g3; bad.compress_ratios[2] = 1; read_attention_weights(model3, bad, 3); });
        rejects("layer-3 names cannot use layer 2", [&] { const auto m = read(meta3, {tensors2[0]}); read_attention_weights(m, g3, 3); });
        for (int layer : {4,5,6,7,9,10,11,12,13,15,16,17,18,19,21,22,23}) {
            const int source=layer<8?2:layer<14?8:layer<20?14:20;
            auto geometry=g3; geometry.compress_ratios[size_t(layer)]=layer<20?2:1; geometry.sources[size_t(layer)]={source,source,-1};
            geometry.compress_ratios[size_t(source)]=layer<20?2:1;geometry.sources[size_t(source)]={source,source,-1};
            const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.attention_profile", "1017-layer"+std::to_string(layer)+"-shared"+std::to_string(source)+"-fp8-block32")};
            auto ts=tensors; for (auto& t:ts) t.name.replace(0,5,"blk."+std::to_string(layer));
            const auto m=read(metadata,ts); const auto w=read_attention_weights(m,geometry,layer);
            check(w.layer==layer && w.projections[3].values[4096]==fixture::pattern(4,4096),"explicit consumer preserves its original layer and bytes");
            rejects("consumer cannot borrow another profile",[&] { read_attention_weights(model3,geometry,layer); });
            rejects("consumer cannot use another source",[&] { auto bad=geometry;bad.sources[size_t(layer)].index=layer;read_attention_weights(m,bad,layer); });
            rejects("consumer cannot use candidates",[&] { auto bad=geometry;bad.sources[size_t(layer)].candidates=2;read_attention_weights(m,bad,layer); });
            rejects("consumer ratio must match original source",[&] { auto bad=geometry;bad.compress_ratios[size_t(layer)]=layer<20?1:2;read_attention_weights(m,bad,layer); });
            rejects("consumer layer-specific tensor names",[&] { const auto bad=read(metadata,{tensors3[0]});read_attention_weights(bad,geometry,layer); });
            if(layer>=9) {
                rejects("new consumer rejects source-2 KV",[&]{auto bad=geometry;bad.sources[size_t(layer)].kv=2;read_attention_weights(m,bad,layer);});
                rejects("consumer requires original source ownership",[&]{auto bad=geometry;bad.sources[size_t(source)].index=2;read_attention_weights(m,bad,layer);});
                rejects("consumer requires original source ratio",[&]{auto bad=geometry;bad.compress_ratios[size_t(source)]=layer<20?1:2;read_attention_weights(m,bad,layer);});
                rejects("new consumer requires source topology",[&]{auto bad=geometry;bad.sources.clear();read_attention_weights(m,bad,layer);});
            }
        }
        rejects("consumer profile cannot load owner 8",[&] { read_attention_weights(model3,g3,8); });
        rejects("layer20 rejects a consumer profile",[&]{read_attention_weights(model3,g3,20);});
        rejects("negative layer rejected",[&] { read_attention_weights(model3,g3,-1); });
        for (int layer : {8,14,20}) {
            const int ratio = layer == 20 ? 1 : 2;
            auto source_geometry=g3;source_geometry.compress_ratios[size_t(layer)]=ratio;source_geometry.sources[size_t(layer)]={layer,layer,-1};
            const std::vector<fixture::Kv> source_meta{fixture::str("strata.deepseek41.attention_profile","1017-layer"+std::to_string(layer)+"-ratio"+std::to_string(ratio)+"-fp8-block32")};
            auto source_tensors=tensors;for(auto& t:source_tensors)t.name.replace(0,5,"blk."+std::to_string(layer));
            const auto source_model=read(source_meta,source_tensors);const auto source_weights=read_attention_weights(source_model,source_geometry,layer);
            check(source_weights.layer==layer && source_weights.projections[3].values[4096]==fixture::pattern(4,4096),"original source profile and original bytes");
            rejects("original source must own KV",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].kv=2;read_attention_weights(source_model,bad,layer);});
            rejects("original source must own index",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].index=2;read_attention_weights(source_model,bad,layer);});
            rejects("original source cannot use candidates",[&]{auto bad=source_geometry;bad.sources[size_t(layer)].candidates=2;read_attention_weights(source_model,bad,layer);});
            rejects("original source ratio is fixed",[&]{auto bad=source_geometry;bad.compress_ratios[size_t(layer)]=3-ratio;read_attention_weights(source_model,bad,layer);});
            rejects("original source missing topology",[&]{auto bad=source_geometry;bad.sources.clear();read_attention_weights(source_model,bad,layer);});
            rejects("original source must use own names",[&]{const auto bad=read(source_meta,tensors2);read_attention_weights(bad,source_geometry,layer);});
            rejects("original source truncated scale",[&]{const auto bad=read(source_meta,{source_tensors[0],source_tensors[1]},1);read_attention_weights(bad,source_geometry,layer);});
            for (int other : {2,8,14,20}) if (other != layer)
                rejects("source profile cannot be borrowed by another owner",[&]{read_attention_weights(source_model,source_geometry,other);});
        }
        for(int layer:{24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39}) {
            const int owner=layer/4*4;auto source=g3;source.compress_ratios[20]=source.compress_ratios[size_t(owner)]=source.compress_ratios[size_t(layer)]=1;
            source.sources[20]={20,20,-1};source.sources[size_t(owner)]={20,owner,20};source.sources[size_t(layer)]={20,owner,layer==owner?20:-1};
            const std::string profile=layer==owner?"1017-layer"+std::to_string(layer)+"-candidates20-fp8-block32":"1017-layer"+std::to_string(layer)+"-shared"+std::to_string(owner)+"-fp8-block32";
            const std::vector<fixture::Kv> metadata{fixture::str("strata.deepseek41.attention_profile",profile)};
            auto ts=tensors;for(auto& t:ts)t.name.replace(0,5,"blk."+std::to_string(layer));
            const auto m=read(metadata,ts);const auto w=read_attention_weights(m,source,layer);
            check(w.layer==layer && w.projections[3].values[4096]==fixture::pattern(4,4096),"candidate source group keeps original projection bytes");
            rejects("candidate source group rejects MTP layer40",[&]{read_attention_weights(m,source,40);});
            rejects("candidate source group rejects wrong profile",[&]{read_attention_weights(model3,source,layer);});
            rejects("candidate source group requires source20 KV",[&]{auto bad=source;bad.sources[size_t(layer)].kv=24;read_attention_weights(m,bad,layer);});
            rejects("candidate source group requires own index owner",[&]{auto bad=source;bad.sources[size_t(layer)].index=20;read_attention_weights(m,bad,layer);});
            rejects("candidate source group candidate topology",[&]{auto bad=source;bad.sources[size_t(layer)].candidates=layer==owner?-1:20;read_attention_weights(m,bad,layer);});
            for(int origin:{20,owner})rejects("candidate source group requires ratio1 owners",[&]{auto bad=source;bad.compress_ratios[size_t(origin)]=2;read_attention_weights(m,bad,layer);});
            rejects("consumer candidate source must use source20 candidates",[&]{auto bad=source;bad.sources[size_t(owner)].candidates=-1;read_attention_weights(m,bad,layer);});
            rejects("candidate source group missing source map",[&]{auto bad=source;bad.sources.clear();read_attention_weights(m,bad,layer);});
            rejects("candidate source group cannot use another layer tensor",[&]{const auto bad=read(metadata,{tensors3[0]});read_attention_weights(bad,source,layer);});
            rejects("candidate source group truncated payload",[&]{const auto bad=read(metadata,{ts[0],ts[1]},1);read_attention_weights(bad,source,layer);});
        }
        for (const auto& shape : {std::pair{32768, 32}, std::pair{32, 8192}}) {
            std::vector<uint8_t> values(size_t(shape.first) * shape.second, 0x38), scales(values.size() / 1024, 127);
            const auto expanded = fp8_block32_to_bf16({values, scales, shape.first, shape.second});
            check(expanded.size() == values.size() && expanded.front() == kernels::bf16_from_f32(1.f) && expanded.back() == expanded.front(),
                  "attention FP8 dimension boundary expands exactly");
            rejects("oversized FP8 matrix refused", [&] { fp8_block32_to_bf16({values, scales, 32768, 8192}); });
        }
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
