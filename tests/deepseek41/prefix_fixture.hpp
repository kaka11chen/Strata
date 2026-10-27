#pragma once

#include "decoder_test_utils.hpp"
#include "strata/deepseek41/text_prefix_cuda.hpp"
#include "strata/deepseek41/expert_file.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

namespace prefix_fixture {
namespace fs = std::filesystem;
struct MissingExperts : std::runtime_error {
    int layer; std::vector<int> ids;
    MissingExperts(int l, std::vector<int> i) : std::runtime_error("Original fixture experts missing"), layer(l), ids(std::move(i)) {}
};
struct RealWeights {
    Geometry g = geometry();
    GgufModel input, mhc0, attention0, shared0, projection, lookup;
    std::vector<GgufModel> decoder;
    std::array<GgufModel, 4> routed;
    std::unique_ptr<GgufModel> routed8, routed14, routed20, projection14, lookup14;
    std::array<std::vector<uint16_t>, 40> router;
    std::array<std::vector<float>, 40> bias;
    std::array<std::unique_ptr<ExpertIndex>, 40> index;
    std::array<std::array<bool, 384>, 40> populated{};
    std::vector<uint32_t> map, map14;
    TextPrefixWeights weights;
    std::string table, table14;
    explicit RealWeights(const fs::path& dir, int count = 4)
        : input(GgufModel::open((dir/"real-decoder0.gguf").string())),
          mhc0(GgufModel::open((dir/"real-mhc0.gguf").string())),
          attention0(GgufModel::open((dir/"real-attention0.gguf").string())),
          shared0(GgufModel::open((dir/"real-shared0.gguf").string())),
          projection(GgufModel::open((dir/"real-engram1-projection.gguf").string())),
          lookup(GgufModel::open((dir/"real-engram1-reference.gguf").string())),
          routed{GgufModel::open((dir/"real-decoder-routed0.gguf").string()),GgufModel::open((dir/"real-layer1-decoder-routed.gguf").string()),GgufModel::open((dir/"real-layer2-decoder-routed.gguf").string()),GgufModel::open((dir/"real-layer3-decoder-routed.gguf").string())},
          map(read_engram_token_map(lookup.shard(0))), table((dir/"real-engram1-table.gguf").string()) {
        if (count != 4 && count != 8 && count != 14 && count != 15 && count != 20 && count != 24 && count != 28 && count != 32 && count != 36 && count != 40) throw std::invalid_argument("Fixture requires 4, 8, 14, 15, 20, 24, 28, 32, 36 or 40 layers");
        for (int l=1;l<count;++l) decoder.push_back(GgufModel::open((dir/("real-layer"+std::to_string(l)+"-decoder.gguf")).string()));
        if (count >= 14) routed8=std::make_unique<GgufModel>(GgufModel::open((dir/"real-layer8-decoder-routed.gguf").string()));
        if (count >= 15) {
            for (int l=14;l<std::min(count,20);++l) { g.compress_ratios[size_t(l)]=2; g.sources[size_t(l)]={14,14,-1}; }
            routed14=std::make_unique<GgufModel>(GgufModel::open((dir/"real-layer14-decoder-routed.gguf").string()));
            projection14=std::make_unique<GgufModel>(GgufModel::open((dir/"real-engram14-projection.gguf").string()));
            lookup14=std::make_unique<GgufModel>(GgufModel::open((dir/"real-engram14-reference.gguf").string()));
            map14=read_engram_token_map(lookup14->shard(0),14);table14=(dir/"real-engram14-table.gguf").string();
        }
        if (count >= 24) {
            for (int l=20;l<24;++l) {g.compress_ratios[size_t(l)]=1;g.sources[size_t(l)]={20,20,-1};}
            routed20=std::make_unique<GgufModel>(GgufModel::open((dir/"real-layer20-decoder-routed.gguf").string()));
        }
        if (count >= 28) for (int l=24;l<count;++l) {
            g.compress_ratios[size_t(l)]=1;g.sources[size_t(l)]={20,l/4*4,l%4==0?20:-1};
        }
        weights.layers.resize(size_t(count));
        for (int l=0;l<count;++l) {
            const size_t i=size_t(l); const auto& r=l<4?routed[i]:l==8?*routed8:l==14?*routed14:l==20?*routed20:decoder[i-1];
            router[i]=tensor<uint16_t>(r,"router.weight",30,384*hidden); bias[i]=tensor<float>(r,"router.bias",0,384);
            if (l<4 || l==8 || l==14 || l==20) {
                const auto* ids=r.meta().get("strata.deepseek41.populated_ids");
                if (!ids) ids=r.meta().get("strata.deepseek41.selected_ids");
                if (!ids || ids->type!=MetaType::ARRAY || ids->items.size()!=ids->count || ids->count<6 || ids->count>64)
                    throw std::runtime_error("Missing complete original expert coverage");
                for (const auto& id:ids->items) {
                    if (id.u>=384) throw std::runtime_error("Bad fixture expert ID");
                    populated[i][size_t(id.u)]=true;
                }
                index[i]=std::make_unique<ExpertIndex>(r,g,std::vector<int>{l});
            }
            const auto& m=l?decoder[i-1]:mhc0;
            weights.layers[i]={read_mhc_weights(m,g,l,MhcBranch::attention),read_mhc_weights(m,g,l,MhcBranch::ffn),
                read_attention_weights(l?m:attention0,g,l),read_shared_expert(l?m:shared0,g,l),router[i],bias[i]};
        }
        weights.engram=read_layer1_engram_weights(projection);
        weights.compressor=read_compressor_weights(decoder[1],g,2);
        weights.key=read_index_key_weights(decoder[1],g,2);
        weights.query=read_index_query_weights(decoder[1],g,2);
        if (count >= 14) {
            weights.compressor8=read_compressor_weights(decoder[7],g,8);
            weights.key8=read_index_key_weights(decoder[7],g,8);
            weights.query8=read_index_query_weights(decoder[7],g,8);
        }
        if (count >= 15) {
            weights.engram14=read_engram_weights(*projection14,14);
            weights.compressor14=read_compressor_weights(decoder[13],g,14);
            weights.key14=read_index_key_weights(decoder[13],g,14);
            weights.query14=read_index_query_weights(decoder[13],g,14);
        }
        if (count >= 24) {
            weights.compressor20=read_source20_compressor_weights(decoder[19],g);
            weights.key20=read_source20_index_key_weights(decoder[19],g);
            weights.query20=read_source20_index_query_weights(decoder[19],g);
        }
        if (count >= 28) weights.query24=read_candidate_index_query_weights(decoder[23],g);
        if (count >= 32) weights.query28=read_candidate_index_query_weights(decoder[27],g,28);
        if (count >= 36) weights.query32=read_candidate_index_query_weights(decoder[31],g,32);
        if (count == 40) weights.query36=read_candidate_index_query_weights(decoder[35],g,36);
    }
};
// Fixture preparation only: turn known guarded GGUFs and local supplemental
// files into the same generic original-ID index used for a complete checkpoint.
// Runtime reads no longer branch on original layer numbers or fixture file names.
inline ExpertFileIndex expert_file_index(const RealWeights& f, const fs::path& extra) {
    ExpertFileIndex result(f.g);
    for (size_t l = 0; l < f.weights.layers.size(); ++l) if (f.index[l]) {
        const auto& model = l < 4 ? f.routed[l] : l == 8 ? *f.routed8 : l == 14 ? *f.routed14 : *f.routed20;
        result.add_gguf_layer(model, int(l));
    }
    for (int l = 0; l < int(f.weights.layers.size()); ++l) for (int e = 0; e < f.g.experts; ++e) {
        if (result.contains(l, e)) continue; // retain the existing GGUF precedence
        const auto path = extra / ("layer-" + std::to_string(l) + "-expert-" + std::to_string(e) + ".mxfp4");
        if (fs::exists(path)) result.add_raw(l, e, path.string(), checkpoint_1017_revision);
    }
    return result;
}
// Test-only dispatch across existing sparse fixtures and separately hash-verified
// supplemental MXFP4 blobs. Uses the production lossless storage adapter; never
// fabricates expert weights. Normal verification rejects missing coverage before
// ANY expert read; explicit preparation may suspend that same route for verified files.
struct Source : core::ExpertSource {
    const RealWeights& f; fs::path extra;
    bool prepare_missing; // Explicit test-only preparation protocol; normal probes remain read-only.
    std::array<std::unique_ptr<Mxfp4ExpertSource>,40> sources;
    std::array<uint64_t,40> loads{};
    std::array<std::set<int>,40> selected;
    const size_t raw_bytes = size_t(hidden)*2304/32*17*3;
    std::vector<uint8_t> raw, staging;
    std::unique_ptr<Mxfp4ExpertFileReader> file_reader;
    Source(const RealWeights& fixture,const fs::path& directory,bool prepare=false,bool direct=false)
        : f(fixture),extra(directory),prepare_missing(prepare),raw(direct?0:raw_bytes),staging(nvfp4_expert_bytes(f.g)),
          file_reader(direct?std::make_unique<Mxfp4ExpertFileReader>(f.g):nullptr) {
        for (size_t l=0;l<f.weights.layers.size();++l) if (f.index[l]) sources[l]=std::make_unique<Mxfp4ExpertSource>(f.g,*f.index[l]);
    }
    fs::path file(int layer,int expert) const { return extra/("layer-"+std::to_string(layer)+"-expert-"+std::to_string(expert)+".mxfp4"); }
    void begin_layer(int64_t layer,const int32_t* ids,int64_t count) override {
        if (layer<0 || size_t(layer)>=f.weights.layers.size() || count!=6 || !ids) throw std::runtime_error("Bad original layer/route");
        std::vector<int> missing;
        for (int k=0;k<6;++k) {
            const int id=ids[k]; if (id<0 || id>=384) throw std::runtime_error("Bad original expert ID");
            selected[size_t(layer)].insert(id);
            if (!f.populated[size_t(layer)][size_t(id)] && !fs::exists(file(int(layer),id))) missing.push_back(id);
        }
        if (!missing.empty() && prepare_missing) {
            // Stop before reading any expert for this original route. The external
            // preparer supplies bounded, hash-checked files and acknowledges READY.
            std::printf("MISSING_EXPERTS {\"layer\":%lld,\"experts\":[",static_cast<long long>(layer));
            for (size_t k=0;k<missing.size();++k) std::printf("%s%d",k?",":"",missing[k]);
            std::printf("]}\n");std::fflush(stdout);
            std::string reply;
            if (!std::getline(std::cin,reply) || reply!="READY") throw std::runtime_error("Expert preparation was not acknowledged");
            for (int id:missing) {
                const auto path=file(int(layer),id);
                if (!fs::exists(path) || fs::file_size(path)!=raw_bytes) throw std::runtime_error("Acknowledged original expert is missing or malformed");
            }
        } else if (!missing.empty()) throw MissingExperts(int(layer),std::move(missing));
    }
    const uint8_t* blob(int64_t layer,int64_t expert) override {
        if (layer<0 || size_t(layer)>=f.weights.layers.size() || expert<0 || expert>=384) throw std::runtime_error("Expert outside prefix geometry");
        const size_t l=size_t(layer);
        if (f.populated[l][size_t(expert)]) { ++loads[l]; return sources[l]->blob(layer,expert); }
        const auto path=file(int(layer),int(expert));
        if (!fs::exists(path) || fs::file_size(path)!=raw_bytes) throw std::runtime_error("Missing/malformed supplemental original expert");
        if (file_reader) { file_reader->read(path.string(),staging); ++loads[l]; return staging.data(); }
        std::ifstream in(path,std::ios::binary); in.exceptions(std::ios::badbit|std::ios::failbit);
        in.read(reinterpret_cast<char*>(raw.data()),std::streamsize(raw.size()));
        ExpertView view; const size_t part=raw.size()/3;
        for (size_t r=0;r<3;++r) { view.parts[r].data={raw.data()+r*part,part}; view.parts[r].type=39; }
        pack_nvfp4_expert(f.g,view,staging); ++loads[l]; return staging.data();
    }
    bool transient(int64_t,int64_t) const override { return true; }
    int64_t reads() const override { return int64_t(std::accumulate(loads.begin(),loads.end(),uint64_t(0))); }
};
struct Case {
    std::vector<int32_t> tokens;
    std::vector<float> residual,incoming;
};
struct Result {
    std::vector<std::vector<float>> residual,coefficients,kv;
    std::vector<float> compressed,keys,compressed8,keys8,compressed14,keys14,compressed20,keys20;
    std::vector<Route> routes; // layer-major
};
// Independent explicit calls with a separate destination for EVERY layer. The
// production prefix instead uses TWO buffers and owns whole-chain validation.
inline std::vector<Result> direct(const Geometry& g,const TextPrefixWeights& w,const std::vector<uint32_t>& map,
    const std::string& table,core::ExpertSource& source,const std::vector<Case>& cases,int capacity,cudaStream_t stream,
    const std::vector<uint32_t>& map14={},const std::string& table14={}) {
    int max_tokens=0; for (const auto& c:cases) max_tokens=std::max(max_tokens,int(c.tokens.size()));
    EngramIoOptions options; options.allow_partial=true;
    CudaDecoder l0(g,w.layers[0],source,6);
    CudaLayer1Decoder l1(g,w.layers[1],w.engram,map,table,source,6,options);
    CudaLayer2Decoder l2(g,w.layers[2],w.compressor,w.key,w.query,source,6,capacity,0,max_tokens);
    const size_t count=w.layers.size();
    std::unique_ptr<CudaSourceDecoder> l8;
    if (count >= 14) l8=std::make_unique<CudaSourceDecoder>(g,w.layers[8],w.compressor8,w.key8,w.query8,source,6,capacity,0,max_tokens,8);
    std::unique_ptr<CudaLayer14Decoder> l14;
    if (count >= 15) l14=std::make_unique<CudaLayer14Decoder>(g,w.layers[14],w.engram14,w.compressor14,w.key14,w.query14,map14,table14,source,6,capacity,options,0,max_tokens);
    std::unique_ptr<CudaSource20Decoder> l20;
    const int context20=std::min(1048576,2*capacity+1);
    if (count>=24) l20=std::make_unique<CudaSource20Decoder>(g,w.layers[20],w.compressor20,w.key20,w.query20,source,6,context20,std::min(context20,max_tokens));
    std::vector<std::unique_ptr<CudaSharedDecoder>> consumers(count);
    for (size_t l=3;l<count;++l) {
        if (l==24 || l==28 || l==32 || l==36) consumers[l]=std::make_unique<CudaSharedDecoder>(g,w.layers[l],l==24?w.query24:l==28?w.query28:l==32?w.query32:w.query36,source,6,context20,std::min(context20,max_tokens));
        else if (l!=8 && l!=14 && l!=20) consumers[l]=std::make_unique<CudaSharedDecoder>(g,w.layers[l],source,6,0,int(l));
    }
    const size_t residual_bytes=size_t(max_tokens)*residual_size*4, coefficient_bytes=size_t(max_tokens)*16;
    core::DeviceArena io((count+1)*(residual_bytes+(coefficient_bytes+255)/256*256));
    auto alloc=[&](size_t n) { return static_cast<float*>(io.alloc(n*4)); };
    auto *x=alloc(size_t(max_tokens)*residual_size),*p=alloc(size_t(max_tokens)*4);
    std::vector<float*> y(count),pre(count);
    for (size_t l=0;l<count;++l) { y[l]=alloc(size_t(max_tokens)*residual_size); pre[l]=alloc(size_t(max_tokens)*4); }
    int pos=0; std::vector<Result> result;
    for (const auto& c:cases) {
        const int n=int(c.tokens.size()); Result r; r.routes.resize(size_t(n)*count);r.residual.resize(count);r.coefficients.resize(count);r.kv.resize(count);
        upload(x,c.residual,stream);upload(p,c.incoming,stream);
        l0.forward(x,p,y[0],pre[0],n,pos,stream,std::span(r.routes).subspan(0,size_t(n)));
        l1.forward(c.tokens,y[0],pre[0],y[1],pre[1],pos,stream,std::span(r.routes).subspan(size_t(n),size_t(n)));
        l2.forward(y[1],pre[1],y[2],pre[2],n,pos,stream,std::span(r.routes).subspan(2*size_t(n),size_t(n)));
        const auto shared=l2.shared_batch();
        SharedAttentionBatch shared8,shared14,shared24,shared28,shared32,shared36;Source20AttentionBatch shared20;std::vector<float> saved_kv8,saved_kv14,saved_kv20,saved_kv24,saved_kv28,saved_kv32,saved_kv36;
        const auto saved_kv=cache(l2.kv_cache(),stream),saved_compressed=cache(l2.compressed_kv_cache(),stream,size_t(capacity)*512),
                   saved_keys=cache(l2.index_key_cache(),stream,size_t(capacity)*128);
        std::vector<const uint16_t*> rings{l0.kv_cache(),l1.kv_cache(),l2.kv_cache()};
        for (size_t l=3;l<count;++l) {
            if (l==8) {
                l8->forward(y[7],pre[7],y[8],pre[8],n,pos,stream,std::span(r.routes).subspan(8*size_t(n),size_t(n)));
                shared8=l8->shared_batch();saved_kv8=cache(l8->kv_cache(),stream);
                r.compressed8=cache(l8->compressed_kv_cache(),stream,size_t(capacity)*512);
                r.keys8=cache(l8->index_key_cache(),stream,size_t(capacity)*128);
            } else if (l==14) {
                l14->forward(c.tokens,y[13],pre[13],y[14],pre[14],pos,stream,std::span(r.routes).subspan(14*size_t(n),size_t(n)));
                shared14=l14->shared_batch();saved_kv14=cache(l14->kv_cache(),stream);
                r.compressed14=cache(l14->compressed_kv_cache(),stream,size_t(capacity)*512);
                r.keys14=cache(l14->index_key_cache(),stream,size_t(capacity)*128);
                check(l14->shared_batch().source_layer()==14 && l14->next_position()==pos+n,"direct complete layer14 owns its source publication");
            } else if (l==20) {
                l20->forward(y[19],pre[19],y[20],pre[20],n,pos,stream,std::span(r.routes).subspan(20*size_t(n),size_t(n)));
                shared20=l20->shared_batch();saved_kv20=cache(l20->kv_cache(),stream);
                r.compressed20=cache(l20->compressed_kv_cache(),stream,size_t(context20)*512);
                r.keys20=cache(l20->index_key_cache(),stream,size_t(context20)*128);
                check(shared20.view(pos,n,stream).key_count==pos+n && l20->next_position()==pos+n,"direct source20 commits complete decoder state");
            } else if (l>20) {
                if (l<=24 || l==28 || l==32 || l==36) consumers[l]->forward_source20(y[l-1],pre[l-1],y[l],pre[l],n,pos,shared20,stream,std::span(r.routes).subspan(l*size_t(n),size_t(n)));
                else consumers[l]->forward(y[l-1],pre[l-1],y[l],pre[l],n,pos,l<28?shared24:l<32?shared28:l<36?shared32:shared36,stream,std::span(r.routes).subspan(l*size_t(n),size_t(n)));
                if (l==24) {
                    shared24=consumers[l]->shared_batch();saved_kv24=cache(consumers[l]->kv_cache(),stream);
                    check(shared24.source_layer()==24 && consumers[l]->next_position()==pos+n,"direct source24 publishes only after complete decoder");
                }
                if (l==28) {
                    shared28=consumers[l]->shared_batch();saved_kv28=cache(consumers[l]->kv_cache(),stream);
                    check(shared28.source_layer()==28 && consumers[l]->next_position()==pos+n,"direct source28 publishes only after complete decoder");
                }
                if (l==32) {
                    shared32=consumers[l]->shared_batch();saved_kv32=cache(consumers[l]->kv_cache(),stream);
                    check(shared32.source_layer()==32 && consumers[l]->next_position()==pos+n,"direct source32 publishes only after complete decoder");
                }
                if (l==36) {
                    shared36=consumers[l]->shared_batch();saved_kv36=cache(consumers[l]->kv_cache(),stream);
                    check(shared36.source_layer()==36 && consumers[l]->next_position()==pos+n,"direct source36 publishes only after complete decoder");
                }
                if (l>36) check(same(saved_kv36,cache(consumers[36]->kv_cache(),stream)) && shared36.source_layer()==36 && consumers[36]->next_position()==pos+n,
                                "shared consumer preserves source36 state layer="+std::to_string(l));
                if (l>32) check(same(saved_kv32,cache(consumers[32]->kv_cache(),stream)) && shared32.source_layer()==32 && consumers[32]->next_position()==pos+n,
                                "shared consumer preserves source32 state layer="+std::to_string(l));
                if (l>28) check(same(saved_kv28,cache(consumers[28]->kv_cache(),stream)) && shared28.source_layer()==28 && consumers[28]->next_position()==pos+n,
                                "shared consumer preserves source28 state layer="+std::to_string(l));
                if (l>24) check(same(saved_kv24,cache(consumers[24]->kv_cache(),stream)) && shared24.source_layer()==24 && consumers[24]->next_position()==pos+n,
                                "shared consumer preserves source24 state layer="+std::to_string(l));
                check(same(saved_kv20,cache(l20->kv_cache(),stream)) && same(r.compressed20,cache(l20->compressed_kv_cache(),stream,size_t(context20)*512)) &&
                    same(r.keys20,cache(l20->index_key_cache(),stream,size_t(context20)*128)) && shared20.view(pos,n,stream).key_count==pos+n && l20->next_position()==pos+n,
                    "shared consumer preserves source-20 state layer="+std::to_string(l));
            } else {
                consumers[l]->forward(y[l-1],pre[l-1],y[l],pre[l],n,pos,l<8?shared:l<14?shared8:shared14,stream,std::span(r.routes).subspan(l*size_t(n),size_t(n)));
            }
            if (l>14) check(same(saved_kv14,cache(l14->kv_cache(),stream)) && same(r.compressed14,cache(l14->compressed_kv_cache(),stream,size_t(capacity)*512)) &&
                same(r.keys14,cache(l14->index_key_cache(),stream,size_t(capacity)*128)) && l14->next_position()==pos+n && shared14.source_layer()==14,
                "shared consumer preserves source-14 state layer="+std::to_string(l));
            if (l>8) check(same(saved_kv8,cache(l8->kv_cache(),stream)) && same(r.compressed8,cache(l8->compressed_kv_cache(),stream,size_t(capacity)*512)) &&
                same(r.keys8,cache(l8->index_key_cache(),stream,size_t(capacity)*128)) && l8->next_position()==pos+n,
                "shared consumer preserves source-8 state layer="+std::to_string(l));
            check(same(saved_kv,cache(l2.kv_cache(),stream)) && same(saved_compressed,cache(l2.compressed_kv_cache(),stream,size_t(capacity)*512)) &&
                  same(saved_keys,cache(l2.index_key_cache(),stream,size_t(capacity)*128)) && l2.next_position()==pos+n,
                  "shared consumer preserves source state layer="+std::to_string(l));
            rings.push_back(l==20?l20->kv_cache():l==14?l14->kv_cache():l==8?l8->kv_cache():consumers[l]->kv_cache());
        }
        for (size_t l=0;l<count;++l) { r.residual[l]=download(y[l],size_t(n)*residual_size,stream);r.coefficients[l]=download(pre[l],size_t(n)*4,stream);r.kv[l]=cache(rings[l],stream); }
        r.compressed=cache(l2.compressed_kv_cache(),stream,size_t(capacity)*512);
        r.keys=cache(l2.index_key_cache(),stream,size_t(capacity)*128);
        result.push_back(std::move(r));pos+=n;
    }
    return result;
}
} // namespace prefix_fixture
