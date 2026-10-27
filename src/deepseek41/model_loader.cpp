#include "strata/deepseek41/model_loader.hpp"
#include <filesystem>

namespace strata::deepseek41 {
namespace {
void revision(const GgufModel& model) {
    bool found=false;
    for(size_t i=0;i<model.size();++i)for(const char* key:{"strata.deepseek41.revision","strata.deepseek41.model_revision"})
        if(const auto* v=model.shard(i).get(key)){
            if(v->type!=MetaType::STRING || v->s!=checkpoint_1017_revision)throw std::invalid_argument("Model component revision differs from pinned 1017");
            found=true;
        }
    if(!found)throw std::invalid_argument("Model component lacks pinned 1017 revision");
}
template<class T> std::span<const T> router_tensor(const GgufModel& model,const std::string& name,uint32_t type,const std::vector<uint64_t>& shape,size_t n){
    size_t shard=0;const auto* t=model.find(name,&shard);
    if(!t || t->type!=type || t->shape!=shape || !model.in_bounds(*t,shard))throw std::invalid_argument("Malformed model router tensor: "+name);
    const auto* data=model.shard(shard).tensor_data(*t);
    if(uintptr_t(data)%alignof(T))throw std::invalid_argument("Misaligned model router tensor: "+name);
    return {reinterpret_cast<const T*>(data),n};
}
}
const GgufModel& LoadedModel::open(const std::string& path) {
    const auto canonical=std::filesystem::canonical(path).string();
    if(const auto it=models_.find(canonical);it!=models_.end())return *it->second;
    auto model=std::make_unique<GgufModel>(GgufModel::open(canonical));revision(*model);
    const auto* result=model.get();models_.emplace(canonical,std::move(model));return *result;
}
LoadedModel::LoadedModel(const ModelManifest& m) {
    // Source coverage and identity fail before dense GPU allocation/conversion.
    {std::map<std::string,std::unique_ptr<GgufModel>> files;
        for(const auto& f:m.expert_gguf){const auto path=std::filesystem::canonical(f.path).string();
            auto& model=files[path];if(!model)model=std::make_unique<GgufModel>(GgufModel::open(path));
            experts.add_gguf_layer(*model,f.layer);
        }
    } // Expert payload mappings are not retained by the file index.
    for(const auto& f:m.expert_raw)experts.add_raw(f.layer,f.expert,f.path,checkpoint_1017_revision);
    if(m.experts_complete)experts.require_complete();
    experts_complete=m.experts_complete;
    for(size_t i=0;i<2;++i){const int l=i?14:1;const auto& f=m.engram[i];
        {const auto table=GgufModel::open(f.table);revision(table);
            const auto* c=table.meta().get("strata.deepseek41.engram_complete");
            if(!c || c->type!=MetaType::U32 || c->u>1 || (c->u==1)!=f.complete)throw std::invalid_argument("Manifest and Engram table coverage disagree");
            for(size_t shard=1;shard<table.size();++shard)if(const auto* flag=table.shard(shard).get("strata.deepseek41.engram_complete"))
                if(flag->type!=MetaType::U32 || flag->u!=c->u)throw std::invalid_argument("Conflicting Engram shard coverage");
        }
        EngramIoOptions validation;validation.max_inflight=1;validation.cache_rows=0;validation.allow_partial=!f.complete;
        {EngramTable checked(l,f.table,validation);} // reuse full profile/range/guard checks without row payload reads
        const auto& map=open(f.token_map);token_maps[i]=read_engram_token_map(map.meta(),l);
        tables[i]=std::filesystem::canonical(f.table).string();engram_complete[i]=f.complete;
    }
    weights.layers.resize(40);
    for(int l=0;l<40;++l){const auto& f=m.layers[size_t(l)];const auto& router=open(f.router);
        weights.layers[size_t(l)]={read_mhc_weights(open(f.mhc),geometry,l,MhcBranch::attention),
            read_mhc_weights(open(f.mhc),geometry,l,MhcBranch::ffn),read_attention_weights(open(f.attention),geometry,l),
            read_shared_expert(open(f.shared),geometry,l),router_tensor<uint16_t>(router,f.router_weight,30,{5120,384},5120*384),
            router_tensor<float>(router,f.router_bias,0,{384},384)};
    }
    weights.engram=read_engram_weights(open(m.engram[0].projection),1);
    weights.engram14=read_engram_weights(open(m.engram[1].projection),14);
    const auto& s2=open(m.layers[2].attention);weights.compressor=read_compressor_weights(s2,geometry,2);weights.key=read_index_key_weights(s2,geometry,2);weights.query=read_index_query_weights(s2,geometry,2);
    const auto& s8=open(m.layers[8].attention);weights.compressor8=read_compressor_weights(s8,geometry,8);weights.key8=read_index_key_weights(s8,geometry,8);weights.query8=read_index_query_weights(s8,geometry,8);
    const auto& s14=open(m.layers[14].attention);weights.compressor14=read_compressor_weights(s14,geometry,14);weights.key14=read_index_key_weights(s14,geometry,14);weights.query14=read_index_query_weights(s14,geometry,14);
    const auto& s20=open(m.layers[20].attention);weights.compressor20=read_source20_compressor_weights(s20,geometry);weights.key20=read_source20_index_key_weights(s20,geometry);weights.query20=read_source20_index_query_weights(s20,geometry);
    weights.query24=read_candidate_index_query_weights(open(m.layers[24].attention),geometry,24);
    weights.query28=read_candidate_index_query_weights(open(m.layers[28].attention),geometry,28);
    weights.query32=read_candidate_index_query_weights(open(m.layers[32].attention),geometry,32);
    weights.query36=read_candidate_index_query_weights(open(m.layers[36].attention),geometry,36);
    text_io=read_text_io_weights(open(m.text_io));
    std::set<std::string> mapped;
    for(const auto& [path,model]:models_){(void)path;for(size_t s=0;s<model->size();++s)mapped.insert(std::filesystem::canonical(model->shard(s).path()).string());}
    for(int l=0;l<40;++l)for(int e=0;e<384;++e)if(experts.contains(l,e))
        for(const auto& part:experts.at(l,e).parts)if(mapped.contains(experts.file(part.file).path))retained_expert_mapping_upper_bytes+=experts.role_bytes();
}
} // namespace strata::deepseek41
