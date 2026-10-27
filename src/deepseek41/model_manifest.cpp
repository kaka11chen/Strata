#include "strata/deepseek41/model_manifest.hpp"
#include <filesystem>
#include <set>

namespace strata::deepseek41 {
Geometry model_1017_geometry() {
    Geometry g; g.layers=40; g.hidden=5120; g.expert_width=2304; g.experts=384; g.top_k=6;
    g.compress_ratios.resize(40); g.sources.resize(40);
    for(int l=2;l<40;++l) {
        const int kv=l<8?2:l<14?8:l<20?14:20, index=l<24?kv:l/4*4;
        g.compress_ratios[size_t(l)]=l<20?2:1;
        g.sources[size_t(l)]={kv,index,l>=24 && l==index?20:-1};
    }
    return g;
}
size_t ModelManifest::metadata_upper_bytes() const {
    size_t bytes=sizeof(*this)+expert_gguf.capacity()*sizeof(ExpertGgufFile)+expert_raw.capacity()*sizeof(ExpertRawFile);
    auto add=[&](const std::string& path){bytes+=path.capacity()+1;};
    add(reference_root);add(text_io);
    for(const auto& f:layers){add(f.mhc);add(f.attention);add(f.shared);add(f.router);add(f.router_weight);add(f.router_bias);}
    for(const auto& f:engram){add(f.projection);add(f.token_map);add(f.table);}
    for(const auto& f:expert_gguf)add(f.path);
    for(const auto& f:expert_raw)add(f.path);
    return bytes;
}
ModelManifest ModelManifest::read(const std::string& path) {
    namespace fs=std::filesystem;
    if(fs::file_size(path)>8u*1024*1024)throw std::invalid_argument("Model manifest exceeds 8 MiB metadata limit");
    const GgufFile file(path);
    if(!file.tensors().empty())throw std::invalid_argument("Model manifest must contain metadata only");
    std::set<std::string> used;
    auto value=[&](const std::string& key,MetaType type)->const MetaValue& {
        used.insert(key);const auto* v=file.get(key);
        if(!v || v->type!=type)throw std::invalid_argument("Missing/wrong model manifest field: "+key);
        return *v;
    };
    auto text=[&](const std::string& key){
        const auto s=value(key,MetaType::STRING).s;
        if(s.empty() || s.size()>4096 || s.find('\0')!=std::string::npos)throw std::invalid_argument("Invalid model manifest string: "+key);
        return s;
    };
    const std::string p=model_manifest_prefix;
    auto number=[&](const std::string& key,uint64_t max){const auto n=value(p+key,MetaType::U32).u;if(n>max)throw std::invalid_argument("Model manifest integer out of range: "+key);return int(n);};
    if(text("general.architecture")!="deepseek41-manifest" || number("version",1)!=1 ||
       text(p+"revision")!=checkpoint_1017_revision || text(p+"profile")!="1017-full40-context5")
        throw std::invalid_argument("Model manifest must use pinned 1017 schema1/context5");
    const auto root=fs::absolute(path).parent_path();
    auto location=[&](const std::string& key){return (root/fs::path(text(p+key))).lexically_normal().string();};
    ModelManifest result;result.reference_root=location("reference_root");result.text_io=location("text_io");
    result.experts_complete=number("experts_complete",1)!=0;
    for(int l=0;l<40;++l){auto& f=result.layers[size_t(l)];const auto key="layer."+std::to_string(l)+".";
        f.mhc=location(key+"mhc");f.attention=location(key+"attention");f.shared=location(key+"shared");f.router=location(key+"router");
        f.router_weight=text(p+key+"router_weight");f.router_bias=text(p+key+"router_bias");
    }
    for(size_t i=0;i<2;++i){auto& f=result.engram[i];const auto key="engram."+std::to_string(i?14:1)+".";
        f.projection=location(key+"projection");f.token_map=location(key+"token_map");f.table=location(key+"table");f.complete=number(key+"complete",1)!=0;
    }
    const int ggufs=number("expert_gguf_count",40),raws=number("expert_raw_count",40*384);
    std::set<int> layers;std::set<std::pair<int,int>> ids;
    for(int i=0;i<ggufs;++i){const auto key="expert_gguf."+std::to_string(i)+".";const int l=number(key+"layer",39);
        if(!layers.insert(l).second)throw std::invalid_argument("Repeated GGUF expert layer in model manifest");
        result.expert_gguf.push_back({l,location(key+"path")});
    }
    for(int i=0;i<raws;++i){const auto key="expert_raw."+std::to_string(i)+".";const int l=number(key+"layer",39),e=number(key+"expert",383);
        if(!ids.emplace(l,e).second)throw std::invalid_argument("Repeated raw expert ID in model manifest");
        result.expert_raw.push_back({l,e,location(key+"path")});
    }
    if(ggufs+raws==0)throw std::invalid_argument("Model manifest has no expert sources");
    for(const auto& [key,v]:file.metadata()){
        (void)v;if(key!="general.alignment" && !used.contains(key))throw std::invalid_argument("Unknown model manifest field: "+key);
    }
    return result;
}
} // namespace strata::deepseek41
