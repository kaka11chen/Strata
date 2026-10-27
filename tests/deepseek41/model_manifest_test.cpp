#include "strata/deepseek41/model_loader.hpp"
#include "strata/deepseek41/expert_file.hpp"
#include "../core/gguf_fixture.hpp"
#include <chrono>
#include <functional>
#include <cstdio>
using namespace strata;
using namespace strata::deepseek41;
namespace fs=std::filesystem;
namespace {
int checks=0;
void check(bool b,const char* s){++checks;if(!b)throw std::runtime_error(s);}
void rejects(const std::function<void()>& f,const char* s){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught,s);}
std::vector<fixture::Kv> metadata(){
    const std::string p=model_manifest_prefix;std::vector<fixture::Kv> m;
    auto text=[&](const std::string& k,const std::string& v){m.push_back(fixture::str(p+k,v));};
    auto number=[&](const std::string& k,uint32_t v){m.push_back(fixture::u32(p+k,v));};
    m.push_back(fixture::str("general.architecture","deepseek41-manifest"));number("version",1);
    text("revision",std::string(checkpoint_1017_revision));text("profile","1017-full40-context5");
    text("reference_root","tokenizer dir");text("text_io","weights io.gguf");number("experts_complete",0);
    for(int l=0;l<40;++l){const auto k="layer."+std::to_string(l)+".";
        for(const char* role:{"mhc","attention","shared","router"})text(k+role,"part "+std::to_string(l)+".gguf");
        text(k+"router_weight","router.weight");text(k+"router_bias","router.bias");
    }
    for(int l:{1,14}){const auto k="engram."+std::to_string(l)+".";
        for(const char* role:{"projection","token_map","table"})text(k+role,std::to_string(l)+"-"+role+".gguf");
        number(k+"complete",0);
    }
    number("expert_gguf_count",1);number("expert_gguf.0.layer",3);text("expert_gguf.0.path","experts.gguf");
    number("expert_raw_count",1);number("expert_raw.0.layer",1);number("expert_raw.0.expert",383);text("expert_raw.0.path","raw weights.bin");
    return m;
}
}
int main(int argc,char** argv){
    fs::path dir;
    try{
        dir=fs::temp_directory_path()/("strata-model-manifest-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(dir);
        const auto path=dir/"model.gguf";fixture::write(path,metadata(),{});const auto m=ModelManifest::read(path.string());
        check(m.layers[39].attention==(dir/"part 39.gguf").string() && m.reference_root==(dir/"tokenizer dir").string(),"paths resolve relative to manifest, preserving spaces without opening payloads");
        check(m.expert_raw.size()==1 && m.expert_raw[0].layer==1 && m.expert_raw[0].expert==383 && m.expert_gguf[0].layer==3,"manifest retains original expert identities");
        check(!m.experts_complete && !m.engram[0].complete && !m.engram[1].complete,"guarded coverage stays explicit");
        const auto g=model_1017_geometry();check(g.layers==40 && g.compress_ratios[1]==0 && g.compress_ratios[14]==2 && g.compress_ratios[20]==1 && g.sources[35].index==32 && g.sources[36].candidates==20 && g.sources[39].kv==20,"runtime geometry preserves original shared-state owners");
        const std::string p=model_manifest_prefix;
        for(int bad=0;bad<11;++bad){auto mm=metadata();
            auto set=[&](const std::string& k,fixture::Kv value){for(auto& item:mm)if(item.key==p+k){item=std::move(value);return;}throw std::logic_error("missing test key");};
            if(bad==0)set("version",fixture::u32(p+"version",2));
            if(bad==1)set("revision",fixture::str(p+"revision","wrong"));
            if(bad==2)set("experts_complete",fixture::u32(p+"experts_complete",2));
            if(bad==3)set("expert_raw.0.expert",fixture::u32(p+"expert_raw.0.expert",384));
            if(bad==4)set("expert_gguf.0.layer",fixture::u32(p+"expert_gguf.0.layer",40));
            if(bad==5)set("expert_raw_count",fixture::u32(p+"expert_raw_count",15361));
            if(bad==6)set("layer.39.attention",fixture::str(p+"layer.39.attention",""));
            if(bad==7)mm.push_back(fixture::str(p+"expert_complete","typo"));
            if(bad==8)set("engram.14.complete",fixture::str(p+"engram.14.complete","false"));
            if(bad==9){set("expert_raw_count",fixture::u32(p+"expert_raw_count",2));mm.push_back(fixture::u32(p+"expert_raw.1.layer",1));mm.push_back(fixture::u32(p+"expert_raw.1.expert",383));mm.push_back(fixture::str(p+"expert_raw.1.path","other.bin"));}
            if(bad==10){set("expert_gguf_count",fixture::u32(p+"expert_gguf_count",2));mm.push_back(fixture::u32(p+"expert_gguf.1.layer",3));mm.push_back(fixture::str(p+"expert_gguf.1.path","other.gguf"));}
            fixture::write(path,mm,{});rejects([&]{ModelManifest::read(path.string());},"invalid schema/identity/coverage/ID/count/duplicate rejected");
        }
        fixture::write(path,metadata(),{{"payload",{1},0,1}});rejects([&]{ModelManifest::read(path.string());},"manifest cannot hide a payload tensor");
        fixture::write(path,metadata(),{});fs::resize_file(path,8u*1024*1024+1);rejects([&]{ModelManifest::read(path.string());},"oversized manifest rejected before mapping/parsing");
        if(argc==2){
            std::unique_ptr<LoadedModel> loaded;
            {const auto original=ModelManifest::read(argv[1]);loaded=std::make_unique<LoadedModel>(original);}
            check(loaded->weights.layers.size()==40 && loaded->experts.entries()==1702 && loaded->text_io.vocab_size==129280,"original full40 components survive manifest lifetime");
            check(loaded->weights.layers[39].attention.layer==39 && loaded->weights.engram.layer==1 && loaded->weights.engram14.layer==14,"layer and Engram identities remain original");
            check(loaded->token_maps[0]==loaded->token_maps[1] && loaded->tables[0]!=loaded->tables[1],"same pinned token map with independent Engram tables");
            IndexedExpertFileReader reader(loaded->experts);std::vector<uint8_t> native(19906576);reader.read(0,84,native);
            check(reader.stats().converted==1 && reader.stats().failures==0,"loaded model source survives temporary expert mappings");
            const auto original=ModelManifest::read(argv[1]);
            const auto wrong=dir/"wrong-revision.gguf";fixture::write(wrong,{fixture::str("strata.deepseek41.revision","wrong")},{});
            for(int bad=0;bad<8;++bad){auto broken=original;
                if(bad==0)broken.experts_complete=true;
                if(bad==1)broken.engram[0].complete=true;
                if(bad==2)broken.engram[0].table=original.engram[1].table;
                if(bad==3)broken.layers[2].attention=original.layers[8].attention;
                if(bad==4)broken.layers[0].router_weight="missing";
                if(bad==5)broken.layers[0].mhc=wrong.string();
                if(bad==6)broken.engram[0].token_map=original.engram[1].token_map;
                if(bad==7)broken.expert_raw[0].path=(dir/"absent.bin").string();
                rejects([&]{LoadedModel rejected(broken);},"original loader rejects false coverage, wrong layer/revision, or missing tensor/file");
            }
            std::printf("original mapped_files=%zu entries=%zu retained_expert_upper=%llu\n",loaded->mapped_files(),loaded->experts.entries(),(unsigned long long)loaded->retained_expert_mapping_upper_bytes);
        }else if(argc!=1)throw std::invalid_argument("Expected optional original model manifest");
        fs::remove_all(dir);std::printf("%d checks, 0 failures\n",checks);return 0;
    }catch(const std::exception& e){if(!dir.empty()){std::error_code error;fs::remove_all(dir,error);}std::fprintf(stderr,"%s\n",e.what());return 1;}
}
