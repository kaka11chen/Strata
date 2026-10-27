#include "strata/deepseek41/expert_file.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/deepseek41/host_expert_cache.hpp"
#include "../core/gguf_fixture.hpp"
#include <chrono>
#include <functional>
#include <limits>
#include <cstdio>
using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0;
void check(bool ok, const char* label) { ++checks; if (!ok) throw std::runtime_error(label); }
void rejects(const std::function<void()>& run, const char* label) {
    bool caught = false; try { run(); } catch (const std::exception&) { caught = true; } check(caught, label);
}
std::vector<uint8_t> raw_role(const Geometry& g, int layer, int expert, int role) {
    std::vector<uint8_t> raw(size_t(g.hidden) * size_t(g.expert_width) / 32 * 17);
    for (size_t block = 0; block < raw.size(); block += 17) {
        raw[block] = uint8_t(123 + (layer + expert + role) % 8);
        for (size_t j = 1; j <= 16; ++j) raw[block+j] = uint8_t(layer * 53 + expert * 17 + role * 31 + j);
    }
    return raw;
}
void fill(const fs::path& path, const fixture::Written& written, const Geometry& g,
          const std::vector<std::pair<int,int>>& tensor_roles) {
    std::fstream out(path,std::ios::binary|std::ios::in|std::ios::out);out.exceptions(std::ios::badbit|std::ios::failbit);
    for(size_t t=0;t<tensor_roles.size();++t){const auto [layer,role]=tensor_roles[t];
        for(int e=0;e<g.experts;++e){const auto raw=raw_role(g,layer,e,role);
            out.seekp(std::streamoff(written.data_start+written.offsets[t]+size_t(e)*raw.size()));
            out.write(reinterpret_cast<const char*>(raw.data()),std::streamsize(raw.size()));}
    }
}
std::vector<fixture::Tensor> tensors(const Geometry& g,int layer){
    return {{"blk."+std::to_string(layer)+".ffn_gate_exps.weight",{uint64_t(g.hidden),uint64_t(g.expert_width),384},39,1},
            {"blk."+std::to_string(layer)+".ffn_up_exps.weight",{uint64_t(g.hidden),uint64_t(g.expert_width),384},39,2},
            {"blk."+std::to_string(layer)+".ffn_down_exps.weight",{uint64_t(g.expert_width),uint64_t(g.hidden),384},39,3}};
}
std::vector<fixture::Kv> metadata(){return {fixture::str("general.architecture","deepseek41-index-fixture"),fixture::str("strata.deepseek41.revision",std::string(checkpoint_1017_revision))};}
std::vector<uint8_t> packed(const Geometry& g,const ExpertView& view){std::vector<uint8_t> out(nvfp4_expert_bytes(g));pack_nvfp4_expert(g,view,out);return out;}
}
int main(int argc,char** argv){
    fs::path dir;
    try{
        Geometry g;g.layers=2;g.hidden=64;g.expert_width=128;g.experts=384;g.top_k=6;
        dir=fs::temp_directory_path()/("strata-expert-index-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directories(dir);
        std::array<fs::path,3> shards;
        for(int r=0;r<3;++r){
            shards[size_t(r)]=dir/("full-0000"+std::to_string(r+1)+"-of-00003.gguf");
            auto meta=metadata();const auto split=fixture::split_keys(r,3,6);meta.insert(meta.end(),split.begin(),split.end());
            const auto written=fixture::write(shards[size_t(r)],meta,{tensors(g,0)[size_t(r)],tensors(g,1)[size_t(r)]});
            fill(shards[size_t(r)],written,g,{{0,r},{1,r}});
        }
        std::array<std::vector<uint8_t>,3> expected;
        const std::array<std::pair<int,int>,3> ids{{{0,3},{1,3},{1,383}}};
        { const auto model=GgufModel::open(shards[1].string());ExpertIndex mapped(model,g);
          for(size_t i=0;i<ids.size();++i)expected[i]=packed(g,mapped.view(ids[i].first,ids[i].second)); }
        auto index=ExpertFileIndex::open_checkpoint(g,shards[2].string());
        check(index.entries()==768 && index.files()==3,"complete multi-shard index preserves every original ID and deduplicates file paths");
        index.require_complete();check(index.at(1,383).parts[0].offset%4096!=0,"fixture exercises non-page-aligned expert offsets");
        IndexedExpertFileReader reader(index);std::vector<uint8_t> actual(nvfp4_expert_bytes(g));
        const size_t fixed=reader.staging_bytes();
        for(size_t i=0;i<ids.size();++i){reader.read(ids[i].first,ids[i].second,actual);check(actual==expected[i],"direct split-role read matches mapped reference after mappings close");}
        const auto io=reader.stats();check(io.reads==9 && io.converted==3 && io.payload_bytes==9*index.role_bytes(),"all three noncontiguous roles accounted independently");
        check(io.requested_bytes>=io.completed_bytes && io.completed_bytes>io.payload_bytes,"page overhead is separate from exact expert payload bytes");
        check(reader.staging_bytes()==fixed && index.metadata_upper_bytes()<1u<<20,"reader and index metadata remain bounded across reads");
        check(expected[0]!=expected[1],"same expert ID in different layers retains distinct weights");
        rejects([&]{index.at(2,0);},"out-of-range layer rejected");rejects([&]{index.at(0,int64_t(1)<<32);},"wide expert ID rejected before narrowing");
        const auto part=dir/"partial.gguf";auto meta=metadata();meta.push_back(fixture::u32s("strata.deepseek41.populated_ids",{3,7,9}));
        const auto written=fixture::write(part,meta,tensors(g,0));fill(part,written,g,{{0,0},{0,1},{0,2}});
        ExpertFileIndex partial(g);
        {const auto model=GgufModel::open(part.string());partial.add_gguf_layer(model,0);}
        check(partial.entries()==3 && partial.contains(0,3) && !partial.contains(0,4),"populated guard excludes unlisted tensor slots");
        rejects([&]{partial.at(0,4);},"unpopulated original expert cannot become zero weights");
        rejects([&]{partial.require_complete();},"partial index cannot claim complete checkpoint coverage");
        const auto raw_path=dir/"standalone.mxfp4";std::vector<uint8_t> raw;
        for(int r=0;r<3;++r){const auto bytes=raw_role(g,1,3,r);raw.insert(raw.end(),bytes.begin(),bytes.end());}
        auto write_raw=[&]{std::ofstream out(raw_path,std::ios::binary|std::ios::trunc);out.exceptions(std::ios::badbit|std::ios::failbit);out.write(reinterpret_cast<const char*>(raw.data()),std::streamsize(raw.size()));};write_raw();
        partial.add_raw(1,3,raw_path.string(),checkpoint_1017_revision);
        IndexedExpertFileReader mixed(partial);mixed.read(0,3,actual);check(actual==expected[0],"guarded GGUF and raw files share the same original-ID interface");
        const auto before=mixed.stats();mixed.read(1,3,actual);check(actual==expected[1] && mixed.stats().reads==before.reads+1,"contiguous raw roles use one exact demand read");
        rejects([&]{partial.add_raw(1,3,raw_path.string(),checkpoint_1017_revision);},"duplicate original ID cannot replace an existing location");
        rejects([&]{partial.add_raw(1,4,raw_path.string(),"wrong");},"wrong raw provenance rejected");
        fs::resize_file(raw_path,raw.size()-1);const auto good=actual;const auto reads=mixed.stats().reads;
        rejects([&]{mixed.read(1,3,actual);},"file truncated after indexing rejected before reading");
        check(actual==good && mixed.stats().reads==reads,"failed size check preserves destination and read count");write_raw();
        mixed.read(1,3,actual);check(actual==expected[1],"complete original file recovers without stale data");
        for(int bad=0;bad<4;++bad){auto m=metadata();auto ts=tensors(g,0);
            if(bad==0)m[1]=fixture::str("strata.deepseek41.revision","wrong");
            if(bad==1)m.push_back(fixture::u32s("strata.deepseek41.populated_ids",{3,3}));
            if(bad==2)m.push_back(fixture::strs("strata.deepseek41.populated_ids",{"3"}));
            if(bad==3)for(auto& t:ts)t.type=8;
            const auto path=dir/("bad"+std::to_string(bad)+".gguf");fixture::write(path,m,ts);
            ExpertFileIndex invalid(g);const auto model=GgufModel::open(path.string());
            rejects([&]{invalid.add_gguf_layer(model,0);},"invalid revision/coverage/type rejected before publishing locations");check(!invalid.entries(),"rejected GGUF layer publishes no entries");
        }
        const auto cut=dir/"cut.gguf";fixture::write(cut,metadata(),tensors(g,0),1);
        {ExpertFileIndex invalid(g);const auto model=GgufModel::open(cut.string());rejects([&]{invalid.add_gguf_layer(model,0);},"GGUF tensor beyond file bounds rejected");}
        // A partial guard on a non-root shard must not be lost during split discovery.
        for(int mode=0;mode<2;++mode){
            std::array<fs::path,3> guarded;
            for(int r=0;r<3;++r){
                guarded[size_t(r)]=dir/("guarded-0000"+std::to_string(r+1)+"-of-00003.gguf");
                auto m=metadata();const auto split=fixture::split_keys(r,3,3);m.insert(m.end(),split.begin(),split.end());
                if(r==1)m.push_back(fixture::u32s("strata.deepseek41.populated_ids",{3,7,9}));
                if(mode && r==0)m.push_back(fixture::u32s("strata.deepseek41.populated_ids",{3,7}));
                fixture::write(guarded[size_t(r)],m,{tensors(g,0)[size_t(r)]});
            }
            const auto model=GgufModel::open(guarded[0].string());ExpertFileIndex guarded_index(g);
            if(mode)rejects([&]{guarded_index.add_gguf_layer(model,0);},"conflicting split coverage rejected");
            else{guarded_index.add_gguf_layer(model,0);check(guarded_index.entries()==3 && !guarded_index.contains(0,4),"coverage declared on a non-root shard still guards holes");}
        }
        if(argc==4){
            Geometry real;real.layers=40;real.hidden=5120;real.expert_width=2304;real.experts=384;real.top_k=6;
            const int expert=std::stoi(argv[2]);std::vector<uint8_t> golden;
            ExpertFileIndex original(real);
            {const auto model=GgufModel::open(argv[1]);ExpertIndex mapped(model,real,{0});golden=packed(real,mapped.view(0,expert));original.add_gguf_layer(model,0);}
            original.add_raw(0,84,argv[3],checkpoint_1017_revision);
            IndexedExpertFileReader original_reader(original);std::vector<uint8_t> got(golden.size());original_reader.read(0,expert,got);check(got==golden,"original 5120/2304 GGUF range bytes equal the confirmed mapped adapter");
            Mxfp4ExpertFileReader legacy(real);legacy.read(argv[3],golden);original_reader.read(0,84,got);check(got==golden,"original standalone payload uses the identical indexed native adapter");
            std::printf("original entries=%zu files=%zu metadata_upper=%zu staging=%zu reads=%llu payload=%llu completed=%llu\n",original.entries(),original.files(),original.metadata_upper_bytes(),original_reader.staging_bytes(),(unsigned long long)original_reader.stats().reads,(unsigned long long)original_reader.stats().payload_bytes,(unsigned long long)original_reader.stats().completed_bytes);
        }else if(argc!=1)throw std::invalid_argument("Expected no arguments or original routed GGUF, original ID, raw expert file");
        fs::remove_all(dir);std::printf("%d checks, 0 failures\n",checks);return 0;
    }catch(const std::exception& e){if(!dir.empty()){std::error_code error;fs::remove_all(dir,error);}std::fprintf(stderr,"%s\n",e.what());return 1;}
}
