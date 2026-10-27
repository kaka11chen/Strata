#include "strata/deepseek41/engram_projection.hpp"
#include "../core/gguf_fixture.hpp"
#include <chrono>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name); }
void rejects(const char* name, const std::function<void()>& f) {
    try { f(); check(false,name); } catch (const std::exception&) { check(true,name); }
}
// Reuse the fixture header writer with tiny dummy payloads, patch only shapes
// and offsets, then extend sparsely. No 150 MiB payload is allocated or filled.
void write(const fs::path& path, std::vector<fixture::Tensor> ts, const std::string& profile = "1017-layer1-wkv-fp8-block32") {
    auto small = ts; for (auto& t : small) for (auto& d : t.shape) d = 1;
    const auto w = fixture::write(path, {fixture::str("strata.deepseek41.engram_projection_profile",profile)}, small);
    std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out); f.exceptions(std::ios::badbit|std::ios::failbit);
    uint64_t at = 0;
    // Header has one string metadata pair. Tensor directory starts after it.
    f.seekg(24); uint64_t n = 0; f.read(reinterpret_cast<char*>(&n),8); f.seekg(std::streamoff(n+4),std::ios::cur);
    f.read(reinterpret_cast<char*>(&n),8); f.seekg(std::streamoff(n),std::ios::cur);
    for (const auto& t : ts) {
        f.read(reinterpret_cast<char*>(&n),8); f.seekg(std::streamoff(n+4),std::ios::cur);
        const auto shape_pos = f.tellg();
        f.seekp(shape_pos); for (auto d : t.shape) f.write(reinterpret_cast<const char*>(&d),8);
        f.seekp(4,std::ios::cur); f.write(reinterpret_cast<const char*>(&at),8); f.seekg(f.tellp());
        TensorInfo info; info.shape=t.shape; info.type=t.type; at=(at+tensor_payload_bytes(info)+31)/32*32;
    }
    f.close(); fs::resize_file(path,w.data_start+at);
}
}
int main() {
    const auto dir=fs::temp_directory_path()/("strata-engram-projection-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir); const auto path=dir/"weights.gguf";
    try {
        for (int layer : {1,14}) {
            std::printf("projection_weight_layer=%d\n",layer);
            const std::string prefix="blk."+std::to_string(layer)+".engram.";
            const std::string profile="1017-layer"+std::to_string(layer)+"-wkv-fp8-block32";
            const std::vector<fixture::Tensor> tensors{{prefix+"wkv.weight",{6144,25600},24,0},
                {prefix+"wkv.scale",{192,800},24,0},{prefix+"q_weight",{5120,4},30,0},{prefix+"k_weight",{5120,4},30,0}};
            write(path,tensors,profile);
            { const auto m=GgufModel::open(path.string()); const auto w=read_engram_weights(m,layer);
              check(w.layer==layer && w.wkv.rows==25600 && w.wkv.cols==6144 && w.wkv.values.size()==157286400 && w.wkv.scales.size()==153600 && w.q.size()==20480 && w.k.size()==20480,"original layer identity and projection/per-stream weight shapes");
              rejects("other original layer cannot load this profile",[&]{(void)read_engram_weights(m,layer==1?14:1);});
              if (layer==1) {
                  const auto legacy=read_layer1_engram_weights(m);
                  check(legacy.layer==1 && legacy.wkv.values.data()==w.wkv.values.data() && legacy.q.data()==w.q.data(),"legacy loader retains layer-1 identity and borrowed views");
                  for (int bad : {-1,0,2,13,15,40})
                      rejects("unsupported original projection layer",[&]{(void)read_engram_weights(m,bad);});
              } else rejects("legacy layer-1 loader rejects layer-14 profile",[&]{(void)read_layer1_engram_weights(m);});
            }
            for (size_t i=0;i<tensors.size();++i) {
                auto bad=tensors; bad.erase(bad.begin()+ptrdiff_t(i)); write(path,bad,profile);
                rejects("missing required tensor",[&]{const auto m=GgufModel::open(path.string()); (void)read_engram_weights(m,layer);});
                bad=tensors; bad[i].shape[0]/=2; write(path,bad,profile);
                rejects("wrong tensor shape",[&]{const auto m=GgufModel::open(path.string()); (void)read_engram_weights(m,layer);});
                bad=tensors; bad[i].type=0; write(path,bad,profile);
                rejects("wrong tensor dtype",[&]{const auto m=GgufModel::open(path.string()); (void)read_engram_weights(m,layer);});
                bad=tensors; bad[i].name.replace(0,prefix.size(),"blk."+std::to_string(layer==1?14:1)+".engram."); write(path,bad,profile);
                rejects("correct profile with other layer's tensor name",[&]{const auto m=GgufModel::open(path.string()); (void)read_engram_weights(m,layer);});
            }
            write(path,tensors,profile); fs::resize_file(path,fs::file_size(path)-1);
            rejects("truncated final weight",[&]{const auto m=GgufModel::open(path.string()); (void)read_engram_weights(m,layer);});
            write(path,tensors,profile); fs::resize_file(path,4096);
            rejects("truncated projection",[&]{const auto m=GgufModel::open(path.string()); (void)read_engram_weights(m,layer);});
        }
    } catch(const std::exception& e) { std::fprintf(stderr,"ERROR %s\n",e.what()); ++failures; }
    fs::remove_all(dir); std::printf("%d checks, %d failures\n",checks,failures); return failures?1:0;
}
