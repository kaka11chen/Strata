#pragma once
#include "strata/deepseek41/model_manifest.hpp"
#include "strata/deepseek41/text_prefix_cuda.hpp"
#include "strata/deepseek41/text_io_weights.hpp"

namespace strata::deepseek41 {
// Host-only owner for views consumed by the existing native CUDA components.
// Files use the confirmed component GGUF layouts; no fixture names or test code.
// A manifest declares coverage; referenced metadata/ranges must validate it.
// Source files remain immutable. Revision agreement is not payload authentication.
class LoadedModel {
    std::map<std::string,std::unique_ptr<GgufModel>> models_;
    const GgufModel& open(const std::string&);
  public:
    explicit LoadedModel(const ModelManifest&);
    LoadedModel(const LoadedModel&)=delete;
    LoadedModel& operator=(const LoadedModel&)=delete;
    const Geometry geometry=model_1017_geometry();
    TextPrefixWeights weights;
    TextIoWeights text_io;
    std::array<std::vector<uint32_t>,2> token_maps;
    std::array<std::string,2> tables;
    std::array<bool,2> engram_complete{};
    bool experts_complete=false;
    ExpertFileIndex experts{geometry}; // may move into IndexedExpertSource after loading
    uint64_t retained_expert_mapping_upper_bytes=0;
    size_t mapped_files() const { return models_.size(); }
};
} // namespace strata::deepseek41
