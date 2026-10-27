#pragma once
#include "strata/deepseek41/expert_file_index.hpp"

namespace strata::deepseek41 {
inline constexpr const char* model_manifest_prefix = "strata.deepseek41.manifest.";
struct LayerFiles {
    std::string mhc, attention, shared, router, router_weight, router_bias;
};
struct EngramFiles { std::string projection, token_map, table; bool complete = false; };
struct ExpertGgufFile { int layer; std::string path; };
struct ExpertRawFile { int layer, expert; std::string path; };
// Metadata-only GGUF manifest. Paths resolve against its directory, never cwd.
// Reading checks the manifest contract, not referenced payload existence/identity.
struct ModelManifest {
    std::string reference_root, text_io;
    std::array<LayerFiles, 40> layers;
    std::array<EngramFiles, 2> engram; // original layers 1 and 14
    std::vector<ExpertGgufFile> expert_gguf;
    std::vector<ExpertRawFile> expert_raw;
    bool experts_complete = false;
    size_t metadata_upper_bytes() const;
    static ModelManifest read(const std::string& path);
};
Geometry model_1017_geometry();
} // namespace strata::deepseek41
