// DeepSeek-V4.1-Flash-1017 artifact access. This is not an inference backend.
// Metadata / mixed-expert names follow Xolotl; see src/deepseek41/NOTICE.md.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <map>
#include <span>
#include <vector>

namespace strata::deepseek41 {

struct LayerSources {
    int kv = -1;
    int index = -1;
    int candidates = -1;
};

struct Geometry {
    int layers = 0, hidden = 0, expert_width = 0, experts = 0, top_k = 0;
    std::vector<int> compress_ratios;
    std::vector<LayerSources> sources;
};

// Checks the core 1017 text profile exported by Xolotl's v41_gguf_meta.py.
// This does not establish checkpoint identity or validate dense / Engram / DSpark weights.
// A missing, wrong-typed or mismatched field throws, naming that field.
Geometry read_1017_geometry(const GgufFile& file);

struct ExpertPart {
    std::span<const uint8_t> data;
    uint32_t type = 0;
    size_t shard = 0;
    uint64_t file_offset = 0;
};

struct ExpertView {
    std::array<ExpertPart, 3> parts; // gate, up, down; possibly in different shards / formats
    size_t bytes() const;
};

// Reuses Strata's mapped, validated shard set. Does not allocate or decode expert weights.
// The model must outlive the index and every view. Mxfp4ExpertSource adapts this
// byte index to Strata storage without using Qwen's global layout or execution rules.
class ExpertIndex {
  public:
    // Empty `layers` means ALL backbone layers. A partial check must explicitly name its layers.
    ExpertIndex(const GgufModel& model, const Geometry& geometry, std::vector<int> layers = {});
    const ExpertView& view(int layer, int expert) const;
    // Caller owns a buffer of exactly view(...).bytes(); no accumulating expert cache here.
    void copy(int layer, int expert, std::span<uint8_t> destination) const;
    size_t layer_count() const { return layers_.size(); }
    uint64_t total_bytes() const { return total_bytes_; }

  private:
    std::map<int, std::vector<ExpertView>> layers_;
    uint64_t total_bytes_ = 0;
};

} // namespace strata::deepseek41
