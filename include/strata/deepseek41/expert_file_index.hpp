#pragma once
#include "strata/deepseek41/artifact.hpp"
#include <string_view>

namespace strata::deepseek41 {
inline constexpr std::string_view checkpoint_1017_revision = "1bae0e1088993b9b82de08f59de5be5a34e42170";
struct ExpertSourceFile { std::string path; uint64_t bytes = 0; };
struct ExpertFilePart { uint32_t file = 0; uint64_t offset = 0; };
struct ExpertFileLocation { std::array<ExpertFilePart, 3> parts{}; bool present = false; };
// Original-ID locations only: owns file names/offsets, never mapped payloads.
// GGUF ranges reuse ExpertIndex validation. Populated/selected-ID guards are
// honored; missing experts never acquire a zero-filled location. Input files must
// remain immutable while the source is in use. Revision metadata is provenance
// agreement, not a replacement for offline payload/hash verification.
class ExpertFileIndex {
  public:
    explicit ExpertFileIndex(const Geometry&);
    void add_gguf_layer(const GgufModel&, int layer);
    void add_raw(int layer, int expert, const std::string& path, std::string_view revision);
    // Complete MXFP4 expert GGUF, including split files. Other model weights are
    // not loaded here. Fails unless every original expert is represented.
    static ExpertFileIndex open_checkpoint(const Geometry&, const std::string& path);
    const ExpertFileLocation& at(int64_t layer, int64_t expert) const;
    bool contains(int64_t layer, int64_t expert) const;
    const ExpertSourceFile& file(uint32_t id) const;
    const Geometry& geometry() const { return geometry_; }
    size_t role_bytes() const { return role_bytes_; }
    size_t entries() const { return count_; }
    size_t files() const { return files_.size(); }
    void require_complete() const;
    // Conservative accounting for the fixed lookup array, file table, path map
    // nodes/string capacity and geometry vectors; allocator headroom is separate.
    size_t metadata_upper_bytes() const;
  private:
    size_t key(int64_t layer, int64_t expert) const;
    uint32_t add_file(const std::string&, uint64_t expected_bytes);
    Geometry geometry_;
    size_t role_bytes_, count_ = 0;
    std::vector<ExpertFileLocation> locations_;
    std::vector<ExpertSourceFile> files_;
    std::map<std::string, uint32_t> file_ids_;
};
} // namespace strata::deepseek41
