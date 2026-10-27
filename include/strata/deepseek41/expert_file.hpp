#pragma once
#include "strata/deepseek41/artifact.hpp"
#include "strata/deepseek41/expert_file_index.hpp"
#include "strata/platform/direct_file.hpp"
#include <memory>

namespace strata::deepseek41 {
struct ExpertFileStats {
    uint64_t reads = 0, requested_bytes = 0, completed_bytes = 0, converted = 0, failures = 0, payload_bytes = 0;
};
// Demand-read one original [gate|up|down] MXFP4 payload into one aligned buffer,
// then reuse the lossless native adapter. No mapping, buffered fallback or read-ahead.
// Single caller, one open file/read at a time. Caller owns the native destination.
class Mxfp4ExpertFileReader {
  public:
    explicit Mxfp4ExpertFileReader(const Geometry&);
    void read(const std::string& path, std::span<uint8_t> native);
    size_t raw_bytes() const { return raw_bytes_; }
    size_t staging_bytes() const { return aligned_bytes_; }
    ExpertFileStats stats() const { return stats_; }
  private:
    Geometry geometry_;
    size_t native_bytes_, raw_bytes_, aligned_bytes_;
    std::unique_ptr<void, decltype(&platform::DirectFile::free_aligned)> staging_;
    ExpertFileStats stats_;
};
// Indexed ranges can start inside an unaligned GGUF page and span three files.
// Owns one aligned read slab plus one assembled raw expert; no mapped payloads.
// The index outlives the reader. One caller, one outstanding DirectFile request.
class IndexedExpertFileReader {
  public:
    explicit IndexedExpertFileReader(const ExpertFileIndex&);
    void read(int64_t layer, int64_t expert, std::span<uint8_t> native);
    size_t staging_bytes() const { return aligned_bytes_ + raw_.size(); }
    ExpertFileStats stats() const { return stats_; }
  private:
    void read_span(const ExpertFilePart&, size_t bytes, uint8_t* destination);
    const ExpertFileIndex& index_;
    size_t aligned_bytes_;
    std::unique_ptr<void, decltype(&platform::DirectFile::free_aligned)> staging_;
    std::vector<uint8_t> raw_;
    ExpertFileStats stats_;
};
} // namespace strata::deepseek41
