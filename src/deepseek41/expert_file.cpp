#include "strata/deepseek41/expert_file.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include <stdexcept>
#include <cstring>

namespace strata::deepseek41 {
Mxfp4ExpertFileReader::Mxfp4ExpertFileReader(const Geometry& g)
    : geometry_(g), native_bytes_(nvfp4_expert_bytes(g)),
      raw_bytes_(size_t(g.hidden) * size_t(g.expert_width) / 32 * 17 * 3),
      aligned_bytes_((raw_bytes_ + platform::DirectFile::alignment() - 1) /
                     platform::DirectFile::alignment() * platform::DirectFile::alignment()),
      staging_(platform::DirectFile::alloc_aligned(aligned_bytes_), &platform::DirectFile::free_aligned) {
    if (!staging_) throw std::bad_alloc();
}
void Mxfp4ExpertFileReader::read(const std::string& path, std::span<uint8_t> native) {
    if (!native.data() || native.size() != native_bytes_)
        throw std::invalid_argument("Expert file read requires one full native destination");
    try {
        platform::DirectFile file; std::string error;
        if (!file.open(path, error)) throw std::runtime_error(error);
        if (file.size() != raw_bytes_) throw std::runtime_error("Original MXFP4 expert file has the wrong size: " + path);
        if (!file.submit(0, staging_.get(), uint32_t(aligned_bytes_), 1, error)) throw std::runtime_error(error);
        ++stats_.reads; stats_.requested_bytes += aligned_bytes_;
        platform::Completion completion;
        const int n = file.wait(&completion, 1, -1);
        if (n == 1) stats_.completed_bytes += completion.bytes;
        if (n != 1 || completion.tag != 1 || !completion.ok || completion.bytes != raw_bytes_)
            throw std::runtime_error("Incomplete direct read of original MXFP4 expert: " + path);
        const auto* raw = static_cast<const uint8_t*>(staging_.get()); const size_t part = raw_bytes_ / 3;
        ExpertView view;
        for (size_t r = 0; r < 3; ++r) { view.parts[r].data = {raw + r * part, part}; view.parts[r].type = 39; }
        pack_nvfp4_expert(geometry_, view, native); ++stats_.converted; stats_.payload_bytes += raw_bytes_;
    } catch (...) { ++stats_.failures; throw; }
}
IndexedExpertFileReader::IndexedExpertFileReader(const ExpertFileIndex& index)
    : index_(index),
      aligned_bytes_((3 * index.role_bytes() + 2 * platform::DirectFile::alignment() - 2) /
                     platform::DirectFile::alignment() * platform::DirectFile::alignment()),
      staging_(platform::DirectFile::alloc_aligned(aligned_bytes_), &platform::DirectFile::free_aligned),
      raw_(3 * index.role_bytes()) {
    if (!staging_) throw std::bad_alloc();
}
void IndexedExpertFileReader::read_span(const ExpertFilePart& part, size_t bytes, uint8_t* destination) {
    const auto& source = index_.file(part.file); platform::DirectFile file; std::string error;
    if (!file.open(source.path, error)) throw std::runtime_error(error);
    if (file.size() != source.bytes || part.offset > source.bytes || bytes > source.bytes - part.offset)
        throw std::runtime_error("Indexed expert source size/range changed: " + source.path);
    constexpr uint64_t alignment = platform::DirectFile::alignment();
    const uint64_t begin = part.offset / alignment * alignment;
    const size_t prefix = size_t(part.offset - begin), request = (prefix + bytes + alignment - 1) / alignment * alignment;
    if (request > aligned_bytes_) throw std::logic_error("Indexed expert read exceeds its fixed slab");
    if (!file.submit(begin, staging_.get(), uint32_t(request), 1, error)) throw std::runtime_error(error);
    ++stats_.reads; stats_.requested_bytes += request;
    platform::Completion completion; const int n = file.wait(&completion, 1, -1);
    if (n == 1) stats_.completed_bytes += completion.bytes;
    if (n != 1 || completion.tag != 1 || !completion.ok || completion.bytes < prefix + bytes || completion.bytes > request)
        throw std::runtime_error("Incomplete indexed expert read: " + source.path);
    std::memcpy(destination, static_cast<const uint8_t*>(staging_.get()) + prefix, bytes); stats_.payload_bytes += bytes;
}
void IndexedExpertFileReader::read(int64_t layer, int64_t expert, std::span<uint8_t> native) {
    if (!native.data() || native.size() != nvfp4_expert_bytes(index_.geometry()))
        throw std::invalid_argument("Indexed expert read needs one complete native destination");
    try {
        const auto& parts = index_.at(layer, expert).parts; const size_t bytes = index_.role_bytes();
        const bool contiguous = parts[0].file == parts[1].file && parts[0].file == parts[2].file &&
                                parts[1].offset == parts[0].offset + bytes && parts[2].offset == parts[1].offset + bytes;
        if (contiguous) read_span(parts[0], raw_.size(), raw_.data());
        else for (size_t role = 0; role < 3; ++role) read_span(parts[role], bytes, raw_.data() + role * bytes);
        ExpertView view;
        for (size_t role = 0; role < 3; ++role) { view.parts[role].data = {raw_.data() + role * bytes, bytes}; view.parts[role].type = 39; }
        pack_nvfp4_expert(index_.geometry(), view, native); ++stats_.converted;
    } catch (...) { ++stats_.failures; throw; }
}
} // namespace strata::deepseek41
