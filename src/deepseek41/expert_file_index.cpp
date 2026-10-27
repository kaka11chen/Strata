#include "strata/deepseek41/expert_file_index.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace strata::deepseek41 {
namespace {
void revision(const GgufModel& model) {
    bool declared = false;
    for (size_t shard = 0; shard < model.size(); ++shard) {
        for (const char* key : {"strata.deepseek41.revision", "strata.deepseek41.model_revision"}) {
            if (const auto* value = model.shard(shard).get(key)) {
                if (value->type != MetaType::STRING || value->s != checkpoint_1017_revision)
                    throw std::invalid_argument("Expert GGUF revision differs from pinned 1017");
                declared = true;
            }
        }
    }
    if (!declared) throw std::invalid_argument("Expert GGUF needs pinned 1017 revision metadata");
}
std::vector<int> coverage(const GgufModel& model, int experts) {
    // Split files may keep guards outside shard zero. Prefer populated coverage
    // globally; selected_ids is only the legacy fallback and can be a subset.
    for (const char* key : {"strata.deepseek41.populated_ids", "strata.deepseek41.selected_ids"}) {
        bool declared = false; std::vector<int> result;
        for (size_t shard = 0; shard < model.size(); ++shard) {
            const auto* ids = model.shard(shard).get(key);
            if (!ids) continue;
            if (ids->type != MetaType::ARRAY || ids->count != ids->items.size() || !ids->count || ids->count > uint64_t(experts))
                throw std::invalid_argument("Malformed populated expert coverage");
            std::vector<int> parsed; std::vector<bool> seen(size_t(experts), false);
            for (const auto& id : ids->items) {
                if (id.type != MetaType::U32 || id.u >= uint64_t(experts) || seen[size_t(id.u)])
                    throw std::invalid_argument("Invalid/repeated original expert in populated coverage");
                seen[size_t(id.u)] = true; parsed.push_back(int(id.u));
            }
            std::sort(parsed.begin(), parsed.end());
            if (declared && parsed != result) throw std::invalid_argument("Conflicting populated coverage across GGUF shards");
            result = std::move(parsed); declared = true;
        }
        if (declared) return result;
    }
    std::vector<int> all(size_t(experts), 0); std::iota(all.begin(), all.end(), 0); return all;
}
} // namespace
ExpertFileIndex::ExpertFileIndex(const Geometry& g) : geometry_(g) {
    (void)nvfp4_expert_bytes(g);
    if (g.layers < 1 || g.layers > 40 || g.experts != 384 || g.top_k != 6)
        throw std::invalid_argument("Expert file index requires 1017 backbone geometry");
    role_bytes_ = size_t(g.hidden) * size_t(g.expert_width) / 32 * 17;
    locations_.resize(size_t(g.layers) * size_t(g.experts));
}
size_t ExpertFileIndex::key(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= geometry_.layers || expert < 0 || expert >= geometry_.experts)
        throw std::out_of_range("Original layer/expert outside file index geometry");
    return size_t(layer * geometry_.experts + expert);
}
bool ExpertFileIndex::contains(int64_t layer, int64_t expert) const { return locations_[key(layer, expert)].present; }
const ExpertFileLocation& ExpertFileIndex::at(int64_t layer, int64_t expert) const {
    const auto& location = locations_[key(layer, expert)];
    if (!location.present) throw std::runtime_error("Missing indexed original expert " + std::to_string(layer) + "/" + std::to_string(expert));
    return location;
}
const ExpertSourceFile& ExpertFileIndex::file(uint32_t id) const { return files_.at(id); }
uint32_t ExpertFileIndex::add_file(const std::string& name, uint64_t bytes) {
    const auto path = std::filesystem::canonical(name).string();
    if (std::filesystem::file_size(path) != bytes) throw std::runtime_error("Expert source file size changed: " + path);
    if (const auto it = file_ids_.find(path); it != file_ids_.end()) {
        if (files_[it->second].bytes != bytes) throw std::runtime_error("Conflicting expert source file sizes");
        return it->second;
    }
    if (files_.size() >= 3 * locations_.size()) throw std::runtime_error("Expert index file count exceeds geometry");
    const auto id = uint32_t(files_.size()); files_.push_back({path, bytes}); file_ids_.emplace(path, id); return id;
}
void ExpertFileIndex::add_raw(int layer, int expert, const std::string& path, std::string_view rev) {
    const auto k = key(layer, expert);
    if (rev != checkpoint_1017_revision) throw std::invalid_argument("Raw expert provenance differs from pinned 1017");
    if (locations_[k].present) throw std::invalid_argument("Duplicate original expert file location");
    const auto id = add_file(path, 3 * role_bytes_); ExpertFileLocation location; location.present = true;
    for (size_t role = 0; role < 3; ++role) location.parts[role] = {id, role * role_bytes_};
    locations_[k] = location; ++count_;
}
void ExpertFileIndex::add_gguf_layer(const GgufModel& model, int layer) {
    (void)key(layer, 0); revision(model); const auto ids = coverage(model, geometry_.experts);
    const ExpertIndex checked(model, geometry_, {layer});
    // Validate the whole requested layer before publishing any of its locations.
    for (int expert : ids) {
        if (contains(layer, expert)) throw std::invalid_argument("Duplicate original expert GGUF location");
        for (const auto& part : checked.view(layer, expert).parts) {
            const auto& shard = model.shard(part.shard);
            if (part.type != 39 || part.data.size() != role_bytes_ || part.file_offset > shard.file_size() ||
                role_bytes_ > shard.file_size() - part.file_offset)
                throw std::invalid_argument("Expert file source needs bounded, shape-matched MXFP4 role slices");
        }
    }
    std::vector<uint32_t> shard_files(model.size(), ~uint32_t{0});
    for (int expert : ids) {
        ExpertFileLocation location; location.present = true;
        const auto& view = checked.view(layer, expert);
        for (size_t role = 0; role < 3; ++role) {
            const auto& part = view.parts[role]; const auto& shard = model.shard(part.shard);
            auto& id = shard_files[part.shard];
            if (id == ~uint32_t{0}) id = add_file(shard.path(), shard.file_size());
            location.parts[role] = {id, part.file_offset};
        }
        locations_[key(layer, expert)] = location; ++count_;
    }
}
void ExpertFileIndex::require_complete() const {
    if (count_ != locations_.size()) throw std::runtime_error("Expert checkpoint does not cover every original layer/expert");
}
ExpertFileIndex ExpertFileIndex::open_checkpoint(const Geometry& g, const std::string& path) {
    const auto model = GgufModel::open(path); ExpertFileIndex result(g);
    for (int layer = 0; layer < g.layers; ++layer) result.add_gguf_layer(model, layer);
    result.require_complete(); return result;
}
size_t ExpertFileIndex::metadata_upper_bytes() const {
    size_t bytes = sizeof(*this) + locations_.capacity() * sizeof(ExpertFileLocation) +
                   files_.capacity() * sizeof(ExpertSourceFile) + geometry_.compress_ratios.capacity() * sizeof(int) +
                   geometry_.sources.capacity() * sizeof(LayerSources);
    for (const auto& f : files_) bytes += f.path.capacity() + 1;
    for (const auto& [path, id] : file_ids_) { (void)id; bytes += sizeof(decltype(file_ids_)::value_type) + 64 + path.capacity() + 1; }
    return bytes;
}
} // namespace strata::deepseek41
