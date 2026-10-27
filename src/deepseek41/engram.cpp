#include "strata/deepseek41/engram.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cmath>
#include <numeric>

namespace strata::deepseek41 {
namespace {
constexpr int max_position = 1048576;
int checked_layer(int layer) {
    if (layer != 1 && layer != 14) throw std::invalid_argument("Engram requires original layer 1 or 14");
    return layer;
}
uint32_t table_rows(int layer) { return layer == 1 ? engram1_rows : engram14_rows; }
constexpr char tokenizer_hash[] = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
void profile(const GgufFile& file, const char* key, const std::string& value) {
    const auto* m = file.get(key);
    if (!m || m->type != MetaType::STRING || m->s != value)
        throw std::invalid_argument(std::string("Missing/incorrect Engram profile: ") + key);
}
void buffer(const void* p, size_t bytes, size_t alignment) {
    if (!p || uintptr_t(p) % alignment || bytes > UINTPTR_MAX - uintptr_t(p))
        throw std::invalid_argument("Engram needs aligned, valid caller buffers");
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    return an && bn && (uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn);
}
void validate_map(std::span<const uint32_t> map) {
    if (map.size() != engram_token_count) throw std::invalid_argument("Engram token map must contain 129280 entries");
    buffer(map.data(), map.size_bytes(), 4);
    if (map[2] != 2 || std::any_of(map.begin(), map.end(), [](uint32_t x) { return x >= engram_compressed_tokens; }))
        throw std::invalid_argument("Engram token map has an invalid compressed ID/pad token");
}
static_assert(std::accumulate(engram1_primes.begin(), engram1_primes.end(), uint32_t(0)) == engram1_rows);
static_assert(std::accumulate(engram14_primes.begin(), engram14_primes.end(), uint32_t(0)) == engram14_rows);
} // namespace
std::vector<uint32_t> read_engram_token_map(const GgufFile& file, int layer) {
    checked_layer(layer);
    profile(file, "strata.deepseek41.engram_hash_profile", "1017-layer" + std::to_string(layer) + "-compressed-4gram");
    profile(file, "strata.deepseek41.tokenizer_sha256", tokenizer_hash);
    const auto* t = file.find("engram.token_map");
    const size_t bytes = engram_token_count * sizeof(uint32_t);
    if (!t || t->type != 24 || t->shape != std::vector<uint64_t>{4, engram_token_count} ||
        t->offset > file.file_size() - file.data_start() || bytes > file.file_size() - file.data_start() - t->offset)
        throw std::invalid_argument("Malformed Engram uint32 token-map payload");
    std::vector<uint32_t> map(engram_token_count);
    std::memcpy(map.data(), file.tensor_data(*t), bytes);
    validate_map(map); return map;
}
EngramHash::EngramHash(int layer, std::span<const uint32_t> map) : layer_(checked_layer(layer)) {
    validate_map(map); map_.assign(map.begin(), map.end());
}
void EngramHash::reset() noexcept { next_ = 0; previous_.fill(-1); }
void EngramHash::forward(std::span<const int32_t> tokens, std::span<uint32_t> rows,
                              int start, std::span<const uint8_t> live) {
    if (tokens.empty() || tokens.size() > max_position || start < 0 || start != next_ ||
        tokens.size() > size_t(max_position - start) || rows.size() != tokens.size() * engram_heads ||
        (!live.empty() && live.size() != tokens.size()))
        throw std::invalid_argument("Engram hash requires dimension-matched contiguous chunks within 1048576 positions");
    buffer(tokens.data(), tokens.size_bytes(), 4); buffer(rows.data(), rows.size_bytes(), 4);
    if (!live.empty()) buffer(live.data(), live.size_bytes(), 1);
    if (overlap(tokens.data(), tokens.size_bytes(), rows.data(), rows.size_bytes()) ||
        overlap(live.data(), live.size_bytes(), rows.data(), rows.size_bytes()) ||
        overlap(live.data(), live.size_bytes(), tokens.data(), tokens.size_bytes()))
        throw std::invalid_argument("Engram hash input/output buffers overlap");
    for (int32_t token : tokens)
        if (token < 0 || size_t(token) >= map_.size()) throw std::invalid_argument("Engram original token ID is out of range");
    for (uint8_t v : live) if (v > 1) throw std::invalid_argument("Engram live mask must contain only 0 or 1");
    const auto& multipliers = layer_ == 1 ? engram1_multipliers : engram14_multipliers;
    const auto& primes = layer_ == 1 ? engram1_primes : engram14_primes;
    for (size_t t = 0; t < tokens.size(); ++t) {
        const int32_t current = live.empty() || live[t] ? int32_t(map_[size_t(tokens[t])]) : -1;
        const std::array<int32_t, 4> context{current, previous_[0], previous_[1], previous_[2]};
        bool blocked = false; uint64_t mixed = 0; uint32_t offset = 0;
        for (size_t shift = 0; shift < 4; ++shift) {
            blocked |= context[shift] == -1;
            const uint32_t token = blocked ? map_[2] : uint32_t(context[shift]);
            mixed ^= uint64_t(token) * multipliers[shift];
            if (shift) for (size_t h = 0; h < 8; ++h) {
                const size_t column = (shift - 1) * 8 + h;
                rows[t * engram_heads + column] = uint32_t(mixed % primes[column]) + offset;
                offset += primes[column];
            }
        }
        previous_ = {current, previous_[0], previous_[1]};
    }
    next_ += int(tokens.size());
}
void engram_row_to_bf16(std::span<const uint8_t> values, std::span<const uint8_t> scales, std::span<uint16_t> output) {
    if (values.size() != engram_head_dim || scales.size() != engram_head_dim / 32 || output.size() != engram_head_dim)
        throw std::invalid_argument("Engram row requires 256 FP8 bytes, 8 scale bytes and 256 BF16 outputs");
    buffer(values.data(), values.size_bytes(), 1); buffer(scales.data(), scales.size_bytes(), 1); buffer(output.data(), output.size_bytes(), 2);
    std::array<uint16_t, engram_head_dim> result{};
    for (size_t i = 0; i < result.size(); ++i) {
        const uint8_t code = values[i], mag = code & 127, scale = scales[i / 32];
        if (mag == 127 || scale == 255) throw std::invalid_argument("Nonfinite Engram FP8 value/scale");
        const int exponent = mag >> 3, fraction = mag & 7;
        const float v = exponent ? std::ldexp(float(8 + fraction), exponent - 10) : std::ldexp(float(fraction), -9);
        const float restored = std::ldexp(code & 128 ? -v : v, int(scale) - 127);
        result[i] = kernels::bf16_from_f32(restored);
        if (!std::isfinite(restored) || !std::isfinite(kernels::f32_from_bf16(result[i])))
            throw std::invalid_argument("Engram row overflows finite BF16");
    }
    std::copy(result.begin(), result.end(), output.begin());
}

struct EngramTable::Impl {
    ngram::PleReader values, scales;
    std::array<uint8_t, engram_heads * engram_head_dim> raw_values{};
    std::array<uint8_t, engram_heads * engram_head_dim / 32> raw_scales{};
    std::array<uint16_t, engram_heads * engram_head_dim> restored{};
    std::vector<uint32_t> populated;
    bool partial = false, failed = false;
};
EngramTable::EngramTable(int layer, const std::string& path, const EngramIoOptions& options)
    : layer_(checked_layer(layer)) {
    const uint32_t rows = table_rows(layer_);
    const std::string prefix = "blk." + std::to_string(layer_) + ".engram_";
    if (!options.max_inflight || options.max_inflight > 1024 || options.cache_rows > (1u << 20))
        throw std::invalid_argument("Engram I/O needs 1..1024 inflight reads and at most 1048576 cached rows");
    auto p = std::make_unique<Impl>();
    struct Table { std::string path; uint64_t offset = 0, bytes = 0; } tables[2];
    {
        const auto model = GgufModel::open(path);
        profile(model.meta(), "strata.deepseek41.engram_table_profile", "1017-layer" + std::to_string(layer_) + "-fp8-row32");
        const auto* complete = model.meta().get("strata.deepseek41.engram_complete");
        if (!complete || complete->type != MetaType::U32 || complete->u > 1)
            throw std::invalid_argument("Engram table must declare complete or guarded partial coverage");
        p->partial = complete->u == 0;
        size_t coverage_shard = 0;
        const auto* coverage = model.find(prefix + "populated_rows", &coverage_shard);
        if (p->partial) {
            if (!options.allow_partial || !coverage || coverage->type != 24 || coverage->shape.size() != 2 ||
                coverage->shape[0] != 4 || !coverage->shape[1] || coverage->shape[1] > rows || !model.in_bounds(*coverage, coverage_shard))
                throw std::invalid_argument("Partial Engram fixture needs explicit opt-in and original-row uint32 payload");
            p->populated.resize(size_t(coverage->shape[1]));
            std::memcpy(p->populated.data(), model.shard(coverage_shard).tensor_data(*coverage), p->populated.size() * 4);
            for (size_t i = 0; i < p->populated.size(); ++i)
                if (p->populated[i] >= rows || (i && p->populated[i] <= p->populated[i - 1]))
                    throw std::invalid_argument("Engram populated rows must be sorted unique original IDs");
        } else if (coverage) throw std::invalid_argument("Complete Engram table cannot declare partial coverage");
        for (size_t i = 0; i < 2; ++i) {
            size_t shard = 0; const auto* t = model.find(prefix + (i ? "embed.scale" : "embed.weight"), &shard);
            const uint64_t width = i ? 8 : 256;
            if (!t || t->type != 24 || t->shape != std::vector<uint64_t>{width, rows} || !model.in_bounds(*t, shard))
                throw std::invalid_argument("Malformed original Engram table shape/type/range");
            tables[i] = {model.shard(shard).path(), model.shard(shard).data_start() + t->offset, rows * width};
        }
        if (tables[0].path == tables[1].path && tables[0].offset < tables[1].offset + tables[1].bytes &&
            tables[1].offset < tables[0].offset + tables[0].bytes)
            throw std::invalid_argument("Engram value and scale tensor ranges overlap");
        if (coverage) {
            const auto& shard = model.shard(coverage_shard);
            const uint64_t begin = shard.data_start() + coverage->offset, end = begin + coverage->shape[1] * 4;
            for (const auto& table : tables)
                if (shard.path() == table.path && begin < table.offset + table.bytes && table.offset < end)
                    throw std::invalid_argument("Engram coverage payload overlaps table data");
        }
    } // Release every GGUF mmap before opening either direct reader.
    std::string error;
    if (!p->values.open(tables[0].path, tables[0].offset, rows, options.max_inflight, options.cache_rows, error, true, 256) ||
        !p->scales.open(tables[1].path, tables[1].offset, rows, options.max_inflight, options.cache_rows, error, true, 8))
        throw std::runtime_error("Engram row reader: " + error);
    impl_ = std::move(p);
}
EngramTable::~EngramTable() = default;
void EngramTable::gather(std::span<const uint32_t> rows, std::span<uint16_t> output) {
    auto& p = *impl_;
    if (p.failed) throw std::logic_error("Engram table must be reconstructed after I/O/data failure");
    if (rows.size() != engram_heads || output.size() != engram_heads * engram_head_dim)
        throw std::invalid_argument("Engram gather requires 24 row IDs and 6144 BF16 outputs");
    buffer(rows.data(), rows.size_bytes(), 4); buffer(output.data(), output.size_bytes(), 2);
    if (overlap(rows.data(), rows.size_bytes(), output.data(), output.size_bytes()))
        throw std::invalid_argument("Engram gather input/output overlap");
    for (uint32_t row : rows)
        if (row >= table_rows(layer_) || (p.partial && !std::binary_search(p.populated.begin(), p.populated.end(), row)))
            throw std::invalid_argument("Engram row is out of range or is an unpopulated fixture hole");
    try {
        const auto v = p.values.issue(rows.data(), rows.size(), p.raw_values.data());
        const auto s = p.scales.issue(rows.data(), rows.size(), p.raw_scales.data());
        std::string error;
        if (!p.values.collect(v, error) || !p.scales.collect(s, error)) throw std::runtime_error("Engram row read: " + error);
        for (size_t head = 0; head < engram_heads; ++head)
            engram_row_to_bf16(std::span(p.raw_values).subspan(head * 256, 256), std::span(p.raw_scales).subspan(head * 8, 8),
                              std::span(p.restored).subspan(head * 256, 256));
        std::copy(p.restored.begin(), p.restored.end(), output.begin());
    } catch (...) { p.failed = true; p.values.close(); p.scales.close(); throw; }
}
std::array<ngram::ReaderStats, 2> EngramTable::stats() const { return {impl_->values.snapshot(), impl_->scales.snapshot()}; }
uint64_t EngramTable::cache_capacity() const { return impl_->values.cache_capacity(); }
} // namespace strata::deepseek41
