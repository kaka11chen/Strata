#pragma once

// Reuse the original-size, guarded sparse Engram fixtures across lookup/decoder tests.
#include "strata/deepseek41/engram.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "../core/gguf_fixture.hpp"
#include <algorithm>
#include <cmath>

namespace engram_fixture {
using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
inline std::vector<uint32_t> synthetic_map() {
    std::vector<uint32_t> m(engram_token_count);
    for (size_t i = 0; i < m.size(); ++i) m[i] = uint32_t(i * 17 % engram_compressed_tokens);
    m[2] = 2; return m;
}
inline std::vector<fixture::Kv> table_meta(bool partial, const std::vector<uint32_t>& ids, int layer = 1) {
    std::vector<fixture::Kv> m{fixture::str("strata.deepseek41.engram_table_profile", "1017-layer" + std::to_string(layer) + "-fp8-row32"),
                              fixture::u32("strata.deepseek41.engram_complete", partial ? 0 : 1)};
    if (partial) m.push_back(fixture::u32s("strata.deepseek41.engram_populated_rows", {ids.begin(), ids.end()}));
    return m;
}
inline std::vector<fixture::Tensor> table_tensors(int layer = 1) {
    const uint32_t rows = layer == 1 ? engram1_rows : engram14_rows;
    const std::string prefix = "blk." + std::to_string(layer) + ".engram_embed.";
    return {{prefix + "weight", {256, rows}, 24, 1}, {prefix + "scale", {8, rows}, 24, 2}};
}
// Sparse original-size tensor fixture: writes only a small header and selected
// rows. Never allocates, fills or scans the 101 GB logical payload.
inline fixture::Written sparse(const fs::path& path, const std::vector<fixture::Kv>& meta,
                        const std::vector<fixture::Tensor>& tensors, const std::vector<uint32_t>& ids,
                        bool corrupt = false, bool overlap_tensors = false, int layer = 1) {
    std::vector<uint8_t> bytes;
    auto put = [&](const void* p, size_t n) { const auto* v = static_cast<const uint8_t*>(p); bytes.insert(bytes.end(), v, v + n); };
    auto u32 = [&](uint32_t x) { put(&x, 4); }; auto u64 = [&](uint64_t x) { put(&x, 8); };
    auto str = [&](const std::string& x) { u64(x.size()); put(x.data(), x.size()); };
    const auto coverage = std::find_if(meta.begin(), meta.end(), [](const auto& m) { return m.key == "strata.deepseek41.engram_populated_rows"; });
    const bool partial = coverage != meta.end();
    u32(0x46554747); u32(3); u64(tensors.size() + size_t(partial)); u64(meta.size() - size_t(partial));
    for (const auto& m : meta) {
        if (m.key == "strata.deepseek41.engram_populated_rows") continue;
        str(m.key); u32(m.type);
        if (m.type == 8) str(m.s);
        else if (m.type == 2) { const uint16_t n = uint16_t(m.u); put(&n, 2); }
        else if (m.type == 4 || m.type == 5) u32(uint32_t(m.u));
        else if (m.type == 9 && m.elem == 4) { u32(4); u64(m.arr.size()); for (auto x : m.arr) u32(uint32_t(x)); }
        else throw std::runtime_error("Unsupported sparse fixture metadata");
    }
    fixture::Written result; uint64_t end = 0;
    if (partial) {
        str("blk." + std::to_string(layer) + ".engram_populated_rows"); u32(2); u64(4); u64(coverage->arr.size()); u32(24); u64(0);
        end = (coverage->arr.size() * 4 + 31) / 32 * 32;
    }
    for (const auto& t : tensors) {
        const uint64_t offset = overlap_tensors ? 0 : end;
        str(t.name); u32(uint32_t(t.shape.size())); for (auto d : t.shape) u64(d);
        u32(t.type); u64(offset); result.offsets.push_back(offset);
        TensorInfo info; info.shape = t.shape; info.type = t.type;
        end = std::max(end, (offset + tensor_payload_bytes(info) + 31) / 32 * 32);
    }
    bytes.resize((bytes.size() + 31) / 32 * 32); result.data_start = bytes.size(); result.size = result.data_start + end;
    std::ofstream f(path, std::ios::binary | std::ios::trunc); f.exceptions(std::ios::badbit | std::ios::failbit);
    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    f.seekp(std::streamoff(result.size - 1)); f.put(0);
    if (partial) {
        f.seekp(std::streamoff(result.data_start));
        for (uint64_t v : coverage->arr) { const uint32_t value = uint32_t(v); f.write(reinterpret_cast<const char*>(&value), 4); }
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].type != 24 || tensors[i].shape.size() != 2 || tensors[i].shape[1] != (layer == 1 ? engram1_rows : engram14_rows)) continue;
        const size_t width = size_t(tensors[i].shape[0]);
        if (width != 256 && width != 8) continue;
        for (uint32_t row : ids) {
            std::vector<uint8_t> raw(width);
            for (size_t j = 0; j < width; ++j)
                raw[j] = width == 256 ? uint8_t(row % 2 ? 0xb8 : 0x38) : uint8_t(123 + (row + j) % 9);
            if (corrupt && width == 256) raw.back() = 0x7f;
            f.seekp(std::streamoff(result.data_start + result.offsets[i] + uint64_t(row) * width));
            f.write(reinterpret_cast<const char*>(raw.data()), std::streamsize(raw.size()));
        }
    }
    return result;
}
inline std::vector<uint16_t> row_oracle(std::span<const uint32_t> rows) {
    std::vector<uint16_t> v(rows.size() * 256);
    for (size_t i = 0; i < rows.size(); ++i) for (size_t d = 0; d < 256; ++d)
        v[i * 256 + d] = kernels::bf16_from_f32(std::ldexp(rows[i] % 2 ? -1.f : 1.f, int((rows[i] + d / 32) % 9) - 4));
    return v;
}
} // namespace engram_fixture
