#include "strata/deepseek41/engram.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "engram_fixture.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>

using namespace strata;
using namespace strata::deepseek41;
namespace fs = std::filesystem;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& label) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", label.c_str()); }
void rejects(const std::string& label, const std::function<void()>& f) {
    try { f(); check(false, label); } catch (const std::exception&) { check(true, label); }
}
using namespace engram_fixture;
void synthetic_hash(const fs::path& dir, int layer) {
    const auto& primes = layer == 1 ? engram1_primes : engram14_primes;
    const uint32_t row_count = layer == 1 ? engram1_rows : engram14_rows;
    const std::string hash_profile = "1017-layer" + std::to_string(layer) + "-compressed-4gram";
    const auto map = synthetic_map(); EngramHash hash(layer, map);
    const std::vector<int32_t> tokens{0, 11, 1, 19, 129279, 21, 99, 5, 7};
    const std::vector<uint8_t> live{1, 1, 1, 1, 0, 1, 1, 1, 1};
    std::vector<uint32_t> full(tokens.size() * 24), split(full.size()), plain(full.size());
    hash.forward(tokens, full, 0, live);
    check(hash.next_position() == int(tokens.size()), "hash advances one complete contiguous chunk");
    hash.reset();
    for (size_t t = 0; t < tokens.size(); ++t) hash.forward(std::span(tokens).subspan(t, 1), std::span(split).subspan(t * 24, 24), int(t), std::span(live).subspan(t, 1));
    check(split == full, "all-at-once and single-token hash state agree");
    hash.reset(); hash.forward(std::span(tokens).first(3), std::span(split).first(72), 0, std::span(live).first(3));
    hash.forward(std::span(tokens).subspan(3), std::span(split).subspan(72), 3, std::span(live).subspan(3));
    check(split == full, "chunk boundary retains the last three compressed token IDs");
    hash.reset(); hash.forward(tokens, plain, 0); check(plain != full, "dead-token mask changes affected hash rows");
    bool ranges = true; uint32_t base = 0;
    for (size_t h = 0; h < 24; ++h) {
        for (size_t t = 0; t < tokens.size(); ++t) ranges &= full[t * 24 + h] >= base && full[t * 24 + h] < base + primes[h];
        base += primes[h];
    }
    check(ranges && base == row_count, "head ranges are disjoint and span the exact original table");
    auto altered = tokens; altered[0] = 5; altered[1] = 7; altered[3] = 88;
    hash.reset(); hash.forward(altered, split, 0, live);
    check(std::equal(full.begin() + 4 * 24, full.end(), split.begin() + 4 * 24), "dead token blocks every older context and remains a barrier");
    hash.reset(); const std::array<int32_t, 1> zero{0}; const std::array<uint8_t, 1> dead{0};
    std::array<uint32_t, 24> z{}, dz{}; hash.forward(zero, z, 0); hash.reset(); hash.forward(zero, dz, 0, dead);
    check(z != dz, "token zero is text, not missing history");
    hash.reset(); const std::array<int32_t, 4> with_eos{55, 1, 66, 77}; std::array<uint32_t, 96> eos{};
    hash.forward(with_eos, eos, 0); hash.reset(); std::array<uint32_t, 48> tail{}; hash.forward(std::span(with_eos).last(2), tail, 0);
    check(!std::equal(tail.begin(), tail.end(), eos.begin() + 48), "EOS does not reset or truncate Engram history");
    hash.reset(); std::fill(split.begin(), split.end(), 0xdeadbeef); const auto sentinel = split;
    auto invalid = tokens; invalid.back() = 129280;
    rejects("invalid final token is rejected before any output/state change", [&] { hash.forward(invalid, split, 0); });
    invalid.back() = -1; rejects("negative original token", [&] { hash.forward(invalid, split, 0); });
    auto bad_mask = live; bad_mask.back() = 2; rejects("nonboolean live mask", [&] { hash.forward(tokens, split, 0, bad_mask); });
    rejects("mask length", [&] { hash.forward(tokens, split, 0, std::span(live).first(1)); });
    rejects("missing prefix", [&] { hash.forward(tokens, split, 1); });
    rejects("empty chunk", [&] { hash.forward({}, {}, 0); });
    rejects("output dimensions", [&] { hash.forward(tokens, std::span(split).first(24), 0); });
    rejects("null token pointer", [&] { hash.forward({static_cast<const int32_t*>(nullptr), 1}, z, 0); });
    rejects("pointer overflow", [&] { hash.forward({reinterpret_cast<const int32_t*>(UINTPTR_MAX - 3), 1}, z, 0); });
    rejects("token/output alias", [&] { hash.forward({reinterpret_cast<const int32_t*>(split.data()), tokens.size()}, split, 0); });
    rejects("mask/output alias", [&] { hash.forward(tokens, split, 0, {reinterpret_cast<const uint8_t*>(split.data()), tokens.size()}); });
    check(hash.next_position() == 0 && split == sentinel, "all argument rejections preserve output and token history");
    rejects("wrong token-map length", [&] { EngramHash bad{layer, std::span(map).first(3)}; });
    auto bad_map = map; bad_map[2] = 0; rejects("wrong compressed pad", [&] { EngramHash bad(layer, bad_map); });
    bad_map = map; bad_map.back() = 99092; rejects("compressed ID outside vocabulary", [&] { EngramHash bad(layer, bad_map); });
    std::vector<fixture::Kv> meta{
        fixture::str("strata.deepseek41.engram_hash_profile", hash_profile),
        fixture::str("strata.deepseek41.tokenizer_sha256", "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b")};
    const auto path = dir / "map.gguf";
    auto write_map = [&](const std::vector<fixture::Tensor>& tensors) {
        const auto written = fixture::write(path, meta, tensors);
        if (!tensors.empty() && tensors[0].type == 24 && tensors[0].shape == std::vector<uint64_t>{4,engram_token_count}) {
            std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
            file.seekp(std::streamoff(written.data_start)); file.write(reinterpret_cast<const char*>(map.data()), std::streamsize(map.size() * 4));
        }
    };
    const std::vector<fixture::Tensor> map_tensor{{"engram.token_map", {4,engram_token_count},24,0}};
    write_map(map_tensor);
    { GgufFile file(path.string()); check(read_engram_token_map(file, layer) == map, "complete uint32 token-map payload exceeds metadata's 64-entry sample"); }
    rejects("token-map profile from the other Engram layer", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer == 1 ? 14 : 1); });
    rejects("unsupported layer for token-map loading", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, 2); });
    if (layer == 1) {
        GgufFile file(path.string()); check(read_engram_token_map(file) == map, "legacy map loader still selects layer 1");
        Layer1EngramHash legacy(map); std::vector<uint32_t> out(plain.size()); legacy.forward(tokens, out, 0);
        check(out == plain && legacy.layer() == 1, "legacy hash entry point retains layer-1 constants");
    }
    meta[0].s = "qwen"; write_map(map_tensor);
    rejects("wrong hash profile", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer); });
    meta[0].s = hash_profile; meta[1].s = "other"; write_map(map_tensor);
    rejects("wrong tokenizer pin", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer); });
    meta[1].s = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b";
    write_map({}); rejects("missing token-map payload", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer); });
    auto malformed = map_tensor; malformed[0].type = 0; write_map(malformed);
    rejects("wrong token-map payload type", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer); });
    malformed = map_tensor; malformed[0].shape = {4,64}; write_map(malformed);
    rejects("sample-sized token-map payload rejected", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer); });
    write_map(map_tensor); fs::resize_file(path, fs::file_size(path)-1);
    rejects("truncated token-map payload", [&] { GgufFile file(path.string()); (void)read_engram_token_map(file, layer); });
    // Exercise the complete supported position range without prompt-sized state.
    hash.reset(); std::vector<int32_t> chunk(1024, 17); std::vector<uint32_t> rows(1024 * 24);
    for (int pos = 0; pos < 1048576; pos += 1024) hash.forward(chunk, rows, pos);
    rejects("position capacity before access", [&] { hash.forward(zero, z, 1048576); });
    check(hash.next_position() == 1048576, "history remains bounded through the full sequence limit");
}
void synthetic_table(const fs::path& dir, int layer) {
    const uint32_t row_count = layer == 1 ? engram1_rows : engram14_rows;
    auto write_sparse = [layer](const fs::path& path, const std::vector<fixture::Kv>& meta,
                               const std::vector<fixture::Tensor>& tensors, const std::vector<uint32_t>& ids,
                               bool corrupt = false, bool overlap_tensors = false) {
        return sparse(path, meta, tensors, ids, corrupt, overlap_tensors, layer);
    };
    const std::vector<uint32_t> ids{0,1,15,16,17,31,32,4095,4096,4097,65535,65536,65537,1000000,16000056,16000057,32000000,64000000,128000000,256000000,300000000,row_count-3,row_count-2,row_count-1};
    const auto path = dir / "table.gguf"; const auto ts = table_tensors(layer); const auto meta = table_meta(true, ids, layer);
    write_sparse(path, meta, ts, ids);
    rejects("partial table needs explicit opt-in", [&] { EngramTable bad(layer, path.string()); });
    rejects("table from the other original Engram layer", [&] { EngramTable bad(layer == 1 ? 14 : 1, path.string(), {4,0,true}); });
    if (layer == 1) {
        Layer1EngramTable legacy(path.string(), {4,0,true}); std::vector<uint16_t> out(6144);
        legacy.gather(ids, out); check(legacy.layer() == 1 && out == row_oracle(ids), "legacy table entry point still reads layer 1");
    }
    const auto expected = row_oracle(ids); std::vector<uint16_t> output(24 * 256, 0x1234);
    for (uint64_t capacity : {uint64_t(64), uint64_t(8), uint64_t(0)}) {
        EngramTable table(layer, path.string(), {4, capacity, true});
        table.gather(ids, output); check(output == expected, "cold original rows and per-row group scales including >4 GiB offsets");
        const auto cold = table.stats(); table.gather(ids, output); const auto warm = table.stats();
        check(output == expected && table.cache_capacity() == capacity, "warm/eviction/disabled cache preserves BF16 values");
        check(warm[0].requests == 48 && warm[1].requests == 48 && cold[0].reads > 0 && cold[1].reads > 0, "uses existing PleReader request/read accounting");
        if (capacity == 64) check(warm[0].cache_hits == 24 && warm[1].cache_hits == 24 && warm[0].reads == cold[0].reads && warm[1].reads == cold[1].reads, "warm rows and scales need no SSD reads");
        else check(warm[0].reads > cold[0].reads && warm[1].reads > cold[1].reads, "small or disabled cache forces original row reloads");
        auto bad = ids; bad.back() = row_count;
        const auto saved = output;
        rejects("out-of-range original row", [&] { table.gather(bad, output); });
        bad.back() = 7; rejects("unpopulated fixture hole", [&] { table.gather(bad, output); });
        rejects("gather shape mismatch", [&] { table.gather(std::span(ids).first(1), output); });
        rejects("gather null output", [&] { table.gather(ids, {static_cast<uint16_t*>(nullptr), 6144}); });
        rejects("gather input/output alias", [&] { table.gather({reinterpret_cast<uint32_t*>(output.data()), 24}, output); });
        const auto after = table.stats();
        check(output == saved && after[0].requests == warm[0].requests && after[1].requests == warm[1].requests, "gather validation errors preserve output and issue no I/O");
    }
    const auto invalid = dir / "invalid.gguf";
    auto metadata = meta; metadata[0].s = "other"; write_sparse(invalid, metadata, ts, {});
    rejects("wrong table profile", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    metadata = meta; metadata[1].u = 2; write_sparse(invalid, metadata, ts, {});
    rejects("invalid completeness marker", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    metadata = meta; metadata[2].arr[1] = metadata[2].arr[0]; write_sparse(invalid, metadata, ts, {});
    rejects("duplicate partial row IDs", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    metadata = meta; metadata[1].u = 1; write_sparse(invalid, metadata, ts, {});
    rejects("complete and partial declarations conflict", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    auto wrong_names = ts;
    const auto other = table_tensors(layer == 1 ? 14 : 1);
    for (size_t i = 0; i < wrong_names.size(); ++i) wrong_names[i].name = other[i].name;
    write_sparse(invalid, meta, wrong_names, {});
    rejects("correct profile with wrong original tensor names", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    auto wrong_rows = ts;
    for (auto& t : wrong_rows) t.shape[1] = layer == 1 ? engram14_rows : engram1_rows;
    write_sparse(invalid, meta, wrong_rows, {});
    rejects("correct profile with the other layer's row count", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    metadata = meta; metadata[2].arr.back() = row_count; write_sparse(invalid, metadata, ts, {});
    rejects("coverage row beyond original table", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    auto tensors = ts; tensors[1].shape[0] = 256; write_sparse(invalid, meta, tensors, {});
    rejects("matrix-block scales cannot replace row-local scales", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    tensors = ts; tensors[0].type = 30; write_sparse(invalid, meta, tensors, {});
    rejects("wrong table dtype", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    tensors = ts; tensors.pop_back(); write_sparse(invalid, meta, tensors, {});
    rejects("missing scale table", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    write_sparse(invalid, meta, ts, {}, false, true);
    rejects("overlapping value/scale tensors", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    const auto header = write_sparse(invalid, meta, ts, {});
    const uint64_t payload_end = header.data_start + header.offsets[1] + uint64_t(row_count) * 8;
    if (header.size > payload_end) {
        fs::resize_file(invalid, payload_end);
        EngramTable valid(layer, invalid.string(), {4,0,true});
        check(valid.layer() == layer, "omitted trailing alignment padding leaves complete tensors readable");
    }
    fs::resize_file(invalid, payload_end - 1);
    rejects("truncated scale table", [&] { EngramTable bad(layer, invalid.string(), {4,0,true}); });
    rejects("zero I/O capacity", [&] { EngramTable bad(layer, path.string(), {0,0,true}); });
    rejects("unbounded cache request", [&] { EngramTable bad(layer, path.string(), {4,1u<<21,true}); });
    write_sparse(invalid, meta, ts, ids, true);
    {
        EngramTable bad(layer, invalid.string(), {4,0,true}); output.assign(output.size(), 0x1234); const auto saved = output;
        rejects("nonfinite source row", [&] { bad.gather(ids, output); });
        check(output == saved, "corrupt row never commits partial output");
        rejects("data failure poisons lookup", [&] { bad.gather(ids, output); });
    }
    write_sparse(invalid, meta, ts, ids);
    {
        EngramTable bad(layer, invalid.string(), {4,0,true}); fs::resize_file(invalid, 4096);
        output.assign(output.size(), 0x1234); const auto saved = output;
        rejects("short direct read after open", [&] { bad.gather(ids, output); });
        check(output == saved, "I/O failure leaves output untouched");
        rejects("I/O failure poisons lookup", [&] { bad.gather(ids, output); });
    }
    const auto p1 = dir / "split-00001-of-00002.gguf", p2 = dir / "split-00002-of-00002.gguf";
    auto m1 = meta, m2 = fixture::split_keys(1,2,3), first_keys = fixture::split_keys(0,2,3);
    m1.push_back(fixture::str("general.architecture", "deepseek41-engram-table-fixture")); m1.insert(m1.end(), first_keys.begin(), first_keys.end());
    write_sparse(p1, m1, {ts[0]}, ids); write_sparse(p2, m2, {ts[1]}, ids);
    { EngramTable split(layer, p1.string(), {4,0,true}); split.gather(ids, output); check(output == expected, "split weight/scale shards use original rows and release mappings before I/O"); }
    // Finite binary boundary cases, independent power-of-two expected values.
    std::array<uint8_t,256> values{}; std::array<uint8_t,8> scales{}; std::array<uint16_t,256> out{};
    values.fill(0x38); scales.fill(127); engram_row_to_bf16(values, scales, out);
    check(std::all_of(out.begin(), out.end(), [](uint16_t x) { return x == 0x3f80; }), "FP8 1.0 row restores BF16 1.0");
    values.fill(0x80); scales.fill(0); engram_row_to_bf16(values, scales, out);
    check(std::all_of(out.begin(), out.end(), [](uint16_t x) { return x == 0x8000; }), "signed zero retained at minimum E8M0 scale");
    values.fill(1); scales.fill(0); engram_row_to_bf16(values, scales, out);
    check(std::all_of(out.begin(), out.end(), [](uint16_t x) { return x == 0; }), "tiny finite values follow BF16 underflow rounding");
    values.fill(0x38); scales.fill(255); const auto saved = out;
    rejects("NaN E8M0 scale", [&] { engram_row_to_bf16(values, scales, out); });
    check(out == saved, "failed scalar conversion does not modify output");
    scales.fill(254); values.fill(0x7e); rejects("nonfinite restored BF16", [&] { engram_row_to_bf16(values, scales, out); });
}
void independent_layers(const fs::path& dir) {
    const auto map = synthetic_map();
    for (int layer : {-1,0,2,13,15,40}) {
        rejects("unsupported original hash layer " + std::to_string(layer), [&] { EngramHash bad(layer, map); });
        rejects("unsupported original table layer " + std::to_string(layer), [&] { EngramTable bad(layer, "unused"); });
    }
    const std::array<int32_t, 7> tokens{0,455,1805,1,20852,240,129279};
    const std::array<uint8_t, 7> live{1,1,0,1,1,1,1};
    EngramHash h1(1,map), h14(14,map), oracle1(1,map), oracle14(14,map);
    std::array<uint32_t, 7*24> one{}, fourteen{}, expected1{}, expected14{};
    oracle1.forward(tokens,expected1,0,live); oracle14.forward(tokens,expected14,0,live);
    h1.forward(std::span(tokens).first(3),std::span(one).first(72),0,std::span(live).first(3));
    h14.forward(std::span(tokens).first(5),std::span(fourteen).first(120),0,std::span(live).first(5));
    h1.forward(std::span(tokens).subspan(3),std::span(one).subspan(72),3,std::span(live).subspan(3));
    h1.reset();
    check(h1.next_position() == 0 && h14.next_position() == 5, "resetting layer 1 preserves layer-14 history and position");
    h14.forward(std::span(tokens).subspan(5),std::span(fourteen).subspan(120),5,std::span(live).subspan(5));
    check(one == expected1 && fourteen == expected14 && one != fourteen,
          "interleaved layers hash the same original tokens with independent state and constants");
    h1.forward(tokens,one,0,live); h14.reset();
    check(h1.next_position() == 7 && one == expected1 && h14.next_position() == 0,
          "resetting layer 14 preserves layer-1 replay");
    const auto p1 = dir/"independent1.gguf", p14 = dir/"independent14.gguf";
    sparse(p1,table_meta(true,{0}),table_tensors(),{0});
    const auto written = sparse(p14,table_meta(true,{0},14),table_tensors(14),{0},false,false,14);
    { // Distinct data at the same original row number exposes accidental cache sharing.
        std::fstream f(p14,std::ios::binary|std::ios::in|std::ios::out);
        const std::array<uint8_t,256> values = [] { std::array<uint8_t,256> v{}; v.fill(0x40); return v; }();
        f.seekp(std::streamoff(written.data_start+written.offsets[0]));
        f.write(reinterpret_cast<const char*>(values.data()),values.size());
    }
    EngramTable t1(1,p1.string(),{4,64,true}), t14(14,p14.string(),{4,64,true});
    std::array<uint32_t,24> ids{}; std::vector<uint16_t> v1(6144),v14(6144);
    auto expected_values = row_oracle(ids);
    t1.gather(ids,v1);
    check(t14.stats()[0].requests == 0 && t14.stats()[1].requests == 0, "layer-1 I/O leaves layer-14 readers untouched");
    for (auto& x : expected_values) x = kernels::bf16_from_f32(2 * kernels::f32_from_bf16(x));
    t14.gather(ids,v14);
    check(v1 == row_oracle(ids) && v14 == expected_values && v1 != v14, "equal original row IDs retain distinct per-layer table data");
    const auto before = t14.stats(); t1.gather(ids,v1); const auto after = t14.stats();
    check(v1 == row_oracle(ids) && before[0].requests == after[0].requests && before[1].requests == after[1].requests,
          "interleaved warm reads preserve other layer's cache and statistics");
}
template<class T> std::vector<T> tensor(const GgufFile& f, const std::string& name, uint32_t type, size_t count) {
    const auto* t = f.find(name);
    if (!t || t->type != type || tensor_payload_bytes(*t) != count * sizeof(T) ||
        t->offset > f.file_size() - f.data_start() || count * sizeof(T) > f.file_size() - f.data_start() - t->offset)
        throw std::runtime_error("Malformed fixture tensor " + name);
    std::vector<T> out(count); std::memcpy(out.data(), f.tensor_data(*t), count * sizeof(T)); return out;
}
void real(const char* fixture_path, const char* table_path, int layer) {
    const GgufFile f(fixture_path); const auto map = read_engram_token_map(f, layer);
    check(map[455] == map[1805] && map[1805] == map[20852], "pinned tokenizer normalizes The/the/THE to identical compressed IDs");
    const auto tokens = tensor<int32_t>(f, "tokens", 24, 9);
    const auto live = tensor<uint8_t>(f, "live", 24, 9);
    const auto masked = tensor<uint32_t>(f, "hash.masked", 24, 9 * 24);
    const auto plain = tensor<uint32_t>(f, "hash.plain", 24, 9 * 24);
    const auto expected = tensor<uint16_t>(f, "expected.lookup", 30, 9 * 24 * 256);
    for (bool mask : {false, true}) for (int first : {1,3,9}) {
        EngramHash hash(layer, map); std::vector<uint32_t> out(9 * 24);
        for (int pass = 0; pass < 2; ++pass) {
            hash.reset();
            for (size_t pos = 0; pos < tokens.size();) {
                const size_t n = pos ? 1 : size_t(first);
                hash.forward(std::span(tokens).subspan(pos,n), std::span(out).subspan(pos*24,n*24), int(pos), mask ? std::span(live).subspan(pos,n) : std::span<const uint8_t>{});
                pos += n;
            }
            check(out == (mask ? masked : plain) && hash.next_position() == 9,
                  "original hash IDs vs pinned NgramHashState mask=" + std::to_string(mask) + " prefill=" + std::to_string(first) + " pass=" + std::to_string(pass));
        }
    }
    const auto rt = tensor<int32_t>(f, "random.tokens", 24, 257);
    const auto rm = tensor<uint8_t>(f, "random.live", 24, 257);
    const auto rh = tensor<uint32_t>(f, "random.hash", 24, 257 * 24);
    for (size_t chunk : {size_t(1),size_t(7),size_t(31),size_t(257)}) {
        EngramHash hash(layer, map); std::vector<uint32_t> out(rh.size());
        for (size_t pos = 0; pos < rt.size();) {
            const size_t n = std::min(chunk, rt.size() - pos);
            hash.forward(std::span(rt).subspan(pos,n), std::span(out).subspan(pos*24,n*24), int(pos), std::span(rm).subspan(pos,n)); pos += n;
        }
        check(out == rh, "257-token original oracle with dead barriers and chunk size=" + std::to_string(chunk));
    }
    const auto* populated_info = f.find("populated");
    if (!populated_info || tensor_payload_bytes(*populated_info) % 4) throw std::runtime_error("Malformed populated row list");
    const size_t count = size_t(tensor_payload_bytes(*populated_info) / 4);
    const auto ids = tensor<uint32_t>(f, "populated", 24, count);
    const auto expected_rows = tensor<uint16_t>(f, "expected.rows", 30, count * 256);
    rejects("real token-map layer identity mismatch", [&] { (void)read_engram_token_map(f, layer == 1 ? 14 : 1); });
    rejects("real table layer identity mismatch", [&] { EngramTable invalid(layer == 1 ? 14 : 1, table_path, {8,0,true}); });
    rejects("real sparse table is never treated as a complete model", [&] { EngramTable invalid(layer, table_path); });
    for (uint64_t capacity : {uint64_t(4096),uint64_t(512),uint64_t(8),uint64_t(0)}) {
        EngramHash hash(layer, map); EngramTable table(layer, table_path, {8,capacity,true});
        for (int pass = 0; pass < 2; ++pass) {
            hash.reset(); std::vector<uint32_t> out(masked.size()); std::vector<uint16_t> actual(expected.size());
            const auto before = table.stats();
            for (size_t pos = 0; pos < tokens.size();) {
                const size_t n = pos ? 1 : 3;
                hash.forward(std::span(tokens).subspan(pos,n), std::span(out).subspan(pos*24,n*24), int(pos), std::span(live).subspan(pos,n));
                for (size_t t = pos; t < pos+n; ++t) table.gather(std::span(out).subspan(t*24,24), std::span(actual).subspan(t*6144,6144));
                pos += n;
            }
            check(out == masked && actual == expected, "native token history + original-row lookup exactly matches pinned BF16 reference cache=" + std::to_string(capacity) + " pass=" + std::to_string(pass));
            const auto after = table.stats();
            check(after[0].requests - before[0].requests == 216 && after[1].requests - before[1].requests == 216, "one token requests exactly 24 original value rows and scale rows");
            if (pass && capacity == 4096) check(after[0].reads == before[0].reads && after[1].reads == before[1].reads &&
                                             after[0].cache_hits - before[0].cache_hits == 216 && after[1].cache_hits - before[1].cache_hits == 216,
                                             "warm real sequence reuses bounded Strata row caches");
            // PleReader is 8-way set associative. These 216 rows put 12 and 9
            // entries in two sets at capacity 512; total capacity alone cannot
            // guarantee a fully warm pass. At capacity 4096 the maximum is 3.
            if (pass && capacity == 512 && layer == 1) check(after[0].reads > before[0].reads && after[1].reads > before[1].reads &&
                                              after[0].cache_hits > before[0].cache_hits && after[1].cache_hits > before[1].cache_hits,
                                              "real set conflicts retain partial reuse and correct outputs");
            if (pass && capacity == 512 && layer == 14)
                check(after[0].cache_hits > before[0].cache_hits && after[1].cache_hits > before[1].cache_hits,
                      "layer-14 rows reuse their own bounded caches");
            if (pass && capacity <= 8) check(after[0].reads > before[0].reads && after[1].reads > before[1].reads,
                                            "real eviction/disabled cache rereads original rows without output changes");
            std::printf("real_engram_pass cache=%llu pass=%d value_reads=%llu scale_reads=%llu value_hits=%llu scale_hits=%llu bytes=%llu\n",
                        (unsigned long long)capacity, pass,
                        (unsigned long long)(after[0].reads-before[0].reads), (unsigned long long)(after[1].reads-before[1].reads),
                        (unsigned long long)(after[0].cache_hits-before[0].cache_hits), (unsigned long long)(after[1].cache_hits-before[1].cache_hits),
                        (unsigned long long)(after[0].bytes+after[1].bytes-before[0].bytes-before[1].bytes));
        }
        std::array<uint32_t,24> batch{}; std::array<uint16_t,6144> values{}; bool exact = true;
        for (size_t start = 0; start < count; start += 24) {
            batch.fill(ids.front()); const size_t n = std::min(size_t(24), count-start);
            std::copy_n(ids.begin()+ptrdiff_t(start), n, batch.begin()); table.gather(batch, values);
            exact &= std::equal(values.begin(), values.begin()+ptrdiff_t(n*256), expected_rows.begin()+ptrdiff_t(start*256));
        }
        check(exact, "every downloaded real row including table boundaries exactly matches official embedding conversion");
        const auto stats = table.stats();
        std::printf("real_engram_cache=%llu populated_rows=%zu value_reads=%llu scale_reads=%llu value_hits=%llu scale_hits=%llu bytes=%llu\n",
                    (unsigned long long)capacity, count, (unsigned long long)stats[0].reads, (unsigned long long)stats[1].reads,
                    (unsigned long long)stats[0].cache_hits, (unsigned long long)stats[1].cache_hits,
                    (unsigned long long)(stats[0].bytes + stats[1].bytes));
    }
}
} // namespace
int main(int argc, char** argv) {
    int layer = 0, arg = 1;
    if (argc > 1 && std::string(argv[1]) == "--layer") {
        if (argc < 3 || (std::string(argv[2]) != "1" && std::string(argv[2]) != "14")) return 2;
        layer = std::string(argv[2]) == "1" ? 1 : 14; arg = 3;
    }
    if (!(argc == arg || (argc == arg+3 && std::string(argv[arg]) == "--real"))) {
        std::fprintf(stderr, "usage: deepseek41_engram_test [--layer 1|14] [--real REFERENCE.gguf TABLE.gguf]\n"); return 2;
    }
    const auto dir = fs::temp_directory_path() / ("strata-engram-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    try {
        if (argc == arg) {
            for (int original : {1,14}) if (!layer || layer == original) {
                std::printf("synthetic_engram_layer=%d\n", original);
                synthetic_hash(dir,original); synthetic_table(dir,original);
            }
            if (!layer) independent_layers(dir);
        } else real(argv[arg+1],argv[arg+2],layer ? layer : 1);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    fs::remove_all(dir);
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
