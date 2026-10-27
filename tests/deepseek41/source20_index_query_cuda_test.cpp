#include "strata/deepseek41/source20_index_key_cuda.hpp"
#include "strata/deepseek41/source20_compressor_cuda.hpp"
#include "strata/deepseek41/source20_index_query_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <climits>
#include <numeric>
#include <type_traits>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int dim = 128, hidden = 5120, qr_dim = 1280, query_size = 4096;
int checks = 0, failures = 0;
size_t bf16_values = 0, bf16_differences = 0, score_values = 0, score_differences = 0;
void check(bool ok, const std::string& name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void rejects(const std::string& name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
template<class T> void upload(T* dst, const std::vector<T>& src, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
}
template<class T> std::vector<T> download(const T* src, size_t n, cudaStream_t stream) {
    std::vector<T> result(n);
    ck(cudaMemcpyAsync(result.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return result;
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
void exact(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::string& name) {
    if (a.size() != b.size()) throw std::runtime_error("Reference shape mismatch");
    size_t different = 0; double maximum = 0.;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i] && different < 8)
            std::printf("    mismatch index=%zu actual=%04x expected=%04x\n", i, unsigned(a[i]), unsigned(b[i]));
        different += a[i] != b[i];
        maximum = std::max(maximum, std::abs(double(kernels::f32_from_bf16(a[i])) - kernels::f32_from_bf16(b[i])));
    }
    bf16_values += a.size(); bf16_differences += different;
    std::printf("  %s different=%zu/%zu max_abs=%.9g\n", name.c_str(), different, a.size(), maximum);
    check(different == 0, name);
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, size_t count) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != (sizeof(T) == 2 ? 30u : 0u) || tensor_payload_bytes(*t) != count * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Malformed index query fixture tensor " + name);
    const auto* data = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*t)); return {data, data + count};
}
int number(const GgufModel& model, const std::string& name, int maximum) {
    const auto* m = model.meta().get(name);
    if (!m || m->type != MetaType::U32 || m->u > uint64_t(maximum)) throw std::runtime_error("Malformed fixture number " + name);
    return int(m->u);
}
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
std::vector<int32_t> select(const std::vector<float>& score, int visible, int offset) {
    std::vector<int32_t> ids(score.size()); std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return score[a] > score[b]; });
    ids.resize(std::min(size_t(512), ids.size())); std::sort(ids.begin(), ids.end());
    for (auto& id : ids) id = id < visible ? id + offset : -1;
    return ids;
}
void scores_exact(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name) {
    if (actual.size() != expected.size()) throw std::runtime_error("Score shape mismatch");
    size_t different = 0; double maximum = 0.;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            if (different < 8) std::printf("    score mismatch i=%zu actual=%.9g expected=%.9g\n", i, actual[i], expected[i]);
            ++different;
            maximum = std::max(maximum, std::abs(double(actual[i]) - expected[i]));
        }
    }
    score_values += actual.size(); score_differences += different;
    std::printf("  %s different=%zu/%zu max_abs=%.9g\n", name.c_str(), different, actual.size(), maximum);
    check(different == 0, name);
}
static_assert(!std::is_constructible_v<CudaIndexQuery, Source20IndexQueryWeights, int>);
static_assert(!std::is_constructible_v<CudaSource20IndexQuery, IndexQueryWeights, int>);
std::vector<float> block_scores(const std::vector<float>& scores, int visible) {
    std::vector<float> blocks((scores.size() + 7) / 8, -INFINITY);
    for (int i = 0; i < visible; ++i) blocks[size_t(i) / 8] = std::max(blocks[size_t(i) / 8], scores[size_t(i)]);
    blocks[size_t(visible - 1) / 8] = INFINITY;
    return blocks;
}
std::vector<int32_t> candidate_reference(const std::vector<float>& scores, int visible) {
    const auto blocks = block_scores(scores, visible);
    std::vector<int32_t> ids(blocks.size()); std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return blocks[size_t(a)] > blocks[size_t(b)]; });
    ids.resize(std::min(size_t(2048), ids.size()));
    for (auto& id : ids) if (blocks[size_t(id)] == -INFINITY) id = INT_MAX;
    std::sort(ids.begin(), ids.end());
    for (auto& id : ids) if (id == INT_MAX) id = -1;
    return ids;
}
std::vector<float> selected_scores(const std::vector<float>& scores, const std::vector<int32_t>& ids, int offset = 0) {
    std::vector<float> values;
    for (auto id : ids) if (id >= 0) values.push_back(scores.at(size_t(id - offset)));
    std::sort(values.begin(), values.end()); return values;
}
void validate_candidates(const std::vector<float>& scores, const Source20Batch& batch, int row,
                         const std::vector<int32_t>& ids, const std::string& name) {
    const auto expected = candidate_reference(scores, batch.visible_keys(row));
    check(ids == expected, name + " canonical block max/top-2048/newest pin");
    auto masked = scores;
    std::fill(masked.begin() + batch.visible_keys(row), masked.end(), -INFINITY);
    validate_source20_candidates(batch, row, masked, ids);
    check(true, name + " validated by confirmed source-20 contract");
}
Source20Batch prefill(int tokens) { return Source20Batch(0, tokens, 0, tokens, tokens); }
Source20Batch decode(int count) { return Source20Batch(count - 1, 1, count - 1, count, count); }
void candidate_synthetic() {
    Stream stream, other;
    constexpr int cap = Source20Batch::max_context, guard = 64;
    for (int capacity : {0, -1, cap + 1}) rejects("invalid candidate capacity", [&] { CudaSource20Candidates bad(capacity); });
    CudaSource20Candidates selector(cap);
    core::DeviceArena arena(size_t(cap + 2 * guard) * 4);
    auto* storage = static_cast<float*>(arena.alloc(size_t(cap + 2 * guard) * 4));
    auto* input = storage + guard;
    upload(storage, std::vector<float>(cap + 2 * guard, -777.f), stream.value);
    const auto bytes = selector.device_bytes(), workspace = selector.sort_workspace_bytes();
    check(selector.capacity() == cap && bytes > workspace && workspace > 0, "bounded candidate allocation at maximum context");
    const std::vector<std::pair<int,int>> sizes{{1,1},{7,1},{8,7},{8,8},{9,8},{9,9},{17,1},{17,7},{17,8},{17,9},{17,17},
        {16384,16384},{16385,16385},{16393,16385},{16393,16393},{16393,9},{cap,cap}};
    for (const auto& [width, visible] : sizes) {
        for (int mode = 0; mode < 3; ++mode) {
            std::vector<float> scores(size_t(width), -INFINITY);
            for (int i = 0; i < visible; ++i) {
                if (mode == 0) scores[size_t(i)] = float(width - i);
                if (mode == 1) scores[size_t(i)] = i % 2 ? -0.f : 0.f;
                if (mode == 2 && (i / 8) % 3 != 1 && i / 8 != (visible - 1) / 8) scores[size_t(i)] = float(i % 37 - 18);
            }
            // Even an accidentally unmasked future score cannot select a future block.
            if (mode == 2) std::fill(scores.begin() + visible, scores.end(), 1e20f);
            upload(input, scores, stream.value);
            const auto batch = prefill(width);
            const auto result = selector.forward(input, batch, visible - 1, stream.value);
            const auto ids = download(result.blocks, result.count, stream.value);
            check(result.count == batch.candidate_stride(), "candidate stride includes unreachable padding");
            validate_candidates(scores, batch, visible - 1, ids, "candidate width " + std::to_string(width));
            check(same(scores, download(input, scores.size(), stream.value)), "candidate scores remain unchanged");
        }
    }
    const auto batch = prefill(cap);
    const auto saved = selector.forward(input, batch, cap - 1, stream.value);
    const auto before = download(saved.blocks, saved.count, stream.value);
    rejects("null candidate input", [&] { selector.forward(nullptr, batch, 0, stream.value); });
    rejects("unaligned candidate input", [&] { selector.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(input) + 1), batch, 0, stream.value); });
    rejects("candidate input overflow", [&] { selector.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), batch, 0, stream.value); });
    rejects("candidate output cannot be input", [&] { selector.forward(reinterpret_cast<const float*>(saved.blocks), batch, 0, stream.value); });
    rejects("negative query row", [&] { selector.forward(input, batch, -1, stream.value); });
    rejects("query row exceeds batch", [&] { selector.forward(input, batch, cap, stream.value); });
    rejects("null candidate stream", [&] { selector.forward(input, batch, 0, nullptr); });
    rejects("candidate stream switch", [&] { selector.forward(input, batch, 0, other.value); });
    check(same(before, download(saved.blocks, saved.count, stream.value)), "argument failures retain last candidate result");
    check(same(download(storage, guard, stream.value), std::vector<float>(guard, -777.f)) &&
          same(download(input + cap, guard, stream.value), std::vector<float>(guard, -777.f)), "candidate input guards preserved");
    check(selector.device_bytes() == bytes && selector.sort_workspace_bytes() == workspace, "candidate calls retain startup allocation");
    for (int capacity : {1,7,8,9,17,2048,16385}) {
        CudaSource20Candidates small(capacity);
        rejects("candidate capacity rejection before stream binding", [&] { small.forward(input, prefill(capacity + 1), 0, other.value); });
        const auto result = small.forward(input, prefill(capacity), 0, stream.value);
        check(result.count == std::min(2048, (capacity + 7) / 8) && download(result.blocks, 1, stream.value)[0] == 0,
              "candidate exact capacity and first visible block");
    }
    std::printf("candidate_capacity=%d candidate_device_bytes=%zu candidate_workspace_bytes=%zu\n", selector.capacity(), bytes, workspace);
}
void query_synthetic() {
    Stream stream, other;
    constexpr int capacity = 640;
    std::vector<uint8_t> codes(query_size * qr_dim, 0), scales(128 * 40, 127);
    std::vector<uint16_t> weights(32 * hidden, 0);
    for (int head = 0; head < 32; ++head) codes[size_t(head) * dim * qr_dim] = 0x38;
    weights[0] = kernels::bf16_from_f32(1.f);
    Source20IndexQueryWeights w{{codes, scales, query_size, qr_dim}, weights};
    for (int cap : {0, -1, 1048577}) rejects("invalid query capacity", [&] { CudaSource20IndexQuery bad(w, cap); });
    rejects("wrong query geometry", [&] { auto bad = w; bad.query.rows = 128; CudaSource20IndexQuery object(bad, 1); });
    rejects("truncated FP8 payload", [&] { auto bad = w; bad.query.values = {}; CudaSource20IndexQuery object(bad, 1); });
    rejects("truncated scales", [&] { auto bad = w; bad.query.scales = {}; CudaSource20IndexQuery object(bad, 1); });
    rejects("missing head weights", [&] { auto bad = w; bad.head_weights = {}; CudaSource20IndexQuery object(bad, 1); });
    weights[0] = 0x7fc0;
    rejects("nonfinite BF16 weights", [&] { CudaSource20IndexQuery bad(w, 1); }); weights[0] = 0x3f80;
    codes[0] = 0x7f;
    rejects("nonfinite FP8 weights", [&] { CudaSource20IndexQuery bad(w, 1); }); codes[0] = 0x38;
    CudaSource20IndexQuery index(w, capacity);
    core::DeviceArena arena((hidden + qr_dim) * 4 + capacity * dim * 2 + 512);
    auto* x = static_cast<float*>(arena.alloc(hidden * 4));
    auto* qr = static_cast<float*>(arena.alloc(qr_dim * 4));
    auto* keys = static_cast<uint16_t*>(arena.alloc(capacity * dim * 2));
    std::vector<float> input(hidden, 0.f), lowrank(qr_dim, 0.f); lowrank[0] = 1.f;
    std::vector<uint16_t> key_values(capacity * dim, 0);
    for (int i = 0; i < capacity; ++i) key_values[size_t(i) * dim] = kernels::bf16_from_f32(float((i * 73) % 641 - 320));
    upload(qr, lowrank, stream.value); upload(keys, key_values, stream.value);
    const auto bytes = index.device_bytes(), workspace = index.sort_workspace_bytes();
    check(index.capacity() == capacity && bytes > workspace && workspace > 0, "bounded query and candidate allocations");
    for (float sign : {1.f,-1.f,0.f}) {
        input[0] = sign * 64.f; upload(x, input, stream.value);
        for (int position : {0,6,7,8,510,511,512,639}) {
            const auto batch = prefill(capacity);
            const auto result = index.forward(x, qr, keys, batch, position, 128, stream.value);
            std::vector<float> expected(capacity, -INFINITY);
            for (int i = 0; i <= position; ++i) expected[size_t(i)] = bf(std::max(0.f, kernels::f32_from_bf16(key_values[size_t(i) * dim])) * sign);
            const auto ids = download(result.index.indices, result.index.count, stream.value);
            check(result.index.count == 512 && download(result.index.scores, capacity, stream.value) == expected &&
                  ids == select(expected, position + 1, 128), "ratio-1 causal top-512 including signed/zero weights and ties");
            validate_candidates(expected, batch, position, download(result.candidates.blocks, result.candidates.count, stream.value), "query candidates");
        }
    }
    const auto batch = prefill(capacity);
    const auto saved = index.forward(x, qr, keys, batch, capacity - 1, 0, stream.value);
    const auto before_ids = download(saved.index.indices, saved.index.count, stream.value);
    const auto before_scores = download(saved.index.scores, capacity, stream.value);
    const auto before_blocks = download(saved.candidates.blocks, saved.candidates.count, stream.value);
    const auto before_query = download(index.query(), query_size, stream.value);
    const auto before_weights = download(index.head_weights(), 32, stream.value);
    rejects("query key capacity overflow", [&] { index.forward(x, qr, keys, prefill(641), 0, 0, stream.value); });
    rejects("negative query row", [&] { index.forward(x, qr, keys, batch, -1, 0, stream.value); });
    rejects("query row overflow", [&] { index.forward(x, qr, keys, batch, 640, 0, stream.value); });
    rejects("negative query offset", [&] { index.forward(x, qr, keys, batch, 0, -1, stream.value); });
    rejects("query offset overflow", [&] { index.forward(x, qr, keys, batch, 0, INT_MAX, stream.value); });
    rejects("null query x", [&] { index.forward(nullptr, qr, keys, batch, 0, 0, stream.value); });
    rejects("null lowrank query", [&] { index.forward(x, nullptr, keys, batch, 0, 0, stream.value); });
    rejects("null key history", [&] { index.forward(x, qr, nullptr, batch, 0, 0, stream.value); });
    rejects("unaligned input", [&] { index.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(x) + 1), qr, keys, batch, 0, 0, stream.value); });
    rejects("input address overflow", [&] { index.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), qr, keys, batch, 0, 0, stream.value); });
    rejects("query arena cannot be input", [&] { index.forward(saved.index.scores, qr, keys, batch, 0, 0, stream.value); });
    rejects("candidate arena cannot be input", [&] { index.forward(reinterpret_cast<const float*>(saved.candidates.blocks), qr, keys, batch, 0, 0, stream.value); });
    rejects("null query stream", [&] { index.forward(x, qr, keys, batch, 0, 0, nullptr); });
    rejects("query stream switch", [&] { index.forward(x, qr, keys, batch, 0, 0, other.value); });
    check(before_ids == download(saved.index.indices, saved.index.count, stream.value) && same(before_scores, download(saved.index.scores, capacity, stream.value)) &&
          before_blocks == download(saved.candidates.blocks, saved.candidates.count, stream.value) && same(before_query, download(index.query(), query_size, stream.value)) &&
          same(before_weights, download(index.head_weights(), 32, stream.value)), "all argument failures preserve both results and diagnostics");
    const auto again = index.forward(x, qr, keys, decode(capacity), 0, 0, stream.value);
    check(before_ids == download(again.index.indices, again.index.count, stream.value) && before_blocks == download(again.candidates.blocks, again.candidates.count, stream.value),
          "same prefill/decode query has identical results");
    check(same(input, download(x, input.size(), stream.value)) && same(lowrank, download(qr, lowrank.size(), stream.value)) &&
          same(key_values, download(keys, key_values.size(), stream.value)), "query inputs and key history preserved");
    check(index.device_bytes() == bytes && index.sort_workspace_bytes() == workspace, "query calls retain startup allocations");
    // Exact analytical cancellation case: 2^24 + 5118 ones - 2^24 = 5118.
    // Serial F32 loses the first 31 ones; the existing warp reduction preserves
    // them. BF16 projection then scaling must give 5120/64 == 80, not 79.5.
    std::vector<uint16_t> cancellation(32 * hidden, 0);
    std::fill(cancellation.begin(), cancellation.begin() + hidden, uint16_t(0x3f80));
    cancellation[0] = kernels::bf16_from_f32(std::ldexp(1.f, 24));
    cancellation[32] = kernels::bf16_from_f32(-std::ldexp(1.f, 24));
    CudaSource20IndexQuery stable({w.query, cancellation}, 1);
    upload(x, std::vector<float>(hidden, 1.f), stream.value);
    auto one_key = std::vector<uint16_t>(dim, 0); one_key[0] = 0x3f80;
    upload(keys, one_key, stream.value);
    const auto stable_result = stable.forward(x, qr, keys, prefill(1), 0, 0, stream.value);
    check(download(stable.head_weights(), 1, stream.value)[0] == 80.f &&
          download(stable_result.index.scores, 1, stream.value)[0] == 80.f,
          "source-20 head projection preserves BF16 rounding under cancellation");
    std::printf("query_capacity=%d query_device_bytes=%zu sort_workspace_bytes=%zu\n", index.capacity(), bytes, workspace);
}
std::string string_value(const GgufModel& model, const std::string& name) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::STRING) throw std::runtime_error("Missing fixture string " + name);
    return value->s;
}
std::vector<uint8_t> raw_bytes(const GgufModel& model, const std::string& name, size_t count) {
    size_t shard = 0; const auto* value = model.find(name, &shard);
    if (!value || value->type != 24 || tensor_payload_bytes(*value) != count || !model.in_bounds(*value, shard))
        throw std::runtime_error("Malformed byte fixture " + name);
    const auto* bytes = model.shard(shard).tensor_data(*value); return {bytes, bytes + count};
}
void candidate_real(const GgufModel& model) {
    if (string_value(model, "fixture.candidate_encoding") != "sorted-i32-little-endian-bytes-minus-one-padding")
        throw std::runtime_error("Wrong candidate fixture encoding");
    Stream stream; CudaSource20Candidates selector(Source20Batch::max_context);
    core::DeviceArena arena(size_t(Source20Batch::max_context) * 4);
    auto* input = static_cast<float*>(arena.alloc(size_t(Source20Batch::max_context) * 4));
    const int rows = number(model, "fixture.candidate_rows", 20);
    if (rows != 20) throw std::runtime_error("Incomplete official candidate fixture");
    int future_cases = 0, pruned_cases = 0;
    for (int row = 0; row < rows; ++row) {
        const std::string prefix = "candidate." + std::to_string(row) + ".";
        const int width = number(model, prefix + "width", Source20Batch::max_context);
        const int visible = number(model, prefix + "visible", width);
        if (width < 1 || visible < 1) throw std::runtime_error("Empty candidate row");
        const bool is_decode = number(model, prefix + "decode", 1) != 0;
        if (is_decode && visible != width) throw std::runtime_error("Malformed candidate decode row");
        const auto batch = is_decode ? decode(width) : prefill(width);
        const int query = is_decode ? 0 : visible - 1;
        const auto scores = tensor<float>(model, prefix + "scores", width);
        const auto mask = raw_bytes(model, prefix + "mask", width);
        const auto packed = raw_bytes(model, prefix + "blocks", size_t(batch.candidate_stride()) * 4);
        std::vector<int32_t> reference_ids(size_t(batch.candidate_stride()));
        for (size_t i = 0; i < reference_ids.size(); ++i) {
            const uint32_t value = uint32_t(packed[i * 4]) | uint32_t(packed[i * 4 + 1]) << 8 |
                                   uint32_t(packed[i * 4 + 2]) << 16 | uint32_t(packed[i * 4 + 3]) << 24;
            reference_ids[i] = value == UINT32_MAX ? -1 : int32_t(value);
        }
        upload(input, scores, stream.value);
        const auto result = selector.forward(input, batch, query, stream.value);
        const auto actual = download(result.blocks, result.count, stream.value);
        validate_candidates(scores, batch, query, actual, prefix);
        validate_source20_candidates(batch, query, scores, reference_ids);
        const auto block_values = block_scores(scores, visible);
        check(selected_scores(block_values, actual) == selected_scores(block_values, reference_ids), prefix + "Torch candidate score membership including ties");
        bool exact_mask = true, future = false;
        const auto end = std::find(actual.begin(), actual.end(), -1);
        for (int i = 0; i < width; ++i) {
            const bool selected = std::binary_search(actual.begin(), end, i / 8);
            exact_mask &= selected == (mask[size_t(i)] != 0);
            future |= selected && i >= visible;
        }
        if (string_value(model, prefix + "mode") != "ties") check(exact_mask, prefix + "every original candidate mask bit matches");
        check(same(scores, download(input, width, stream.value)), prefix + "original score row preserved");
        future_cases += future; pruned_cases += (width + 7) / 8 > 2048 && end - actual.begin() == 2048;
    }
    check(future_cases > 0 && pruned_cases >= 4, "candidate partial-future mask and actual pruning covered");
    std::printf("official_candidate_rows=%d max_context_candidate_bytes=%zu\n", rows, selector.device_bytes());
}
void real(const char* fixture_path, const char* contract_path) {
    const auto model = GgufModel::open(fixture_path), contract = GgufModel::open(contract_path);
    if (string_value(model, "general.architecture") != "deepseek41-source20-index-query-fixture" ||
        string_value(contract, "general.architecture") != "deepseek41-source20-contract-fixture" ||
        string_value(model, "strata.deepseek41.revision") != "1bae0e1088993b9b82de08f59de5be5a34e42170" ||
        string_value(contract, "strata.deepseek41.revision") != string_value(model, "strata.deepseek41.revision") ||
        string_value(model, "strata.deepseek41.official_source_sha256") != "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d" ||
        string_value(contract, "strata.deepseek41.official_source_sha256") != string_value(model, "strata.deepseek41.official_source_sha256") ||
        string_value(model, "strata.deepseek41.official_kernel_sha256") != "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455" ||
        string_value(contract, "strata.deepseek41.source_hashes") != string_value(model, "strata.deepseek41.compressor_source_hashes"))
        throw std::runtime_error("Wrong source-20 reference identity");
    candidate_real(contract);
    Geometry g; g.layers = 40; g.hidden = hidden; g.compress_ratios.resize(40); g.compress_ratios[20] = 1;
    g.sources.resize(40); g.sources[20] = {20,20,-1};
    CudaSource20Compressor compressor(read_source20_compressor_weights(contract, g), 640, 640);
    CudaSource20IndexKeyCache cache(read_source20_index_key_weights(model, g), 640);
    CudaSource20IndexQuery index(read_source20_index_query_weights(model, g), 16393);
    Stream stream;
    constexpr int tokens = 17, guard = 64;
    core::DeviceArena arena(size_t(tokens * hidden + tokens * qr_dim + 640 * 512 + query_size + 8 * guard) * 4 +
                            size_t(16393 * dim + query_size + 4 * guard) * 2 + 2048);
    auto* x_storage = static_cast<float*>(arena.alloc(size_t(tokens * hidden + 2 * guard) * 4));
    auto* qr_storage = static_cast<float*>(arena.alloc(size_t(tokens * qr_dim + 2 * guard) * 4));
    auto* latent_storage = static_cast<float*>(arena.alloc(size_t(640 * 512 + 2 * guard) * 4));
    auto* projected_storage = static_cast<float*>(arena.alloc(size_t(query_size + 2 * guard) * 4));
    auto* history_storage = static_cast<uint16_t*>(arena.alloc(size_t(16393 * dim + 2 * guard) * 2));
    auto* query_storage = static_cast<uint16_t*>(arena.alloc(size_t(query_size + 2 * guard) * 2));
    float* x = x_storage + guard; float* qr = qr_storage + guard;
    float* latent = latent_storage + guard; float* projected = projected_storage + guard;
    uint16_t* history = history_storage + guard; uint16_t* query_out = query_storage + guard;
    upload(x_storage, std::vector<float>(tokens * hidden + 2 * guard, -333.f), stream.value);
    upload(qr_storage, std::vector<float>(tokens * qr_dim + 2 * guard, -333.f), stream.value);
    upload(latent_storage, std::vector<float>(640 * 512 + 2 * guard, -333.f), stream.value);
    upload(projected_storage, std::vector<float>(query_size + 2 * guard, -333.f), stream.value);
    upload(history_storage, std::vector<uint16_t>(16393 * dim + 2 * guard, 0x5a5a), stream.value);
    upload(query_storage, std::vector<uint16_t>(query_size + 2 * guard, 0x5a5a), stream.value);
    const auto original_projected = tensor<float>(model, "rope.input", query_size);
    upload(projected, original_projected, stream.value);
    const int rope_cases = number(model, "fixture.rope_cases", 6);
    if (rope_cases != 6) throw std::runtime_error("Missing RoPE cases");
    for (int i = 0; i < rope_cases; ++i) {
        const auto prefix = "rope." + std::to_string(i) + ".";
        postprocess_index_query(projected, query_out, number(model, prefix + "position", 1048575), stream.value);
        exact(download(query_out, query_size, stream.value), tensor<uint16_t>(model, prefix + "output", query_size), prefix + "all query heads use token position");
    }
    check(same(original_projected, download(projected, query_size, stream.value)), "original projected query preserved");
    int query_rows = 0;
    auto compare = [&](const std::string& prefix, const Source20Batch& batch, const uint16_t* keys) {
        const int length = batch.latent_count(), count = batch.key_count(), stride = std::min(count, 512);
        const auto input = tensor<float>(model, prefix + "x", size_t(length) * hidden);
        const auto lowrank = tensor<float>(model, prefix + "qr", size_t(length) * qr_dim);
        upload(x, input, stream.value); upload(qr, lowrank, stream.value);
        const auto queries = tensor<uint16_t>(model, prefix + "query", size_t(length) * query_size);
        const auto head_weights = tensor<float>(model, prefix + "weights", size_t(length) * 32);
        const auto scores = tensor<float>(model, prefix + "scores", size_t(length) * count);
        const auto index_floats = tensor<float>(model, prefix + "indices", size_t(length) * stride);
        const auto candidate_floats = tensor<float>(model, prefix + "candidates", size_t(length) * batch.candidate_stride());
        const auto saved_keys = download(keys, size_t(count) * dim, stream.value);
        for (int row = 0; row < length; ++row) {
            const auto call = prefix + "row." + std::to_string(row) + ".";
            const auto result = index.forward(x + size_t(row) * hidden, qr + size_t(row) * qr_dim, keys, batch, row, 128, stream.value);
            exact(download(index.query(), query_size, stream.value), {queries.begin() + size_t(row) * query_size, queries.begin() + size_t(row + 1) * query_size}, call + "query");
            scores_exact(download(index.head_weights(), 32, stream.value), {head_weights.begin() + row * 32, head_weights.begin() + (row + 1) * 32}, call + "head weights");
            const std::vector<float> expected(scores.begin() + size_t(row) * count, scores.begin() + size_t(row + 1) * count);
            scores_exact(download(result.index.scores, count, stream.value), expected, call + "causal scores");
            const auto actual_ids = download(result.index.indices, result.index.count, stream.value);
            const std::vector<int32_t> torch_ids(index_floats.begin() + size_t(row) * stride, index_floats.begin() + size_t(row + 1) * stride);
            check(result.index.count == stride && actual_ids == select(expected, batch.visible_keys(row), 128), call + "canonical unfiltered causal top-512");
            check(selected_scores(expected, actual_ids, 128) == selected_scores(expected, torch_ids, 128), call + "Torch top-512 score membership with ties");
            const auto blocks = download(result.candidates.blocks, result.candidates.count, stream.value);
            const std::vector<int32_t> torch_blocks(candidate_floats.begin() + size_t(row) * batch.candidate_stride(), candidate_floats.begin() + size_t(row + 1) * batch.candidate_stride());
            validate_candidates(expected, batch, row, blocks, call);
            validate_source20_candidates(batch, row, expected, torch_blocks);
            const auto block_values = block_scores(expected, batch.visible_keys(row));
            check(selected_scores(block_values, blocks) == selected_scores(block_values, torch_blocks), call + "Torch candidate score membership with ties");
            ++query_rows;
        }
        check(same(input, download(x, input.size(), stream.value)) && same(lowrank, download(qr, lowrank.size(), stream.value)), prefix + "x and lowrank query preserved");
        check(same(saved_keys, download(keys, saved_keys.size(), stream.value)), prefix + "key history preserved");
    };
    const size_t bytes = index.device_bytes(), workspace = index.sort_workspace_bytes();
    const int plans = number(model, "fixture.plans", 9); if (plans != 9) throw std::runtime_error("Incomplete query plans");
    int calls = 0; std::vector<uint16_t> baseline;
    for (int plan = 0; plan < plans; ++plan) {
        const auto prefix = "plan." + std::to_string(plan) + ".";
        const int length = number(model, prefix + "tokens", tokens), first = number(model, prefix + "prefill", tokens);
        if (length < 1 || first < 1 || first > length) throw std::runtime_error("Invalid sequence plan");
        const auto input = tensor<float>(model, prefix + "x", size_t(length) * hidden);
        compressor.reset(stream.value); cache.reset(stream.value);
        for (int start = 0; start < length;) {
            const int count = start == 0 ? first : 1;
            const auto call = prefix + "at." + std::to_string(start) + ".";
            upload(x, std::vector<float>(input.begin() + size_t(start) * hidden, input.begin() + size_t(start + count) * hidden), stream.value);
            if (count > cache.capacity() - cache.size()) throw std::runtime_error("Test key capacity exhausted");
            const auto batch = compressor.forward(x, latent, count, start, stream.value);
            const auto preserved = download(latent, size_t(count) * 512, stream.value);
            cache.append(latent, batch, stream.value);
            exact(download(cache.data(), size_t(batch.key_count()) * dim, stream.value), tensor<uint16_t>(model, call + "keys", size_t(batch.key_count()) * dim), call + "native compressor/key cache");
            compare(call, batch, cache.data());
            check(same(preserved, download(latent, preserved.size(), stream.value)), call + "pre-RoPE latent preserved");
            check(cache.size() == batch.key_count() && compressor.next_position() == batch.key_count(), call + "producer and cache positions agree");
            start += count; ++calls;
        }
        const auto actual = download(cache.data(), size_t(length) * dim, stream.value);
        if (plan == 0) baseline = actual; else if (plan < 6) check(same(actual, baseline), prefix + "partition/reset key parity");
    }
    check(calls == 58, "all fixed source-20 sequence calls consumed");
    cache.reset(stream.value);
    const auto long_latent = tensor<float>(model, "long.latent", 640 * 512);
    upload(latent, long_latent, stream.value); cache.append(latent, prefill(640), stream.value);
    exact(download(cache.data(), 640 * dim, stream.value), tensor<uint16_t>(model, "long.keys", 640 * dim), "640 original-weight keys");
    const int decode_cases = number(model, "fixture.decode_cases", 8); if (decode_cases != 8) throw std::runtime_error("Missing decode cases");
    for (int i = 0; i < decode_cases; ++i) {
        const auto prefix = "decode." + std::to_string(i) + ".";
        compare(prefix, decode(number(model, prefix + "tokens", 640)), cache.data());
    }
    const auto long_history = tensor<uint16_t>(model, "selection.keys", 16393 * dim);
    upload(history, long_history, stream.value);
    const int selection_cases = number(model, "fixture.selection_cases", 2); if (selection_cases != 2) throw std::runtime_error("Missing pruning cases");
    for (int i = 0; i < selection_cases; ++i) {
        const auto prefix = "selection." + std::to_string(i) + ".";
        compare(prefix, decode(number(model, prefix + "tokens", 16393)), history);
    }
    check(query_rows == 118, "all 118 original-weight query rows consumed");
    for (const auto& [storage, size] : {std::pair{x_storage, tokens * hidden}, std::pair{qr_storage, tokens * qr_dim},
                                       std::pair{latent_storage, 640 * 512}, std::pair{projected_storage, query_size}}) {
        check(download(storage, guard, stream.value) == std::vector<float>(guard, -333.f) &&
              download(storage + guard + size, guard, stream.value) == std::vector<float>(guard, -333.f), "F32 caller allocation guards intact");
    }
    for (const auto& [storage, size] : {std::pair{history_storage, 16393 * dim}, std::pair{query_storage, query_size}}) {
        check(download(storage, guard, stream.value) == std::vector<uint16_t>(guard, 0x5a5a) &&
              download(storage + guard + size, guard, stream.value) == std::vector<uint16_t>(guard, 0x5a5a), "BF16 caller allocation guards intact");
    }
    check(index.device_bytes() == bytes && index.sort_workspace_bytes() == workspace, "all real queries retain bounded allocations");
    std::printf("query_capacity=%d query_device_bytes=%zu sort_workspace_bytes=%zu compressor_and_key_cache_bytes=%zu\n",
                index.capacity(), bytes, workspace, compressor.device_bytes() + cache.device_bytes());
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 4 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 4) real(argv[2], argv[3]); else { candidate_synthetic(); query_synthetic(); }
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); ++failures; }
    std::printf("bf16_values=%zu bf16_differences=%zu score_values=%zu score_differences=%zu\n", bf16_values, bf16_differences, score_values, score_differences);
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
