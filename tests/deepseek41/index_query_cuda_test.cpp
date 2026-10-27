#include "strata/deepseek41/index_key_cuda.hpp"
#include "strata/deepseek41/index_query_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <climits>
#include <numeric>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int dim = 128, hidden = 5120, qr_dim = 1280, query_size = 4096;
int checks = 0, failures = 0;
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
    std::printf("  %s different=%zu/%zu max_abs=%.9g\n", name.c_str(), different, actual.size(), maximum);
    check(different == 0, name);
}
void synthetic() {
    Stream stream, other;
    constexpr int capacity = 640;
    std::vector<uint8_t> codes(query_size * qr_dim, 0), scales(128 * 40, 127);
    std::vector<uint16_t> weights(32 * hidden, 0);
    for (int h = 0; h < 32; ++h) codes[size_t(h) * dim * qr_dim] = 0x38; // each head: q[0]=qr[0]
    weights[0] = kernels::bf16_from_f32(1.f);
    IndexQueryWeights w{{codes, scales, query_size, qr_dim}, weights};
    for (int cap : {0, -1, 524289}) rejects("invalid capacity refused", [&] { CudaIndexQuery bad(w, cap); });
    rejects("wrong query geometry refused", [&] { auto bad = w; bad.query.rows = 128; CudaIndexQuery object(bad, 1); });
    rejects("truncated FP8 payload refused", [&] { auto bad = w; bad.query.values = {}; CudaIndexQuery object(bad, 1); });
    rejects("wrong head projection width refused", [&] { auto bad = w; bad.head_weights = {}; CudaIndexQuery object(bad, 1); });
    weights[0] = 0x7fc0;
    rejects("nonfinite BF16 weight refused", [&] { CudaIndexQuery bad(w, 1); });
    weights[0] = kernels::bf16_from_f32(1.f); codes[0] = 0x7f;
    rejects("nonfinite FP8 weight refused", [&] { CudaIndexQuery bad(w, 1); }); codes[0] = 0x38;
    CudaIndexQuery index(w, capacity);
    core::DeviceArena arena((hidden + qr_dim) * 4 + capacity * dim * 2 + 512);
    auto* x = static_cast<float*>(arena.alloc(hidden * 4)); auto* qr = static_cast<float*>(arena.alloc(qr_dim * 4));
    auto* keys = static_cast<uint16_t*>(arena.alloc(capacity * dim * 2));
    std::vector<float> input(hidden, 0), lowrank(qr_dim, 0); input[0] = 64.f; lowrank[0] = 1.f;
    std::vector<uint16_t> key_values(capacity * dim, 0);
    for (int i = 0; i < capacity; ++i) key_values[i * dim] = kernels::bf16_from_f32(float((i * 73) % 641 - 320));
    upload(x, input, stream.value); upload(qr, lowrank, stream.value); upload(keys, key_values, stream.value);
    const size_t owned = index.device_bytes(), workspace = index.sort_workspace_bytes();
    check(index.capacity() == capacity && workspace > 0 && owned > workspace, "explicit capacity and startup sort workspace");
    const auto empty = index.forward(x, qr, nullptr, 0, 0, 128, stream.value);
    check(empty.count == 0, "first token with no completed key returns empty selection");
    for (float sign : {1.f, -1.f, 0.f}) {
        input[0] = sign * 64.f; upload(x, input, stream.value);
        for (int pos : {0, 1, 126, 127, 1023, 1025, 1279}) {
            const int visible = (pos + 1) / 2, offset = 128;
            const auto result = index.forward(x, qr, keys, capacity, pos, offset, stream.value);
            std::vector<float> expected(capacity, -INFINITY);
            for (int i = 0; i < visible; ++i) expected[i] = bf(std::max(0.f, kernels::f32_from_bf16(key_values[i * dim])) * sign);
            const auto actual = download(result.indices, result.count, stream.value);
            check(result.count == 512 && download(result.scores, capacity, stream.value) == expected && actual == select(expected, visible, offset),
                  "causal top-512 including negative weights, ReLU, cutoff ties and padding at " + std::to_string(pos));
        }
    }
    const auto saved = index.forward(x, qr, keys, capacity, 1279, 0, stream.value);
    const auto before = download(saved.indices, saved.count, stream.value);
    const auto previous_scores = download(saved.scores, capacity, stream.value);
    rejects("negative count refused", [&] { index.forward(x, qr, keys, -1, 0, 0, stream.value); });
    rejects("capacity overflow refused", [&] { index.forward(x, qr, keys, 641, 0, 0, stream.value); });
    rejects("missing completed group refused", [&] { index.forward(x, qr, keys, 639, 1279, 0, stream.value); });
    rejects("negative position refused", [&] { index.forward(x, qr, keys, 1, -1, 0, stream.value); });
    rejects("context overflow refused", [&] { index.forward(x, qr, keys, 640, 1048576, 0, stream.value); });
    rejects("negative offset refused", [&] { index.forward(x, qr, keys, 640, 1279, -1, stream.value); });
    rejects("offset overflow refused", [&] { index.forward(x, qr, keys, 640, 1279, INT_MAX, stream.value); });
    rejects("null input refused", [&] { index.forward(nullptr, qr, keys, 640, 1279, 0, stream.value); });
    rejects("null low-rank query refused", [&] { index.forward(x, nullptr, keys, 640, 1279, 0, stream.value); });
    rejects("null keys refused", [&] { index.forward(x, qr, nullptr, 640, 1279, 0, stream.value); });
    rejects("misaligned input refused", [&] { index.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(x) + 1), qr, keys, 640, 1279, 0, stream.value); });
    rejects("overflowing input range refused", [&] { index.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), qr, keys, 640, 1279, 0, stream.value); });
    rejects("owned query cannot be cache input", [&] { index.forward(x, qr, index.query(), 640, 1279, 0, stream.value); });
    rejects("null stream refused", [&] { index.forward(x, qr, keys, 640, 1279, 0, nullptr); });
    rejects("stream switch refused", [&] { index.forward(x, qr, keys, 640, 1279, 0, other.value); });
    check(same(before, download(saved.indices, saved.count, stream.value)) && same(previous_scores, download(saved.scores, capacity, stream.value)), "rejected calls preserve previous selection and scores");
    const auto edge = index.forward(x, qr, keys, 640, 1279, INT_MAX - 639, stream.value);
    check(download(edge.indices, edge.count, stream.value).back() == INT_MAX - 128, "largest safe offset accepted");
    check(index.device_bytes() == owned && index.sort_workspace_bytes() == workspace && same(key_values, download(keys, key_values.size(), stream.value)) && same(input, download(x, input.size(), stream.value)) && same(lowrank, download(qr, lowrank.size(), stream.value)), "inputs and bounded allocations preserved");
    rejects("query RoPE negative position refused", [&] { postprocess_index_query(x, keys, -1, stream.value); });
    rejects("query RoPE context overflow refused", [&] { postprocess_index_query(x, keys, 1048576, stream.value); });
    rejects("query RoPE overlap refused", [&] { postprocess_index_query(x, reinterpret_cast<uint16_t*>(x), 1, stream.value); });
    // Exercise CUB's multi-block sorting policy and its startup-sized workspace.
    constexpr int wide_count = 8192;
    CudaIndexQuery wide(w, wide_count);
    core::DeviceArena wide_arena(size_t(wide_count) * dim * 2);
    auto* wide_keys = static_cast<uint16_t*>(wide_arena.alloc(wide_arena.capacity()));
    std::vector<uint16_t> wide_values(size_t(wide_count) * dim, 0);
    for (int i = 0; i < wide_count; ++i) wide_values[size_t(i) * dim] = kernels::bf16_from_f32(float((i * 73) % 8191 - 4096));
    upload(wide_keys, wide_values, stream.value);
    const size_t wide_bytes = wide.device_bytes();
    check(wide.sort_workspace_bytes() > 1, "multi-block sort workspace allocated at startup");
    for (float sign : {1.f, -1.f}) {
        input[0] = sign * 64.f; upload(x, input, stream.value);
        for (int n : {8192, 513}) {
            const auto selected = wide.forward(x, qr, wide_keys, n, n * 2 - 1, 128, stream.value);
            std::vector<float> expected(n);
            for (int i = 0; i < n; ++i) expected[i] = bf(std::max(0.f, kernels::f32_from_bf16(wide_values[size_t(i) * dim])) * sign);
            check(download(selected.scores, n, stream.value) == expected && download(selected.indices, selected.count, stream.value) == select(expected, n, 128),
                  "wide-capacity score/top-k with actual group count " + std::to_string(n));
        }
    }
    check(wide.device_bytes() == wide_bytes, "multi-block sort reuses its fixed allocation");
    std::printf("wide_capacity=%d device_bytes=%zu sort_workspace_bytes=%zu\n", wide_count, wide_bytes, wide.sort_workspace_bytes());
    std::printf("synthetic_capacity=%d device_bytes=%zu sort_workspace_bytes=%zu\n", capacity, owned, workspace);
}
void real(const char* path) {
    const auto model = GgufModel::open(path);
    Geometry g; g.layers = 40; g.hidden = hidden; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
    CudaIndexQuery index(read_index_query_weights(model, g, 2), 640);
    CudaIndexKeyCache cache(read_index_key_weights(model, g, 2), 640);
    Stream stream;
    core::DeviceArena arena((640 * 512 + 13 * hidden + 13 * qr_dim) * 4 + query_size * 2 + 1024);
    auto* latent = static_cast<float*>(arena.alloc(640 * 512 * 4));
    auto* x = static_cast<float*>(arena.alloc(13 * hidden * 4)); auto* qr = static_cast<float*>(arena.alloc(13 * qr_dim * 4));
    auto* query_out = static_cast<uint16_t*>(arena.alloc(query_size * 2));
    const auto projected = tensor<float>(model, "rope.input", query_size); upload(x, projected, stream.value);
    const int cases = number(model, "fixture.rope_cases", 6); check(cases == 6, "all odd/even/long query RoPE cases present");
    for (int i = 0; i < cases; ++i) {
        const std::string prefix = "rope." + std::to_string(i) + ".";
        postprocess_index_query(x, query_out, number(model, prefix + "position", 1048575), stream.value);
        exact(download(query_out, query_size, stream.value), tensor<uint16_t>(model, prefix + "output", query_size), prefix + "all heads share token position");
    }
    check(same(projected, download(x, projected.size(), stream.value)), "query post-processing preserves input");
    auto compare = [&](const std::string& prefix, int tokens, int start, int count, int offset) {
        const auto input = tensor<float>(model, prefix + "x", size_t(tokens) * hidden), lowrank = tensor<float>(model, prefix + "qr", size_t(tokens) * qr_dim);
        upload(x, input, stream.value); upload(qr, lowrank, stream.value);
        const auto ref_q = tensor<uint16_t>(model, prefix + "query", size_t(tokens) * query_size);
        const auto ref_w = tensor<float>(model, prefix + "weights", size_t(tokens) * 32), ref_s = tensor<float>(model, prefix + "scores", size_t(tokens) * count);
        const auto ref_ids_float = tensor<float>(model, prefix + "indices", size_t(tokens) * std::min(512, count));
        const std::vector<int32_t> ref_ids(ref_ids_float.begin(), ref_ids_float.end());
        for (int t = 0; t < tokens; ++t) {
            const std::string call = prefix + "token." + std::to_string(t) + ".";
            const auto result = index.forward(x + size_t(t) * hidden, qr + size_t(t) * qr_dim, cache.data(), count, start + t, offset, stream.value);
            exact(download(index.query(), query_size, stream.value), {ref_q.begin() + size_t(t) * query_size, ref_q.begin() + size_t(t + 1) * query_size}, call + "query");
            scores_exact(download(index.head_weights(), 32, stream.value), {ref_w.begin() + t * 32, ref_w.begin() + (t + 1) * 32}, call + "head weights");
            const std::vector<float> expected(ref_s.begin() + size_t(t) * count, ref_s.begin() + size_t(t + 1) * count);
            scores_exact(download(result.scores, count, stream.value), expected, call + "scores");
            const auto actual = download(result.indices, result.count, stream.value);
            check(result.count == std::min(512, count) && actual == select(expected, (start + t + 1) / 2, offset), call + "canonical causal top-k");
            const std::vector<int32_t> torch_ids(ref_ids.begin() + size_t(t) * result.count, ref_ids.begin() + size_t(t + 1) * result.count);
            std::vector<float> a, b;
            for (auto id : actual) if (id >= 0) a.push_back(expected.at(size_t(id - offset)));
            for (auto id : torch_ids) if (id >= 0) b.push_back(expected.at(size_t(id - offset)));
            std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
            std::printf("  %s torch_ids_equal=%d\n", call.c_str(), actual == torch_ids);
            check(a == b, call + "Torch selected scores match including cutoff ties");
        }
        check(same(input, download(x, input.size(), stream.value)) && same(lowrank, download(qr, lowrank.size(), stream.value)), prefix + "caller inputs preserved");
    };
    const auto prefill_latent = tensor<float>(model, "prefill.latent", 6 * 512); upload(latent, prefill_latent, stream.value);
    cache.append(latent, {6, 0}, stream.value);
    exact(download(cache.data(), 6 * dim, stream.value), tensor<uint16_t>(model, "prefill.keys", 6 * dim), "prefill real key cache");
    compare("prefill.", 13, 0, 6, 13);
    cache.reset(stream.value);
    const auto long_latent = tensor<float>(model, "long.latent", 640 * 512); upload(latent, long_latent, stream.value);
    cache.append(latent, {640, 0}, stream.value);
    const auto keys = download(cache.data(), 640 * dim, stream.value);
    exact(keys, tensor<uint16_t>(model, "long.keys", 640 * dim), "640-group real key cache");
    const int decode_cases = number(model, "fixture.decode_cases", 8); check(decode_cases == 8, "all selection threshold and tie cases present");
    for (int i = 0; i < decode_cases; ++i) {
        const std::string prefix = "decode." + std::to_string(i) + ".";
        compare(prefix, 1, number(model, prefix + "position", 1279), number(model, prefix + "groups", 640), 128);
    }
    check(same(keys, download(cache.data(), keys.size(), stream.value)), "query and selection leave key history intact");
    std::printf("query_capacity=%d query_device_bytes=%zu sort_workspace_bytes=%zu key_cache_bytes=%zu\n", index.capacity(), index.device_bytes(), index.sort_workspace_bytes(), cache.device_bytes());
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 3 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 3) real(argv[2]); else synthetic();
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
