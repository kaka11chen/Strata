#include "strata/deepseek41/index_key_cuda.hpp"
#include "strata/deepseek41/compressed_kv_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int dim = 128, latent_dim = 512, hidden = 5120;
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
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
float nearest(float x, const std::vector<float>& grid) {
    size_t best = 0;
    for (size_t i = 1; i < grid.size(); ++i)
        if (std::abs(x - grid[i]) < std::abs(x - grid[best]) ||
            (std::abs(x - grid[i]) == std::abs(x - grid[best]) && i % 2 == 0)) best = i;
    return grid[best];
}
// Scalar RoPE and nearest-even FP4 oracle, with power-of-two ceiling scales.
std::vector<uint16_t> reference(std::vector<float> x, int first_position = 0) {
    for (size_t row = 0; row < x.size() / dim; ++row)
        for (int pair = 0; pair < 32; ++pair) {
            const float raw = 1.f / float(std::pow(160000., double(pair) / 32.));
            const float smooth = 1.f - std::clamp(float(pair - 15) / 10.f, 0.f, 1.f);
            const float freq = (raw / 16.f) * (1.f - smooth) + raw * smooth;
            const float angle = float(first_position + 2 * int(row)) * freq;
            const float c = std::cos(angle), s = std::sin(angle);
            const size_t at = row * dim + 64 + pair * 2;
            const float a = bf(x[at]), b = bf(x[at + 1]);
            x[at] = bf(a * c - b * s); x[at + 1] = bf(a * s + b * c);
        }
    const std::vector<float> grid4{0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    std::vector<uint16_t> out(x.size());
    for (size_t start = 0; start < x.size(); start += 32) {
        float maximum = 6.f * 0x1p-126f;
        for (size_t i = 0; i < 32; ++i) maximum = std::max(maximum, std::abs(bf(x[start + i])));
        const float scaled = maximum * (1.f / 6.f);
        int exponent = 0; const float fraction = std::frexp(scaled, &exponent);
        const float scale = std::ldexp(1.f, fraction == .5f ? exponent - 1 : exponent);
        for (size_t i = 0; i < 32; ++i) {
            const float value = bf(x[start + i]);
            const float code = nearest(std::min(std::abs(value) / scale, 6.f), grid4);
            out[start + i] = kernels::bf16_from_f32(std::copysign(code, value) * scale);
        }
    }
    return out;
}
void synthetic() {
    Stream stream, other;
    const float values[] = {0.f, -0.f, .25f, -.25f, .75f, -.75f, 1.25f, -1.25f,
                            1.75f, -1.75f, 2.5f, -2.5f, 3.5f, -3.5f, 5.f, 6.f};
    const float scales[] = {1.f, 1.0078125f, .9921875f, 0x1p-126f, 0x1p-130f, 0x1p117f};
    std::vector<float> input(6 * dim, 0.f);
    for (int row = 0; row < 6; ++row)
        for (int d = 0; d < dim; ++d) input[row * dim + d] = values[d % 16] * scales[(row + d / 32) % 6];
    core::DeviceArena arena(6 * latent_dim * 4 + 6 * dim * 2 + 512);
    auto* in = static_cast<float*>(arena.alloc(6 * latent_dim * 4));
    auto* out = static_cast<uint16_t*>(arena.alloc(6 * dim * 2));
    upload(in, input, stream.value);
    postprocess_index_keys(in, out, 6, 0, stream.value);
    exact(download(out, input.size(), stream.value), reference(input), "E8M0 ceiling scales, FP4 midpoint ties, subnormals and large magnitudes");
    check(same(download(in, input.size(), stream.value), input), "normalized input preserved");
    postprocess_index_keys(in, out, 1, 1048574, stream.value);
    exact(download(out, dim, stream.value), reference({input.begin(), input.begin() + dim}, 1048574), "last context position accepted");
    rejects("odd group position refused", [&] { postprocess_index_keys(in, out, 1, 1, stream.value); });
    rejects("negative group position refused", [&] { postprocess_index_keys(in, out, 1, -2, stream.value); });
    rejects("context overflow refused", [&] { postprocess_index_keys(in, out, 2, 1048574, stream.value); });
    rejects("position outside context refused", [&] { postprocess_index_keys(in, out, 1, 1048576, stream.value); });
    rejects("empty raw transform refused", [&] { postprocess_index_keys(in, out, 0, 0, stream.value); });
    rejects("oversized raw transform refused", [&] { postprocess_index_keys(in, out, 524289, 0, stream.value); });
    rejects("null input refused", [&] { postprocess_index_keys(nullptr, out, 1, 0, stream.value); });
    rejects("null output refused", [&] { postprocess_index_keys(in, nullptr, 1, 0, stream.value); });
    rejects("null stream refused", [&] { postprocess_index_keys(in, out, 1, 0, nullptr); });
    rejects("overlapping buffers refused", [&] { postprocess_index_keys(in, reinterpret_cast<uint16_t*>(in), 1, 0, stream.value); });
    rejects("unaligned input refused", [&] { postprocess_index_keys(reinterpret_cast<float*>(reinterpret_cast<char*>(in) + 1), out, 1, 0, stream.value); });
    rejects("unaligned output refused", [&] { postprocess_index_keys(in, reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(out) + 1), 1, 0, stream.value); });
    rejects("overflowing address range refused", [&] { postprocess_index_keys(reinterpret_cast<const float*>(UINTPTR_MAX - 3), out, 1, 0, stream.value); });
    std::vector<uint16_t> weights(dim * latent_dim, 0), gamma(dim, kernels::bf16_from_f32(1.f));
    for (int d = 0; d < dim; ++d) weights[d * latent_dim + d] = kernels::bf16_from_f32(1.f);
    IndexKeyWeights w{weights, gamma};
    for (int capacity : {0, -1, 524289}) rejects("invalid capacity refused", [&] { CudaIndexKeyCache bad(w, capacity); });
    rejects("short projection refused", [&] { CudaIndexKeyCache bad({std::span(weights).first(10), gamma}, 1); });
    rejects("short norm refused", [&] { CudaIndexKeyCache bad({weights, std::span(gamma).first(10)}, 1); });
    weights[0] = 0x7f80;
    rejects("nonfinite projection refused", [&] { CudaIndexKeyCache bad(w, 1); });
    weights[0] = kernels::bf16_from_f32(1.f); gamma[0] = 0x7fc0;
    rejects("nonfinite norm refused", [&] { CudaIndexKeyCache bad(w, 1); });
    gamma[0] = kernels::bf16_from_f32(1.f);
    CudaIndexKeyCache cache(w, 5);
    check(cache.size() == 0 && cache.capacity() == 5 && cache.device_bytes() == 134400, "fixed weights/scratch and capacity allocation");
    check(download(cache.data(), 5 * dim, stream.value) == std::vector<uint16_t>(5 * dim, 0), "startup cache zeroed before nonblocking reads");
    std::vector<float> latent(6 * latent_dim), normalized(6 * dim);
    for (int row = 0; row < 6; ++row) {
        for (int d = 0; d < latent_dim; ++d) latent[row * latent_dim + d] = ((row + d) % 3 ? -1.0001f : 1.0001f);
        for (int d = 0; d < dim; ++d) normalized[row * dim + d] = bf(latent[row * latent_dim + d]);
    }
    upload(in, latent, stream.value); const auto expected = reference(normalized);
    rejects("null cache stream refused", [&] { cache.append(in, {1, 0}, nullptr); });
    rejects("null cache input refused", [&] { cache.append(nullptr, {1, 0}, stream.value); });
    rejects("misaligned cache input refused", [&] { cache.append(reinterpret_cast<float*>(reinterpret_cast<char*>(in) + 1), {1, 0}, stream.value); });
    rejects("cache input address overflow refused", [&] { cache.append(reinterpret_cast<float*>(UINTPTR_MAX - 3), {1, 0}, stream.value); });
    rejects("first group cannot be skipped", [&] { cache.append(in, {1, 2}, stream.value); });
    rejects("malformed empty descriptor refused", [&] { cache.append(nullptr, {0, 0}, stream.value); });
    rejects("negative count refused", [&] { cache.append(in, {-1, 0}, stream.value); });
    cache.append(nullptr, {0, -1}, stream.value);
    check(cache.size() == 0, "incomplete compressor group emits no key");
    cache.append(in, {3, 0}, stream.value);
    exact(download(cache.data(), 3 * dim, stream.value), {expected.begin(), expected.begin() + 3 * dim}, "BF16 projection, normalization, RoPE and quantized keys");
    const auto partial = download(cache.data(), 5 * dim, stream.value);
    rejects("whole batch refused before partial write", [&] { cache.append(in, {3, 6}, stream.value); });
    check(cache.size() == 3 && same(partial, download(cache.data(), 5 * dim, stream.value)), "overflow leaves history and free slots unchanged");
    cache.append(in + 3 * latent_dim, {2, 6}, stream.value);
    exact(download(cache.data(), 5 * dim, stream.value), {expected.begin(), expected.begin() + 5 * dim}, "fills exact capacity without changing history");
    const auto full = download(cache.data(), 5 * dim, stream.value);
    rejects("full cache refuses overwrite", [&] { cache.append(in, {1, 10}, stream.value); });
    rejects("duplicate group refused", [&] { cache.append(in, {1, 8}, stream.value); });
    rejects("append stream switch refused", [&] { cache.append(nullptr, {0, -1}, other.value); });
    rejects("reset stream switch refused", [&] { cache.reset(other.value); });
    cache.append(nullptr, {0, -1}, stream.value);
    check(cache.size() == 5 && same(full, download(cache.data(), 5 * dim, stream.value)), "full-cache no-op and rejected calls preserve state");
    const auto* address = cache.data(); cache.reset(stream.value);
    check(cache.size() == 0 && cache.data() == address && download(cache.data(), 5 * dim, stream.value) == std::vector<uint16_t>(5 * dim, 0), "reset zeroes cache without reallocating");
    rejects("owned cache cannot be latent input", [&] { cache.append(reinterpret_cast<const float*>(cache.data()), {1, 0}, stream.value); });
    for (int row = 0; row < 5; ++row) cache.append(in + row * latent_dim, {1, row * 2}, stream.value);
    check(same(full, download(cache.data(), 5 * dim, stream.value)), "single-group replay equals batched cache exactly");
    check(cache.device_bytes() == 134400 && same(latent, download(in, latent.size(), stream.value)), "bounded storage and original latent preserved");
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, size_t count) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != (sizeof(T) == 2 ? 30u : 0u) || tensor_payload_bytes(*t) != count * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Malformed index key fixture tensor " + name);
    const auto* data = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*t)); return {data, data + count};
}
int number(const GgufModel& model, const std::string& name, int maximum) {
    const auto* m = model.meta().get(name);
    if (!m || m->type != MetaType::U32 || m->u > uint64_t(maximum)) throw std::runtime_error("Malformed fixture number " + name);
    return int(m->u);
}
void real(const char* fixture_path, const char* compressor_path, const char* kv_path) {
    const auto model = GgufModel::open(fixture_path), weights = GgufModel::open(compressor_path), kv_ref = GgufModel::open(kv_path);
    Geometry g; g.layers = 40; g.hidden = hidden; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
    CudaCompressor compressor(read_compressor_weights(weights, g, 2));
    CudaIndexKeyCache cache(read_index_key_weights(model, g, 2), 8);
    CudaCompressedKvCache kv(8);
    Stream stream;
    core::DeviceArena arena((13 * hidden + 6 * latent_dim) * 4 + 4 * dim * 2 + 512);
    auto* in = static_cast<float*>(arena.alloc(13 * hidden * 4));
    auto* latent = static_cast<float*>(arena.alloc(6 * latent_dim * 4));
    auto* transformed = static_cast<uint16_t*>(arena.alloc(4 * dim * 2));
    const int cases = number(model, "fixture.cases", 32);
    check(cases == 12, "all edge and long-position cases present");
    for (int i = 0; i < cases; ++i) {
        const std::string prefix = "case." + std::to_string(i) + ".";
        const int position = number(model, prefix + "position", 1048574), groups = number(model, prefix + "groups", 4);
        const auto input = tensor<float>(model, prefix + "input", size_t(groups) * dim);
        upload(latent, input, stream.value);
        postprocess_index_keys(latent, transformed, groups, position, stream.value);
        exact(download(transformed, size_t(groups) * dim, stream.value), tensor<uint16_t>(model, prefix + "output", size_t(groups) * dim), prefix + "official RoPE + independent E8M0/FP4");
        check(same(input, download(latent, input.size(), stream.value)), prefix + "input preserved");
    }
    const auto input = tensor<float>(model, "sequence.input", 13 * hidden);
    check(same(input, tensor<float>(kv_ref, "sequence.input", 13 * hidden)), "KV and key oracles use the same sequence");
    upload(in, input, stream.value);
    const int plans = number(model, "fixture.plans", 5);
    check(plans == 5, "odd/even prefill and single-token decode plans present");
    std::vector<uint16_t> baseline;
    for (int plan = 0; plan < plans; ++plan) {
        const std::string prefix = "plan." + std::to_string(plan) + ".";
        const int first = number(model, prefix + "prefill", 13);
        if (!first) throw std::runtime_error("Empty reference prefill");
        compressor.reset(stream.value); cache.reset(stream.value); kv.reset(stream.value);
        int position = 0;
        while (position < 13) {
            const int length = position == 0 ? first : 1;
            const int emitted = (position + length) / 2 - position / 2;
            if (emitted > cache.capacity() - cache.size() || emitted > kv.capacity() - kv.size())
                throw std::runtime_error("Test capacity must be preflighted before compressor advancement");
            const auto info = compressor.forward(in + size_t(position) * hidden, latent, length, position, stream.value);
            const auto before = info.count ? download(latent, size_t(info.count) * latent_dim, stream.value) : std::vector<float>{};
            // Both caches consume the original latent; reversing their order is
            // safe here because neither cache modifies it. Shared ownership is later.
            if (plan % 2) { kv.append(latent, info, stream.value); cache.append(latent, info, stream.value); }
            else { cache.append(latent, info, stream.value); kv.append(latent, info, stream.value); }
            const std::string call = prefix + "at." + std::to_string(position) + ".";
            const int count = (position + length) / 2;
            check(info.count == emitted && cache.size() == count && kv.size() == count && compressor.next_position() == position + length, call + "all component positions agree");
            if (count) {
                exact(download(cache.data(), size_t(count) * dim, stream.value), tensor<uint16_t>(model, call + "cache", size_t(count) * dim), call + "official key prefix");
                exact(download(kv.data(), size_t(count) * latent_dim, stream.value), tensor<uint16_t>(kv_ref, call + "cache", size_t(count) * latent_dim), call + "unchanged official KV prefix");
            }
            if (info.count) check(same(before, download(latent, before.size(), stream.value)), call + "both consumers preserve latent");
            position += length;
        }
        const auto actual = download(cache.data(), 6 * dim, stream.value);
        if (plan == 0) baseline = actual;
        else check(same(actual, baseline), prefix + "reset/partition key cache byte parity");
        check(cache.device_bytes() == 135168 && cache.capacity() == 8, prefix + "bounded key allocation");
    }
    std::printf("key_capacity_groups=%d key_device_bytes=%zu compressor_plus_both_caches_bytes=%zu\n", cache.capacity(), cache.device_bytes(), compressor.device_bytes() + cache.device_bytes() + kv.device_bytes());
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 5 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 5) real(argv[2], argv[3], argv[4]); else synthetic();
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
