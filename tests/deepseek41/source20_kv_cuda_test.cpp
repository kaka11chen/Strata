#include "strata/deepseek41/source20_kv_cuda.hpp"
#include "strata/deepseek41/source20_compressor_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int dim = 512, hidden = 5120, total = 17, guard = 64;
int checks = 0, failures = 0;
size_t compared = 0, mismatched = 0;
void check(bool ok, const std::string& name) {
    ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void ck(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void rejects(const std::string& name, const std::function<void()>& fn) {
    try { fn(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(value); }
};
template<class T> void upload(T* pointer, const std::vector<T>& values, cudaStream_t stream) {
    ck(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
}
template<class T> std::vector<T> download(const T* pointer, size_t count, cudaStream_t stream) {
    if (count == 0) return {};
    std::vector<T> values(count);
    ck(cudaMemcpyAsync(values.data(), pointer, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return values;
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
void exact(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& expected, const std::string& name) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    size_t different = 0;
    for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
        if (different < 8) std::printf("  mismatch index=%zu actual=%04x expected=%04x\n", i, unsigned(actual[i]), unsigned(expected[i]));
        ++different;
    }
    compared += actual.size(); mismatched += different;
    std::printf("  %s differing=%zu/%zu\n", name.c_str(), different, actual.size());
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
// Independent scalar grid reference for zero-tail synthetic vectors. Rotation
// sets zero signs; nonzero RoPE tails are checked against the pinned CPU oracle.
std::vector<uint16_t> reference(std::vector<float> values, int start = 0) {
    for (size_t row = 0; row < values.size() / dim; ++row)
        for (int pair = 0; pair < 32; ++pair) {
            const float raw = 1.f / float(std::pow(160000., double(pair) / 32.));
            const float smooth = 1.f - std::clamp(float(pair - 15) / 10.f, 0.f, 1.f);
            const float frequency = (raw / 16.f) * (1.f - smooth) + raw * smooth;
            const float angle = float(start + int(row)) * frequency;
            const float c = std::cos(angle), s = std::sin(angle);
            const size_t at = row * dim + 448 + pair * 2;
            const float a = bf(values[at]), b = bf(values[at + 1]);
            values[at] = bf(a * c - b * s); values[at + 1] = bf(a * s + b * c);
        }
    std::vector<float> grid8, grid4{0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    for (int code = 0; code < 127; ++code)
        grid8.push_back(code < 8 ? float(code) / 512 : std::ldexp(1.f + float(code & 7) / 8, (code >> 3) - 7));
    std::vector<uint16_t> output(values.size());
    for (size_t offset = 0; offset < values.size(); offset += 16) {
        float maximum = 6.f / 512;
        for (size_t j = 0; j < 16; ++j) maximum = std::max(maximum, std::abs(bf(values[offset + j])));
        const float scale = nearest(maximum / 6, grid8);
        for (size_t j = 0; j < 16; ++j) {
            const float x = bf(values[offset + j]);
            output[offset + j] = kernels::bf16_from_f32(std::copysign(nearest(std::min(std::abs(x) / scale, 6.f), grid4), x) * scale);
        }
    }
    return output;
}
struct Buffers {
    Stream stream;
    core::DeviceArena arena{size_t(total * hidden + 2 * guard) * 4 + size_t(total * dim + 2 * guard) * 6};
    float* x_storage = static_cast<float*>(arena.alloc(size_t(total * hidden + 2 * guard) * 4));
    float* latent_storage = static_cast<float*>(arena.alloc(size_t(total * dim + 2 * guard) * 4));
    uint16_t* output_storage = static_cast<uint16_t*>(arena.alloc(size_t(total * dim + 2 * guard) * 2));
    float* x = x_storage + guard;
    float* latent = latent_storage + guard;
    uint16_t* output = output_storage + guard;
    void clear() {
        upload(x_storage, std::vector<float>(total * hidden + 2 * guard, -333.f), stream.value);
        clear_latent();
        upload(output_storage, std::vector<uint16_t>(total * dim + 2 * guard, 0x5a5a), stream.value);
    }
    void clear_latent() { upload(latent_storage, std::vector<float>(total * dim + 2 * guard, -123.f), stream.value); }
    void latent_guards(int rows) {
        check(same(download(latent_storage, guard, stream.value), std::vector<float>(guard, -123.f)) &&
              same(download(latent + rows * dim, size_t(total - rows) * dim + guard, stream.value),
                   std::vector<float>(size_t(total - rows) * dim + guard, -123.f)), "latent guards and unused tail preserved");
    }
    void output_guards(int rows) {
        check(same(download(output_storage, guard, stream.value), std::vector<uint16_t>(guard, 0x5a5a)) &&
              same(download(output + rows * dim, size_t(total - rows) * dim + guard, stream.value),
                   std::vector<uint16_t>(size_t(total - rows) * dim + guard, 0x5a5a)), "postprocess output guards and unused tail preserved");
    }
};
Source20Batch batch(int start, int count, int capacity = total) { return Source20Batch(start, count, start, capacity, capacity); }
void synthetic() {
    Buffers b; Stream other;
    const float values[] = {0.f, -0.f, .25f, -.25f, .75f, -.75f, 1.25f, -1.25f,
                            1.75f, -1.75f, 2.5f, -2.5f, 3.5f, -3.5f, 5.f, 6.f};
    const float scales[] = {1.f, 1.0625f, 1.1875f, 1.f / 512, 128.f, 1.f / 4096};
    std::vector<float> input(total * dim, 0.f);
    for (int row = 0; row < total; ++row)
        for (int column = 0; column < 448; ++column)
            input[size_t(row) * dim + column] = values[column % 16] * scales[(row + column / 16) % 6];
    for (const auto& item : {std::pair{0, 17}, std::pair{1, 3}, std::pair{7, 3},
                             std::pair{65535, 3}, std::pair{1048573, 3}, std::pair{1048575, 1}}) {
        b.clear();
        const std::vector<float> part(input.begin(), input.begin() + item.second * dim);
        upload(b.latent, part, b.stream.value);
        postprocess_source20_kv(b.latent, b.output, item.second, item.first, b.stream.value);
        exact(download(b.output, part.size(), b.stream.value), reference(part, item.first),
              "scalar FP4/E4M3 boundary including odd position " + std::to_string(item.first));
        check(same(download(b.latent, part.size(), b.stream.value), part), "pre-RoPE latent preserved for indexer");
        b.latent_guards(item.second); b.output_guards(item.second);
    }
    const auto saved = download(b.output_storage, total * dim + 2 * guard, b.stream.value);
    for (const auto& item : {std::pair{-1, 1}, std::pair{1048576, 1}, std::pair{1048575, 2},
                             std::pair{0, 0}, std::pair{0, -1}, std::pair{0, 1048577},
                             std::pair{0, std::numeric_limits<int>::max()}}) {
        rejects("invalid position/count rejected before launch", [&] {
            postprocess_source20_kv(b.latent, b.output, item.second, item.first, b.stream.value);
        });
    }
    rejects("null latent", [&] { postprocess_source20_kv(nullptr, b.output, 1, 0, b.stream.value); });
    rejects("null output", [&] { postprocess_source20_kv(b.latent, nullptr, 1, 0, b.stream.value); });
    rejects("null stream", [&] { postprocess_source20_kv(b.latent, b.output, 1, 0, nullptr); });
    rejects("unaligned latent", [&] { postprocess_source20_kv(reinterpret_cast<float*>(reinterpret_cast<char*>(b.latent) + 1), b.output, 1, 0, b.stream.value); });
    rejects("unaligned output", [&] { postprocess_source20_kv(b.latent, reinterpret_cast<uint16_t*>(reinterpret_cast<char*>(b.output) + 1), 1, 0, b.stream.value); });
    rejects("input address overflow", [&] { postprocess_source20_kv(reinterpret_cast<const float*>(UINTPTR_MAX - 3), b.output, 1, 0, b.stream.value); });
    rejects("output address overflow", [&] { postprocess_source20_kv(b.latent, reinterpret_cast<uint16_t*>(UINTPTR_MAX - 1), 1, 0, b.stream.value); });
    rejects("same-buffer overlap", [&] { postprocess_source20_kv(b.latent, reinterpret_cast<uint16_t*>(b.latent), 1, 0, b.stream.value); });
    rejects("partial overlap", [&] { postprocess_source20_kv(b.latent, reinterpret_cast<uint16_t*>(b.latent + 1), 1, 0, b.stream.value); });
    rejects("reverse overlap", [&] { postprocess_source20_kv(b.latent + 1, reinterpret_cast<uint16_t*>(b.latent), 1, 0, b.stream.value); });
    check(same(saved, download(b.output_storage, total * dim + 2 * guard, b.stream.value)), "rejected raw transforms leave output untouched");

    for (int capacity : {0, -1, 1048577, std::numeric_limits<int>::max()})
        rejects("invalid cache capacity", [&] { CudaSource20KvCache bad(capacity); });
    for (int capacity : {1, 5, 17, 33}) {
        CudaSource20KvCache sized(capacity);
        check(sized.capacity() == capacity && sized.device_bytes() == size_t(capacity) * 1024,
              "cache owns exactly 1024 device bytes per token");
    }
    CudaSource20KvCache cache(total);
    const auto* allocation = cache.data();
    check(cache.size() == 0 && download(cache.data(), total * dim, b.stream.value) == std::vector<uint16_t>(total * dim, 0),
          "startup zeros all cache slots before nonblocking reads");
    b.clear(); upload(b.latent, input, b.stream.value);
    rejects("oversized first append refused without stream binding", [&] { cache.append(b.latent, batch(0, 18, 18), other.value); });
    check(cache.size() == 0 && download(cache.data(), total * dim, b.stream.value) == std::vector<uint16_t>(total * dim, 0),
          "capacity rejection writes no prefix");
    cache.append(b.latent, batch(0, 7), b.stream.value);
    const auto expected = reference(input);
    exact(download(cache.data(), 7 * dim, b.stream.value), {expected.begin(), expected.begin() + 7 * dim}, "odd prefill appends all seven tokens");
    const auto prefix = download(cache.data(), 7 * dim, b.stream.value);
    const auto state = download(cache.data(), total * dim, b.stream.value);
    rejects("gap in append", [&] { cache.append(b.latent, batch(8, 1), b.stream.value); });
    rejects("duplicate append", [&] { cache.append(b.latent, batch(6, 1), b.stream.value); });
    rejects("null append latent", [&] { cache.append(nullptr, batch(7, 1), b.stream.value); });
    rejects("unaligned append latent", [&] { cache.append(reinterpret_cast<float*>(reinterpret_cast<char*>(b.latent) + 1), batch(7, 1), b.stream.value); });
    rejects("append input address overflow", [&] { cache.append(reinterpret_cast<const float*>(UINTPTR_MAX - 3), batch(7, 1), b.stream.value); });
    rejects("owned cache cannot be latent", [&] { cache.append(reinterpret_cast<const float*>(cache.data()), batch(7, 1), b.stream.value); });
    rejects("null append stream", [&] { cache.append(b.latent, batch(7, 1), nullptr); });
    rejects("stream switch", [&] { cache.append(b.latent, batch(7, 1), other.value); });
    rejects("reset stream switch", [&] { cache.reset(other.value); });
    check(cache.size() == 7 && same(state, download(cache.data(), total * dim, b.stream.value)),
          "rejected appends preserve history and unused storage");
    for (int row = 7; row < total; ++row) cache.append(b.latent + row * dim, batch(row, 1), b.stream.value);
    exact(download(cache.data(), total * dim, b.stream.value), expected, "single-token decode fills exact capacity");
    check(same(prefix, download(cache.data(), 7 * dim, b.stream.value)), "decode does not rewrite earlier history");
    rejects("full cache never evicts", [&] { cache.append(b.latent, batch(17, 1, 18), b.stream.value); });
    check(cache.size() == total && same(expected, download(cache.data(), total * dim, b.stream.value)), "full-cache rejection preserves size and values");
    cache.reset(b.stream.value);
    check(cache.size() == 0 && cache.data() == allocation && cache.device_bytes() == total * 1024 &&
          download(cache.data(), total * dim, b.stream.value) == std::vector<uint16_t>(total * dim, 0),
          "reset clears storage while preserving allocation");
    rejects("stale descriptor after reset", [&] { cache.append(b.latent, batch(16, 1), b.stream.value); });
    for (int row = 0; row < total; ++row) cache.append(b.latent + row * dim, batch(row, 1), b.stream.value);
    exact(download(cache.data(), total * dim, b.stream.value), expected, "one-token replay matches prefill/decode partition");
    cache.reset(b.stream.value); cache.append(b.latent, batch(0, total), b.stream.value);
    exact(download(cache.data(), total * dim, b.stream.value), expected, "whole-prefill replay matches incremental cache");
    check(same(download(b.latent, input.size(), b.stream.value), input), "all cache appends preserve unrotated input");
    cache.reset(b.stream.value);
    const std::vector<float> zero(dim, 0.f); upload(b.latent, zero, b.stream.value);
    cache.append(b.latent, batch(0, 1), b.stream.value);
    exact(download(cache.data(), dim, b.stream.value), reference(zero), "reset plus zero input cannot retain old KV");
    check(download(cache.data() + dim, (total - 1) * dim, b.stream.value) == std::vector<uint16_t>((total - 1) * dim, 0),
          "unused slots stay zero");
    std::printf("source20_cache_capacity=%d source20_cache_device_bytes=%zu\n", cache.capacity(), cache.device_bytes());
}
int number(const GgufModel& model, const std::string& name, int maximum) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::U32 || value->u > uint64_t(maximum))
        throw std::runtime_error("Malformed fixture integer " + name);
    return int(value->u);
}
std::string string_value(const GgufModel& model, const std::string& name) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::STRING) throw std::runtime_error("Missing fixture string " + name);
    return value->s;
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, int rows, int width) {
    size_t shard = 0; const auto* value = model.find(name, &shard);
    if (!value || value->type != (sizeof(T) == 2 ? 30u : 0u) ||
        value->shape != std::vector<uint64_t>{uint64_t(width), uint64_t(rows), 1} || !model.in_bounds(*value, shard))
        throw std::runtime_error("Malformed reference tensor " + name);
    const auto* data = reinterpret_cast<const T*>(model.shard(shard).tensor_data(*value));
    return {data, data + size_t(rows) * width};
}
void real(const char* fixture_path, const char* weights_path) {
    const auto model = GgufModel::open(fixture_path), weights = GgufModel::open(weights_path);
    if (string_value(model, "general.architecture") != "deepseek41-source20-kv-fixture" ||
        string_value(model, "strata.deepseek41.revision") != "1bae0e1088993b9b82de08f59de5be5a34e42170" ||
        string_value(model, "strata.deepseek41.official_source_sha256") != "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d" ||
        string_value(model, "strata.deepseek41.official_kernel_sha256") != "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455" ||
        string_value(weights, "strata.deepseek41.revision") != string_value(model, "strata.deepseek41.revision") ||
        string_value(weights, "strata.deepseek41.source_hashes") != string_value(model, "strata.deepseek41.source_hashes"))
        throw std::runtime_error("Wrong source-20 reference/checkpoint identity");
    Buffers b;
    const int cases = number(model, "fixture.cases", 32);
    if (cases != 19) throw std::runtime_error("Expected 19 fixed postprocess cases");
    for (int index = 0; index < cases; ++index) {
        const std::string prefix = "case." + std::to_string(index) + ".";
        const int position = number(model, prefix + "position", 1048575), tokens = number(model, prefix + "tokens", 3);
        if (tokens < 1) throw std::runtime_error("Empty reference case");
        const auto input = tensor<float>(model, prefix + "input", tokens, dim);
        b.clear(); upload(b.latent, input, b.stream.value);
        postprocess_source20_kv(b.latent, b.output, tokens, position, b.stream.value);
        exact(download(b.output, input.size(), b.stream.value), tensor<uint16_t>(model, prefix + "output", tokens, dim),
              prefix + "official RoPE and independent FP4 at position " + std::to_string(position));
        check(same(input, download(b.latent, input.size(), b.stream.value)), prefix + "pre-RoPE input preserved");
        b.latent_guards(tokens); b.output_guards(tokens);
    }
    Geometry g; g.layers = 40; g.hidden = hidden; g.compress_ratios.resize(40); g.compress_ratios[20] = 1;
    g.sources.resize(40); g.sources[20] = {20, 20, -1};
    CudaSource20Compressor compressor(read_source20_compressor_weights(weights, g), total, total);
    CudaSource20KvCache native_cache(total), oracle_cache(total);
    const int plans = number(model, "fixture.plans", 16);
    if (plans != 9) throw std::runtime_error("Expected nine source-20 sequence plans");
    int calls = 0; std::vector<uint16_t> baseline;
    for (int plan = 0; plan < plans; ++plan) {
        const std::string prefix = "plan." + std::to_string(plan) + ".";
        const int tokens = number(model, prefix + "tokens", total), first = number(model, prefix + "prefill", total);
        if (tokens < 1 || first < 1 || first > tokens) throw std::runtime_error("Invalid source-20 prefill");
        const auto input = tensor<float>(model, prefix + "input", tokens, hidden);
        b.clear(); upload(b.x, input, b.stream.value);
        compressor.reset(b.stream.value); native_cache.reset(b.stream.value); oracle_cache.reset(b.stream.value);
        for (int start = 0; start < tokens;) {
            const int count = start == 0 ? first : 1;
            const std::string call = prefix + "at." + std::to_string(start) + ".";
            // The later owner must do this preflight BEFORE advancing its producer.
            if (count > native_cache.capacity() - native_cache.size()) throw std::runtime_error("Test cache capacity exhausted");
            b.clear_latent();
            const auto emitted = compressor.forward(b.x + size_t(start) * hidden, b.latent, count, start, b.stream.value);
            const auto before = download(b.latent, size_t(count) * dim, b.stream.value);
            native_cache.append(b.latent, emitted, b.stream.value);
            check(native_cache.size() == start + count && compressor.next_position() == start + count,
                  call + "compressor and cache positions agree");
            const auto expected = tensor<uint16_t>(model, call + "cache", start + count, dim);
            exact(download(native_cache.data(), expected.size(), b.stream.value), expected, call + "native compressor/cache versus official graph");
            check(same(before, download(b.latent, before.size(), b.stream.value)), call + "native latent remains available for index keys");
            b.latent_guards(count);
            const auto original_latent = tensor<float>(model, call + "latent", count, dim);
            upload(b.latent, original_latent, b.stream.value);
            oracle_cache.append(b.latent, emitted, b.stream.value);
            exact(download(oracle_cache.data(), expected.size(), b.stream.value), expected, call + "isolated postprocess/cache with official latent");
            check(same(original_latent, download(b.latent, original_latent.size(), b.stream.value)), call + "official latent preserved");
            const size_t unused = size_t(total - start - count) * dim;
            check(download(native_cache.data() + expected.size(), unused, b.stream.value) == std::vector<uint16_t>(unused, 0) &&
                  download(oracle_cache.data() + expected.size(), unused, b.stream.value) == std::vector<uint16_t>(unused, 0),
                  call + "unused cache slots stay zero");
            ++calls; start += count;
        }
        const auto actual = download(native_cache.data(), size_t(tokens) * dim, b.stream.value);
        if (plan == 0) baseline = actual;
        else if (plan < 6) check(same(actual, baseline), prefix + "native cache partition/reset byte parity");
        check(same(input, download(b.x, input.size(), b.stream.value)), prefix + "original decoder input unchanged");
        check(same(download(b.x_storage, guard, b.stream.value), std::vector<float>(guard, -333.f)) &&
              same(download(b.x + total * hidden, guard, b.stream.value), std::vector<float>(guard, -333.f)), prefix + "decoder input guards preserved");
        check(native_cache.device_bytes() == total * 1024 && oracle_cache.device_bytes() == total * 1024,
              prefix + "fixed cache allocations");
    }
    check(calls == 58, "all 58 fixed sequence calls consumed");
    std::printf("source20_cache_capacity=%d source20_cache_device_bytes=%zu compressor_plus_one_cache_bytes=%zu\n",
                native_cache.capacity(), native_cache.device_bytes(), compressor.device_bytes() + native_cache.device_bytes());
}
} // namespace

int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 4 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 4) real(argv[2], argv[3]); else synthetic();
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); ++failures; }
    std::printf("compared_bf16_values=%zu different_bf16_values=%zu\n", compared, mismatched);
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
