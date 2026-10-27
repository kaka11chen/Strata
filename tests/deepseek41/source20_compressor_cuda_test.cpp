#include "strata/deepseek41/source20_compressor_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int hidden = 5120, dim = 512, total = 17, guard = 64;
int checks = 0, failures = 0;
struct Metrics {
    double rel_l2 = 0., max_abs = 0.;
    size_t values = 0, different = 0;
} projection_metrics, output_metrics;
void check(bool ok, const std::string& name) {
    ++checks; failures += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str());
}
void ck(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void rejects(const std::string& name, const std::function<void()>& fn) {
    try { fn(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(value); }
};
void upload(float* pointer, const std::vector<float>& values, cudaStream_t stream) {
    ck(cudaMemcpyAsync(pointer, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
}
std::vector<float> download(const float* pointer, size_t count, cudaStream_t stream) {
    std::vector<float> values(count);
    ck(cudaMemcpyAsync(values.data(), pointer, count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream));
    return values;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected,
             const std::string& name, bool projection = false) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double error = 0., norm = 0., maximum = 0., peak = 0.;
    bool finite = true, rounded = true;
    size_t different = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i];
        error += delta * delta; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
        rounded &= actual[i] == bf(actual[i]);
        different += actual[i] != expected[i];
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-60));
    auto& metric = projection ? projection_metrics : output_metrics;
    metric.rel_l2 = std::max(metric.rel_l2, relative); metric.max_abs = std::max(metric.max_abs, maximum);
    metric.values += actual.size(); metric.different += different;
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g differing=%zu/%zu\n",
                name.c_str(), relative, maximum, peak, different, actual.size());
    // Declared before running: BF16 GEMV and CPU PyTorch use different F32
    // reduction orders. Retain BF16 boundaries, but do not claim bitwise parity.
    check(finite && rounded && relative <= (projection ? .001 : .003) &&
          maximum <= (projection ? 1e-6 + .01 * peak : 1e-5 + .02 * peak), name);
}
struct Buffers {
    Stream stream;
    core::DeviceArena arena{size_t(total * (hidden + dim) + 4 * guard) * sizeof(float)};
    float* input_storage = static_cast<float*>(arena.alloc(size_t(total * hidden + 2 * guard) * sizeof(float)));
    float* output_storage = static_cast<float*>(arena.alloc(size_t(total * dim + 2 * guard) * sizeof(float)));
    float* input = input_storage + guard;
    float* output = output_storage + guard;
    void prepare(const std::vector<float>& values) {
        upload(input_storage, std::vector<float>(total * hidden + 2 * guard, -333.f), stream.value);
        upload(output_storage, std::vector<float>(total * dim + 2 * guard, -123.f), stream.value);
        upload(input, values, stream.value);
    }
    void preserved(const std::vector<float>& values, int emitted) {
        check(same(download(input, values.size(), stream.value), values), "caller input preserved byte-for-byte");
        auto input_before = download(input_storage, guard, stream.value);
        auto input_after = download(input + total * hidden, guard, stream.value);
        auto output_before = download(output_storage, guard, stream.value);
        auto output_after = download(output + size_t(emitted) * dim, size_t(total - emitted) * dim + guard, stream.value);
        check(same(input_before, std::vector<float>(guard, -333.f)) &&
              same(input_after, std::vector<float>(guard, -333.f)) &&
              same(output_before, std::vector<float>(guard, -123.f)) &&
              same(output_after, std::vector<float>(output_after.size(), -123.f)), "input/output guards and unused output tail preserved");
    }
};
// Independent scalar oracle: exact BF16 products accumulated in double, then
// F32-to-BF16 rounding before RMSNorm. No Strata GPU matrix/norm implementation.
std::vector<float> project(const std::vector<float>& input, std::span<const uint16_t> weights) {
    const size_t tokens = input.size() / hidden;
    std::vector<float> output(tokens * dim);
    for (size_t token = 0; token < tokens; ++token)
        for (int row = 0; row < dim; ++row) {
            double sum = 0.;
            for (int k = 0; k < hidden; ++k) {
                const float weight = kernels::f32_from_bf16(weights[size_t(row) * hidden + k]);
                if (weight != 0.f) sum += double(weight) * bf(input[token * hidden + k]);
            }
            output[token * dim + row] = bf(float(sum));
        }
    return output;
}
std::vector<float> normalize(const std::vector<float>& projected, std::span<const uint16_t> gamma) {
    std::vector<float> output(projected.size());
    for (size_t start = 0; start < projected.size(); start += dim) {
        double squares = 0.;
        for (int row = 0; row < dim; ++row) squares += double(projected[start + row]) * projected[start + row];
        const float scale = float(1. / std::sqrt(squares / dim + 1e-20));
        for (int row = 0; row < dim; ++row)
            output[start + row] = bf((projected[start + row] * scale) * kernels::f32_from_bf16(gamma[size_t(row)]));
    }
    return output;
}
void synthetic() {
    std::vector<uint16_t> weights(size_t(hidden) * dim, 0), norm(dim);
    for (int row = 0; row < dim; ++row) {
        weights[size_t(row) * hidden + row] = kernels::bf16_from_f32(1.0078125f);
        weights[size_t(row) * hidden + dim + row] = kernels::bf16_from_f32(.33203125f);
        weights[size_t(row) * hidden + 2 * dim + row] = kernels::bf16_from_f32(-.203125f);
        norm[size_t(row)] = kernels::bf16_from_f32(.5f + float(row % 19) / 32);
    }
    std::fill(weights.begin(), weights.begin() + 6 * hidden, uint16_t(0));
    auto set = [&](int row, int column, float value) { weights[size_t(row) * hidden + column] = kernels::bf16_from_f32(value); };
    set(0, 0, 1.f); set(0, 1, 0x1p-8f);
    set(1, 0, 1.f); set(1, 1, 0x1p-8f); set(1, 2, 0x1p-16f);
    set(2, 0, 1.0078125f); set(2, 1, 0x1p-8f);
    set(3, 0, -1.f); set(3, 1, -0x1p-8f);
    set(4, 3, .75f); set(5, 4, 1.f);
    std::mt19937 generator(6121);
    std::uniform_real_distribution<float> random(-1.f, 1.f);
    std::vector<float> input(total * hidden);
    for (auto& value : input) value = random(generator);
    for (int token = 0; token < total; ++token) {
        float* row = input.data() + token * hidden;
        row[0] = row[1] = row[2] = 1.f; row[3] = 1.f + 0x1p-8f; row[4] = 1.f + 0x1p-8f + 0x1p-20f;
    }
    const auto projected = project(input, weights), expected = normalize(projected, norm);
    CudaSource20Compressor compressor({weights, norm}, total, total);
    Buffers b;
    check(compressor.next_position() == 0 && !compressor.last_projection(), "fresh source-20 has no pending projection");
    const size_t bytes = compressor.device_bytes();
    check(bytes == size_t(hidden) * dim * 2 + dim * 4 + hidden * 2 + dim * 4,
          "BF16 weights and fixed scratch occupy the declared arena");
    std::vector<float> baseline;
    for (int first : {17, 1, 7, 8, 9}) {
        b.prepare(input); compressor.reset(b.stream.value);
        check(compressor.next_position() == 0 && !compressor.last_projection(), "reset invalidates projection view");
        for (int start = 0; start < total;) {
            const int count = start == 0 ? first : 1;
            const auto batch = compressor.forward(b.input + size_t(start) * hidden, b.output + size_t(start) * dim,
                                                   count, start, b.stream.value);
            check(batch.first_position() == start && batch.latent_count() == count &&
                  batch.key_count() == start + count && batch.position(count - 1) == start + count - 1 &&
                  compressor.next_position() == start + count, "every token emits a ratio-1 latent without odd-tail delay");
            const auto last = download(compressor.last_projection(), dim, b.stream.value);
            compare(last, {projected.begin() + (start + count - 1) * dim, projected.begin() + (start + count) * dim},
                    "scalar BF16 projection", true);
            check(std::equal(last.begin(), last.begin() + 6,
                             std::vector<float>{1.f, 1.0078125f, 1.015625f, -1.f, .75f, 1.0078125f}.begin()),
                  "BF16 ties-to-even, above-midpoint and input-rounding witnesses");
            start += count;
        }
        const auto actual = download(b.output, total * dim, b.stream.value);
        compare(actual, expected, "scalar pre-RoPE normalized output");
        if (baseline.empty()) baseline = actual;
        else check(same(actual, baseline), "prefill/decode partitions and resets are byte-identical");
        b.preserved(input, total);
        check(compressor.device_bytes() == bytes, "owned memory stays fixed across calls");
    }
    const auto before = download(b.output, total * dim, b.stream.value);
    const auto last = download(compressor.last_projection(), dim, b.stream.value);
    rejects("full context refused", [&] { compressor.forward(b.input, b.output, 1, total, b.stream.value); });
    rejects("gap refused", [&] { compressor.forward(b.input, b.output, 1, total + 1, b.stream.value); });
    check(compressor.next_position() == total && same(before, download(b.output, total * dim, b.stream.value)) &&
          same(last, download(compressor.last_projection(), dim, b.stream.value)), "capacity rejection preserves completed state and buffers");
    compressor.reset(b.stream.value);
    b.prepare(input);
    Stream other;
    rejects("empty call refused", [&] { compressor.forward(b.input, b.output, 0, 0, b.stream.value); });
    rejects("negative count refused", [&] { compressor.forward(b.input, b.output, -1, 0, b.stream.value); });
    rejects("negative position refused", [&] { compressor.forward(b.input, b.output, 1, -1, b.stream.value); });
    rejects("oversized count refused without arithmetic overflow", [&] {
        compressor.forward(b.input, b.output, std::numeric_limits<int>::max(), 0, b.stream.value);
    });
    rejects("null input refused", [&] { compressor.forward(nullptr, b.output, 1, 0, b.stream.value); });
    rejects("null output refused", [&] { compressor.forward(b.input, nullptr, 1, 0, b.stream.value); });
    rejects("null stream refused", [&] { compressor.forward(b.input, b.output, 1, 0, nullptr); });
    rejects("changed stream refused", [&] { compressor.forward(b.input, b.output, 1, 0, other.value); });
    rejects("reset cannot change stream", [&] { compressor.reset(other.value); });
    rejects("unaligned input refused", [&] {
        compressor.forward(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(b.input) + 1), b.output, 1, 0, b.stream.value);
    });
    rejects("unaligned output refused", [&] {
        compressor.forward(b.input, reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(b.output) + 1), 1, 0, b.stream.value);
    });
    rejects("pointer range overflow refused", [&] {
        compressor.forward(reinterpret_cast<const float*>(UINTPTR_MAX - 3), b.output, 1, 0, b.stream.value);
    });
    rejects("same buffer refused", [&] { compressor.forward(b.input, b.input, 1, 0, b.stream.value); });
    rejects("partial overlap refused", [&] { compressor.forward(b.input, b.input + 1, 1, 0, b.stream.value); });
    rejects("reverse overlap refused", [&] { compressor.forward(b.input + 1, b.input, 1, 0, b.stream.value); });
    check(compressor.next_position() == 0 && !compressor.last_projection() &&
          same(download(b.output, total * dim, b.stream.value), std::vector<float>(total * dim, -123.f)),
          "argument rejection leaves stream state and output untouched");
    compressor.forward(b.input, b.output, 1, 0, b.stream.value);
    rejects("stale start refused", [&] { compressor.forward(b.input, b.output, 1, 0, b.stream.value); });
    rejects("skipped start refused", [&] { compressor.forward(b.input, b.output, 1, 2, b.stream.value); });
    rejects("second prefill refused", [&] { compressor.forward(b.input, b.output, 2, 1, b.stream.value); });
    rejects("internal projection cannot be input", [&] { compressor.forward(compressor.last_projection(), b.output, 1, 1, b.stream.value); });
    rejects("internal projection cannot be output", [&] {
        compressor.forward(b.input, const_cast<float*>(compressor.last_projection()), 1, 1, b.stream.value);
    });
    check(compressor.next_position() == 1, "rejected decode does not advance position");
    compressor.forward(b.input + hidden, b.output + dim, 1, 1, b.stream.value);
    check(compressor.next_position() == 2, "valid decode still works after rejected arguments");
    compressor.reset(b.stream.value);
    std::vector<float> zero(2 * hidden, 0.f); b.prepare(zero);
    compressor.forward(b.input, b.output, 2, 0, b.stream.value);
    check(same(download(b.output, 2 * dim, b.stream.value), std::vector<float>(2 * dim, 0.f)) &&
          same(download(compressor.last_projection(), dim, b.stream.value), std::vector<float>(dim, 0.f)),
          "reset followed by zeros cannot leak an earlier projection");
    b.preserved(zero, 2);

    rejects("wrong projection size refused", [&] { CudaSource20Compressor bad({std::span(weights).first(512), norm}, total, total); });
    rejects("wrong norm size refused", [&] { CudaSource20Compressor bad({weights, std::span(norm).first(511)}, total, total); });
    for (auto* view : {&weights, &norm}) for (uint16_t bits : {uint16_t(0x7f80), uint16_t(0x7fc0)}) {
        const uint16_t saved = (*view)[0]; (*view)[0] = bits;
        rejects("nonfinite weight refused", [&] { CudaSource20Compressor bad({weights, norm}, total, total); });
        (*view)[0] = saved;
    }
    for (const auto& capacity : {std::pair{0, 1}, std::pair{1048577, 1}, std::pair{17, 0}, std::pair{17, 18}})
        rejects("invalid configured capacity refused", [&] { CudaSource20Compressor bad({weights, norm}, capacity.first, capacity.second); });
    {
        CudaSource20Compressor bounded({weights, norm}, 17, 7);
        rejects("prefill above configured cap refused", [&] { bounded.forward(b.input, b.output, 8, 0, other.value); });
        bounded.forward(b.input, b.output, 1, 0, b.stream.value);
        check(bounded.next_position() == 1, "rejected first call does not bind its stream");
    }
    {
        CudaSource20Compressor maximum({weights, norm}, Source20Batch::max_context, Source20Batch::max_context);
        check(maximum.device_bytes() == bytes, "larger context/prefill limits do not expand compressor scratch");
    }
    std::printf("source20_device_bytes=%zu projection_weight_bytes=%zu norm_and_scratch_bytes=%zu\n",
                bytes, size_t(hidden) * dim * 2, bytes - size_t(hidden) * dim * 2);
}
int number(const GgufModel& model, const std::string& name) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::U32 || value->u < 1 || value->u > total)
        throw std::runtime_error("Malformed fixture number " + name);
    return int(value->u);
}
void string_is(const GgufModel& model, const std::string& name, const std::string& expected) {
    const auto* value = model.meta().get(name);
    if (!value || value->type != MetaType::STRING || value->s != expected)
        throw std::runtime_error("Wrong fixture identity " + name);
}
std::vector<float> tensor(const GgufModel& model, const std::string& name, int tokens, int width) {
    size_t shard = 0; const auto* value = model.find(name, &shard);
    if (!value || value->type != 30 || value->shape != std::vector<uint64_t>{uint64_t(width), uint64_t(tokens), 1} ||
        !model.in_bounds(*value, shard)) throw std::runtime_error("Malformed BF16 reference tensor " + name);
    const auto* data = reinterpret_cast<const uint16_t*>(model.shard(shard).tensor_data(*value));
    std::vector<float> output(size_t(tokens) * width);
    for (size_t i = 0; i < output.size(); ++i) output[i] = kernels::f32_from_bf16(data[i]);
    return output;
}
void real(const char* path) {
    const auto model = GgufModel::open(path);
    string_is(model, "general.architecture", "deepseek41-source20-contract-fixture");
    string_is(model, "strata.deepseek41.revision", "1bae0e1088993b9b82de08f59de5be5a34e42170");
    string_is(model, "strata.deepseek41.official_source_sha256", "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d");
    Geometry g; g.layers = 40; g.hidden = hidden; g.compress_ratios.resize(40); g.compress_ratios[20] = 1;
    g.sources.resize(40); g.sources[20] = {20, 20, -1};
    CudaSource20Compressor compressor(read_source20_compressor_weights(model, g), total, total);
    Buffers b;
    const size_t bytes = compressor.device_bytes();
    const int plans = number(model, "fixture.plans");
    if (plans != 9) throw std::runtime_error("Expected nine pinned source-20 plans");
    std::vector<float> baseline;
    int calls = 0;
    for (int plan = 0; plan < plans; ++plan) {
        const std::string prefix = "plan." + std::to_string(plan) + ".";
        const int tokens = number(model, prefix + "tokens"), first = number(model, prefix + "prefill");
        if (first > tokens) throw std::runtime_error("Invalid reference prefill length");
        const auto input = tensor(model, prefix + "input", tokens, hidden);
        b.prepare(input); compressor.reset(b.stream.value);
        check(compressor.next_position() == 0 && !compressor.last_projection(), prefix + "reset");
        for (int start = 0; start < tokens;) {
            const int count = start == 0 ? first : 1;
            const std::string call = prefix + "at." + std::to_string(start) + ".";
            const auto batch = compressor.forward(b.input + size_t(start) * hidden, b.output + size_t(start) * dim,
                                                   count, start, b.stream.value);
            check(batch.first_position() == start && batch.latent_count() == count &&
                  batch.key_count() == start + count && compressor.next_position() == start + count,
                  call + "positions");
            const auto projection = tensor(model, call + "projection", count, dim);
            compare(download(compressor.last_projection(), dim, b.stream.value),
                    {projection.end() - dim, projection.end()}, call + "official last-token projection", true);
            compare(download(b.output + size_t(start) * dim, size_t(count) * dim, b.stream.value),
                    tensor(model, call + "output", count, dim), call + "official pre-RoPE output");
            ++calls; start += count;
        }
        b.preserved(input, tokens);
        const auto actual = download(b.output, size_t(tokens) * dim, b.stream.value);
        if (plan == 0) baseline = actual;
        else if (plan < 6) check(same(actual, baseline), prefix + "native partition/reset byte parity");
        check(compressor.device_bytes() == bytes && compressor.next_position() == tokens, prefix + "bounded state");
    }
    check(calls == 58, "all 58 pinned reference calls executed");
    std::printf("real_source20_device_bytes=%zu\n", bytes);
}
void metrics() {
    for (auto [name, value] : {std::pair{"projection", projection_metrics}, std::pair{"output", output_metrics}})
        std::printf("metric=%s max_rel_l2=%.12g max_abs=%.12g differing=%zu values=%zu\n",
                    name, value.rel_l2, value.max_abs, value.different, value.values);
}
} // namespace

int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 3 && std::string(argv[1]) == "--real"))) return 2;
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 3) real(argv[2]); else synthetic();
    } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); ++failures; }
    metrics();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
