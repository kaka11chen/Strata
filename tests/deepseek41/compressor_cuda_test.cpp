#include "strata/deepseek41/compressor_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int hidden = 5120, dim = 512, total = 13;
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
float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
void upload(float* dst, const std::vector<float>& src, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, src.data(), src.size() * 4, cudaMemcpyHostToDevice, stream));
    ck(cudaStreamSynchronize(stream));
}
std::vector<float> download(const float* src, size_t n, cudaStream_t stream) {
    std::vector<float> out(n);
    ck(cudaMemcpyAsync(out.data(), src, n * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return out;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name, bool projection = false) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double error = 0., norm = 0., peak = 0., maximum = 0.; bool finite = true, rounded = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i]; error += delta * delta; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]); rounded &= actual[i] == bf(actual[i]);
    }
    const double rel = std::sqrt(error / std::max(norm, 1e-60));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), rel, maximum, peak);
    check(finite && (projection || rounded) && rel <= (projection ? 1e-5 : .003) &&
          maximum <= (projection ? 2e-6 + 2e-5 * peak : 1e-5 + .02 * peak), name);
}
// Stateless scalar CPU oracle: full row-major dot products accumulated in double,
// then independent two-token softmax and RMSNorm over the complete history.
std::vector<float> project(const std::vector<float>& input, std::span<const uint16_t> weights) {
    std::vector<float> result(total * dim);
    for (int t = 0; t < total; ++t)
        for (int d = 0; d < dim; ++d) {
            double sum = 0.;
            for (int k = 0; k < hidden; ++k) {
                const float w = kernels::f32_from_bf16(weights[size_t(d) * hidden + k]);
                if (w != 0) sum += double(w) * bf(input[size_t(t) * hidden + k]);
            }
            result[size_t(t) * dim + d] = float(sum);
        }
    return result;
}
std::vector<float> reference(const std::vector<float>& kv, const std::vector<float>& scores,
                             std::span<const uint16_t> norm, bool round_projection = false) {
    std::vector<float> output((total / 2) * dim);
    for (int group = 0; group < total / 2; ++group) {
        double square = 0.;
        for (int d = 0; d < dim; ++d) {
            const size_t a = size_t(2 * group) * dim + d, b = a + dim;
            const double maximum = std::max(scores[a], scores[b]);
            const double ea = std::exp(double(scores[a]) - maximum), eb = std::exp(double(scores[b]) - maximum);
            const float x = round_projection ? bf(kv[a]) : kv[a], y = round_projection ? bf(kv[b]) : kv[b];
            const float pooled = bf(x * float(ea / (ea + eb)) + y * float(eb / (ea + eb)));
            output[size_t(group) * dim + d] = pooled; square += double(pooled) * pooled;
        }
        const float scale = float(1. / std::sqrt(square / dim + 1e-20));
        for (int d = 0; d < dim; ++d) {
            auto& value = output[size_t(group) * dim + d];
            value = bf((value * scale) * kernels::f32_from_bf16(norm[size_t(d)]));
        }
    }
    return output;
}
void synthetic() {
    std::vector<uint16_t> kv(size_t(hidden) * dim, 0), gate(kv.size(), 0), norm(dim);
    const float gates[] = {0.f, 1.f, 8.f, -8.f, 128.f, -128.f, .25f, -.25f};
    for (int d = 0; d < dim; ++d) {
        kv[size_t(d) * hidden + d] = kernels::bf16_from_f32(1.0078125f);
        kv[size_t(d) * hidden + dim + d] = kernels::bf16_from_f32(.33203125f);
        kv[size_t(d) * hidden + 2 * dim + d] = kernels::bf16_from_f32(-.19140625f);
        gate[size_t(d) * hidden + 3 * dim + d] = kernels::bf16_from_f32(gates[d % 8]);
        norm[size_t(d)] = kernels::bf16_from_f32(.5f + float(d % 7) / 8.f);
    }
    std::mt19937 gen(1017); std::uniform_real_distribution<float> uniform(-3.f, 3.f);
    std::vector<float> input(total * hidden);
    for (float& v : input) v = uniform(gen); // deliberately not already BF16
    const auto projected = project(input, kv), scores = project(input, gate);
    const auto expected = reference(projected, scores, norm);
    check(!same(expected, reference(projected, scores, norm, true)), "oracle detects premature BF16 rounding of projected KV");
    check(*std::max_element(scores.begin(), scores.end()) > 100.f, "oracle exercises stable softmax beyond exp overflow");
    Stream stream, other;
    CudaCompressor compressor({kv, gate, norm});
    core::DeviceArena arena((input.size() + 7 * dim) * 4 + 256);
    auto* in = static_cast<float*>(arena.alloc(input.size() * 4));
    auto* out = static_cast<float*>(arena.alloc(7 * dim * 4));
    upload(in, input, stream.value);
    rejects("null stream refused", [&] { compressor.forward(in, out, 2, 0, nullptr); });
    rejects("null input refused", [&] { compressor.forward(nullptr, out, 1, 0, stream.value); });
    rejects("missing emitted output refused", [&] { compressor.forward(in, nullptr, 2, 0, stream.value); });
    rejects("overlapping input/output refused", [&] { compressor.forward(in, in, 2, 0, stream.value); });
    rejects("unaligned input refused", [&] { compressor.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(in) + 1), out, 2, 0, stream.value); });
    rejects("missing prefill refused", [&] { compressor.forward(in, out, 1, 1, stream.value); });
    rejects("negative position refused", [&] { compressor.forward(in, out, 1, -1, stream.value); });
    rejects("zero tokens refused", [&] { compressor.forward(in, out, 0, 0, stream.value); });
    rejects("sequence bound checked before buffer access", [&] { compressor.forward(in, out, 1048577, 0, stream.value); });
    const auto bytes = compressor.device_bytes();
    std::vector<float> baseline;
    for (int first : {13, 1, 2, 5, 6}) {
        compressor.reset(stream.value);
        check(compressor.next_position() == 0 && !compressor.pending_kv() && !compressor.pending_scores(), "reset discards previous pending tail");
        upload(out, std::vector<float>(7 * dim, -123.f), stream.value);
        int position = 0, emitted = 0;
        while (position < total) {
            const int length = position == 0 ? first : 1;
            const int groups = (position + length) / 2 - position / 2;
            // Exercise null output while no group exists, and an untouched sentinel otherwise.
            auto* destination = groups == 0 && position % 4 == 0 ? nullptr : out + size_t(emitted) * dim;
            const auto info = compressor.forward(in + size_t(position) * hidden, destination, length, position, stream.value);
            check(info.count == groups && info.first_position == (groups ? position / 2 * 2 : -1), "emission count and first-token positions");
            position += length; emitted += groups;
            check(compressor.next_position() == position && bool(compressor.pending_kv()) == bool(position % 2) &&
                  bool(compressor.pending_scores()) == bool(position % 2), "position and active tail agree");
            if (!groups) check(same(download(out + size_t(emitted) * dim, dim, stream.value), std::vector<float>(dim, -123.f)), "incomplete pair leaves output untouched");
        }
        const auto actual = download(out, (total / 2) * dim, stream.value);
        compare(actual, expected, "synthetic prefill=" + std::to_string(first));
        if (first == 13) baseline = actual;
        else check(same(actual, baseline), "prefill/decode partition leaves output byte-identical");
        compare(download(compressor.pending_kv(), dim, stream.value), {projected.end() - dim, projected.end()}, "pending KV stays F32", true);
        compare(download(compressor.pending_scores(), dim, stream.value), {scores.end() - dim, scores.end()}, "pending per-channel scores stay F32", true);
        check(bytes == compressor.device_bytes(), "owned memory does not grow with the sequence");
    }
    const auto tail = download(compressor.pending_kv(), dim, stream.value);
    rejects("gap in decode refused", [&] { compressor.forward(in, out, 1, total + 1, stream.value); });
    rejects("second prefill refused", [&] { compressor.forward(in, out, 2, 0, stream.value); });
    rejects("multi-token continuation refused", [&] { compressor.forward(in, out, 2, total, stream.value); });
    rejects("stream switch refused", [&] { compressor.forward(in, out, 1, total, other.value); });
    rejects("reset stream switch refused", [&] { compressor.reset(other.value); });
    rejects("owned state cannot be an input", [&] { compressor.forward(compressor.pending_kv(), out, 1, total, stream.value); });
    rejects("owned state cannot be an output", [&] { compressor.forward(in, const_cast<float*>(compressor.pending_kv()), 1, total, stream.value); });
    check(compressor.next_position() == total && same(tail, download(compressor.pending_kv(), dim, stream.value)), "rejected calls preserve the active tail");
    compressor.reset(stream.value);
    upload(in, std::vector<float>(2 * hidden, 0.f), stream.value);
    const auto zero = compressor.forward(in, out, 2, 0, stream.value);
    check(zero.count == 1 && zero.first_position == 0 && same(download(out, dim, stream.value), std::vector<float>(dim, 0.f)), "reset cannot leak an old pending token into a zero pair");
    rejects("wrong weight size refused", [&] { CudaCompressor bad({std::span(kv).first(512), gate, norm}); });
    for (auto* weights : {&kv, &gate, &norm}) {
        const auto saved = (*weights)[0]; (*weights)[0] = 0x7f80;
        rejects("nonfinite checkpoint weight refused", [&] { CudaCompressor bad({kv, gate, norm}); });
        (*weights)[0] = saved;
    }
    std::printf("compressor_device_bytes=%zu weight_bytes=%zu pending_and_scratch_bytes=%zu\n", bytes,
                size_t(2 * hidden * dim + dim) * 4, bytes - size_t(2 * hidden * dim + dim) * 4);
}
std::vector<float> tensor(const GgufModel& model, const std::string& name, size_t count) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != 0 || tensor_payload_bytes(*t) != count * 4 || !model.in_bounds(*t, shard))
        throw std::runtime_error("Malformed compressor fixture tensor " + name);
    const auto* data = reinterpret_cast<const float*>(model.shard(shard).tensor_data(*t)); return {data, data + count};
}
int number(const GgufModel& model, const std::string& name) {
    const auto* m = model.meta().get(name);
    if (!m || m->type != MetaType::U32 || m->u == 0 || m->u > total) throw std::runtime_error("Malformed fixture number " + name);
    return int(m->u);
}
void real(const char* path) {
    const auto model = GgufModel::open(path);
    Geometry g; g.layers = 40; g.hidden = hidden; g.compress_ratios.resize(40); g.compress_ratios[2] = 2;
    CudaCompressor compressor(read_compressor_weights(model, g, 2));
    Stream stream;
    core::DeviceArena arena((total * hidden + 7 * dim) * 4 + 256);
    auto* in = static_cast<float*>(arena.alloc(total * hidden * 4));
    auto* out = static_cast<float*>(arena.alloc(7 * dim * 4));
    const auto bytes = compressor.device_bytes();
    const int plans = number(model, "fixture.plans");
    if (plans != 8) throw std::runtime_error("Expected eight fixed compressor reference plans");
    std::vector<float> baseline;
    for (int plan = 0; plan < plans; ++plan) {
        const std::string prefix = "plan." + std::to_string(plan) + ".";
        const int tokens = number(model, prefix + "tokens"), first = number(model, prefix + "prefill");
        if (first > tokens) throw std::runtime_error("Invalid reference prefill length");
        const auto input = tensor(model, prefix + "input", size_t(tokens) * hidden);
        upload(in, input, stream.value); upload(out, std::vector<float>(7 * dim, -123.f), stream.value);
        compressor.reset(stream.value);
        int position = 0, emitted = 0;
        while (position < tokens) {
            const int length = position == 0 ? first : 1;
            const int groups = (position + length) / 2 - position / 2;
            const std::string call = prefix + "at." + std::to_string(position) + ".";
            const auto info = compressor.forward(in + size_t(position) * hidden, out + size_t(emitted) * dim, length, position, stream.value);
            check(info.count == groups && info.first_position == (groups ? position / 2 * 2 : -1), call + "positions");
            if (groups) compare(download(out + size_t(emitted) * dim, size_t(groups) * dim, stream.value), tensor(model, call + "output", size_t(groups) * dim), call + "official output");
            else check(same(download(out + size_t(emitted) * dim, dim, stream.value), std::vector<float>(dim, -123.f)), call + "no early emission");
            position += length; emitted += groups;
            if (position % 2) {
                compare(download(compressor.pending_kv(), dim, stream.value), tensor(model, call + "kv", dim), call + "official pending KV", true);
                compare(download(compressor.pending_scores(), dim, stream.value), tensor(model, call + "scores", dim), call + "official pending scores", true);
            } else check(!compressor.pending_kv() && !compressor.pending_scores(), call + "no active tail");
        }
        check(compressor.next_position() == tokens && compressor.device_bytes() == bytes, prefix + "bounded state");
        if (plan == 0) baseline = download(out, (total / 2) * dim, stream.value);
        else if (plan < 5) check(same(baseline, download(out, (total / 2) * dim, stream.value)), prefix + "reset/partition byte parity");
    }
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
