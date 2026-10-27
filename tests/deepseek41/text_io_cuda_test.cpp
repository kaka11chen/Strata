#include "strata/deepseek41/text_io_cuda.hpp"
#include "text_io_reference.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>

using namespace strata;
using namespace strata::deepseek41;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void rejects(const char* name, const std::function<void()>& f) {
    try { f(); check(false, name); } catch (const std::invalid_argument&) { check(true, name); }
}
float bf(float v) { return kernels::f32_from_bf16(kernels::bf16_from_f32(v)); }
void upload(float* out, const std::vector<float>& values, cudaStream_t stream) {
    ck(cudaMemcpyAsync(out, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream)); ck(cudaStreamSynchronize(stream));
}
std::vector<float> download(const float* in, size_t count, cudaStream_t stream) {
    std::vector<float> out(count); ck(cudaMemcpyAsync(out.data(), in, count * sizeof(float), cudaMemcpyDeviceToHost, stream)); ck(cudaStreamSynchronize(stream)); return out;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& label, double tolerance) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double error = 0., norm = 0., peak = 0., maximum = 0.; bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i]; error += delta * delta; norm += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]);
    }
    const double relative = std::sqrt(error / std::max(norm, 1e-30));
    std::printf("  %s relative_l2=%.9g max_abs=%.9g peak=%.9g\n", label.c_str(), relative, maximum, peak);
    check(finite && relative <= tolerance && maximum <= tolerance * (1. + peak), label);
}

struct Buffers {
    core::DeviceArena arena;
    float *residual, *pre, *logits;
    Buffers(int h, int v, int capacity) : arena((size_t(capacity) * (4 * h + 4) + size_t(v)) * sizeof(float) + 2048) {
        residual = static_cast<float*>(arena.alloc(size_t(capacity) * 4 * h * sizeof(float)));
        pre = static_cast<float*>(arena.alloc(size_t(capacity) * 4 * sizeof(float)));
        logits = static_cast<float*>(arena.alloc(size_t(v) * sizeof(float)));
    }
};
void analytical(cudaStream_t stream, cudaStream_t other) {
    constexpr int h = 64, v = 19;
    std::vector<uint16_t> embedding(h * v), head(h * v), norm(h);
    for (int i = 0; i < h * v; ++i) {
        embedding[size_t(i)] = kernels::bf16_from_f32(float((i * 17 + 3) % 67 - 33) / 32.f);
        head[size_t(i)] = kernels::bf16_from_f32(float((i * 11 + 7) % 43 - 21) / 64.f);
    }
    for (int d = 0; d < h; ++d) norm[size_t(d)] = kernels::bf16_from_f32(.5f + float(d) / 128.f);
    TextIoWeights w{h, v, embedding, norm, head}; CudaTextIo io(w, 4); Buffers b(h, v, 4);
    const size_t bytes = io.device_bytes(), staging = io.host_staging_bytes();
    check(io.hidden_size() == h && io.vocab_size() == v && io.prefill_capacity() == 4, "explicit bounded text IO geometry");
    for (const auto& tokens : {std::vector<int32_t>{0, v - 1, 0, 7}, std::vector<int32_t>{5}}) {
        io.embed(tokens, b.residual, b.pre, stream); const size_t n = tokens.size();
        std::vector<float> expected(n * 4 * h), initial(n * 4, 0.f);
        for (size_t t = 0; t < n; ++t) {
            initial[t * 4] = 1.f;
            for (size_t s = 0; s < 4; ++s) for (int d = 0; d < h; ++d)
                expected[(t * 4 + s) * h + size_t(d)] = kernels::f32_from_bf16(embedding[size_t(tokens[t]) * h + size_t(d)]);
        }
        check(same(download(b.residual, expected.size(), stream), expected), "original token rows repeated bit-exactly into four streams");
        check(same(download(b.pre, initial.size(), stream), initial), "initial coefficients are one-hot, not averaged");
        io.head(b.residual, b.pre, int(n), b.logits, stream);
        compare(download(b.logits, v, stream), text_io_reference::head(w, expected, initial), "embedding through last-token output head vs CPU FP64 dots", 2e-5);
    }
    std::vector<float> residual(3 * 4 * h), pre{1, 0, 0, 0, .5f, .5f, 0, 0, .125f, .625f, .25f, .5f};
    for (float factor : {1.f, 0.f, 1e-12f, 16.f}) {
        for (size_t i = 0; i < residual.size(); ++i) residual[i] = factor * (float(int(i * 13 % 101) - 50) / 64.f + .0001f);
        upload(b.residual, residual, stream); upload(b.pre, pre, stream);
        io.head(b.residual, b.pre, 3, b.logits, stream);
        compare(download(b.logits, v, stream), text_io_reference::head(w, residual, pre), "weighted final collapse, BF16 boundaries, epsilon and FP32 head", 2e-5);
        check(same(download(b.residual, residual.size(), stream), residual) && same(download(b.pre, pre.size(), stream), pre), "output projection preserves all caller residuals and coefficients");
    }
    auto mean = pre; std::fill(mean.end() - 4, mean.end(), .25f);
    check(!same(text_io_reference::head(w, residual, pre), text_io_reference::head(w, residual, mean)), "control distinguishes weighted collapse from stream mean");
    auto first = residual; std::copy(residual.begin(), residual.begin() + 4 * h, first.end() - 4 * h);
    check(!same(text_io_reference::head(w, residual, pre), text_io_reference::head(w, first, pre)), "control distinguishes last token from first token");
    std::vector<float> saved(4 * 4 * h, -17.f), saved_pre(16, -19.f), saved_logits(v, -23.f);
    upload(b.residual, saved, stream); upload(b.pre, saved_pre, stream); upload(b.logits, saved_logits, stream);
    const std::array<int32_t, 2> bad_low{0, -1}, bad_high{0, v}; const std::array<int32_t, 1> good{0};
    rejects("late negative token rejected before output", [&] { io.embed(bad_low, b.residual, b.pre, stream); });
    rejects("late out-of-vocabulary token rejected", [&] { io.embed(bad_high, b.residual, b.pre, stream); });
    rejects("empty embedding rejected", [&] { io.embed({}, b.residual, b.pre, stream); });
    rejects("embedding capacity enforced", [&] { io.embed(std::vector<int32_t>(5, 0), b.residual, b.pre, stream); });
    rejects("overlapping embedding outputs rejected", [&] { io.embed(good, b.residual, b.residual, stream); });
    rejects("unaligned embedding output rejected", [&] { io.embed(good, reinterpret_cast<float*>(reinterpret_cast<char*>(b.residual) + 1), b.pre, stream); });
    rejects("null embedding output rejected", [&] { io.embed(good, nullptr, b.pre, stream); });
    rejects("default stream rejected", [&] { io.embed(good, b.residual, b.pre, nullptr); });
    rejects("changed stream rejected", [&] { io.head(b.residual, b.pre, 1, b.logits, other); });
    rejects("empty head input rejected", [&] { io.head(b.residual, b.pre, 0, b.logits, stream); });
    rejects("head capacity enforced", [&] { io.head(b.residual, b.pre, 5, b.logits, stream); });
    rejects("head cannot overlap residual", [&] { io.head(b.residual, b.pre, 1, b.residual, stream); });
    rejects("head cannot overlap coefficients", [&] { io.head(b.residual, b.pre, 1, b.pre, stream); });
    rejects("head inputs cannot overlap", [&] { io.head(b.residual, b.residual, 1, b.logits, stream); });
    rejects("null logits rejected", [&] { io.head(b.residual, b.pre, 1, nullptr, stream); });
    const auto old = embedding[0]; embedding[0] = 0x7fc0;
    rejects("nonfinite selected embedding row rejected before copy", [&] { io.embed(good, b.residual, b.pre, stream); }); embedding[0] = old;
    check(same(download(b.residual, saved.size(), stream), saved) && same(download(b.pre, saved_pre.size(), stream), saved_pre) && same(download(b.logits, v, stream), saved_logits), "all argument rejections leave output unchanged");
    io.embed(good, b.residual, b.pre, stream); io.head(b.residual, b.pre, 1, b.logits, stream);
    check(io.device_bytes() == bytes && io.host_staging_bytes() == staging, "valid reuse after rejection keeps fixed allocations");
    rejects("zero capacity rejected", [&] { CudaTextIo bad(w, 0); });
    rejects("overlarge capacity rejected", [&] { CudaTextIo bad(w, 1048577); });
    rejects("mismatched head rejected", [&] { auto bad = w; bad.head = bad.head.first(1); CudaTextIo invalid(bad, 1); });
    rejects("invalid geometry rejected", [&] { auto bad = w; bad.hidden = 33; CudaTextIo invalid(bad, 1); });
    head[0] = 0x7f80; rejects("nonfinite head rejected", [&] { CudaTextIo invalid(w, 1); }); head[0] = 0;
    norm[0] = 0x7fc0; rejects("nonfinite norm rejected", [&] { CudaTextIo invalid(w, 1); });
    std::printf("analytical_text_io_device_bytes=%zu host_staging_bytes=%zu\n", bytes, staging);
}
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, uint32_t type) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != type || t->shape.size() != 1 || !t->shape[0] || t->shape[0] > 1048576 || !model.in_bounds(*t, shard))
        throw std::runtime_error("Malformed IO reference " + name);
    std::vector<T> values(size_t(t->shape[0])); std::memcpy(values.data(), model.shard(shard).tensor_data(*t), values.size() * sizeof(T)); return values;
}
void real(const char* path, cudaStream_t stream) {
    const auto model = GgufModel::open(path); const auto w = read_text_io_weights(model);
    CudaTextIo io(w, 4); Buffers b(w.hidden, w.vocab_size, 4);
    for (int c = 0; c < 3; ++c) {
        const std::string base = "embed." + std::to_string(c) + ".";
        const auto* ids = model.meta().get(base + "tokens");
        if (!ids || ids->type != MetaType::ARRAY || ids->elem != MetaType::U32 ||
            !ids->count || ids->count > 4 || ids->items.size() != ids->count)
            throw std::runtime_error("Malformed IO token metadata");
        std::vector<int32_t> tokens;
        for (const auto& id : ids->items) {
            if (id.u >= uint64_t(w.vocab_size)) throw std::runtime_error("Out-of-range IO reference token");
            tokens.push_back(int32_t(id.u));
        }
        io.embed(tokens, b.residual, b.pre, stream);
        const auto expected = tensor<float>(model, base + "residual", 0), pre = tensor<float>(model, base + "pre", 0);
        check(same(download(b.residual, expected.size(), stream), expected), "original embedding vs pinned ParallelEmbedding, case=" + std::to_string(c));
        check(same(download(b.pre, pre.size(), stream), pre), "initial coefficients vs pinned make_identity_pre_mix");
    }
    for (int c = 0; c < 6; ++c) {
        const std::string base = "head." + std::to_string(c) + ".";
        const auto residual = tensor<float>(model, base + "residual", 0), pre = tensor<float>(model, base + "pre", 0);
        if (residual.size() != pre.size() * size_t(w.hidden) || pre.size() % 4) throw std::runtime_error("Reference head shape mismatch");
        upload(b.residual, residual, stream); upload(b.pre, pre, stream);
        io.head(b.residual, b.pre, int(pre.size() / 4), b.logits, stream);
        const auto actual = download(b.logits, size_t(w.vocab_size), stream);
        const auto expected = tensor<float>(model, base + "logits", 0), accurate = tensor<float>(model, base + "logits_fp64", 0);
        compare(actual, expected, "original full-vocabulary logits vs pinned CPU graph, case=" + std::to_string(c), 2e-4);
        compare(actual, accurate, "original full-vocabulary logits vs independent FP64 dots", 2e-4);
        check(std::max_element(actual.begin(), actual.end()) - actual.begin() == std::max_element(expected.begin(), expected.end()) - expected.begin(), "top logit index agrees with CPU reference (no sampling)");
        if (c != 1) check(std::any_of(actual.begin(), actual.end(), [](float x) { return x != bf(x); }), "head keeps FP32 logits rather than BF16-rounding them");
        check(same(download(b.residual, residual.size(), stream), residual) && same(download(b.pre, pre.size(), stream), pre), "original head preserves residual and delayed coefficients");
    }
    std::printf("real_text_io_device_bytes=%zu host_staging_bytes=%zu vocab_size=%d\n", io.device_bytes(), io.host_staging_bytes(), io.vocab_size());
}
} // namespace
int main(int argc, char** argv) {
    if (argc != 1 && !(argc == 3 && std::string(argv[1]) == "--real")) { std::fprintf(stderr, "usage: %s [--real fixture.gguf]\n", argv[0]); return 2; }
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !count)) return 77;
    cudaStream_t stream = nullptr, other = nullptr;
    try {
        ck(status); ck(cudaSetDevice(0)); ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); ck(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
        if (argc == 1) analytical(stream, other); else real(argv[2], stream);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    if (other) cudaStreamDestroy(other);
    if (stream) cudaStreamDestroy(stream);
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
