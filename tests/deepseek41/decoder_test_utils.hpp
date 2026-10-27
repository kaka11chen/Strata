#pragma once

// Shared decoder test fixtures and independent CPU checks; no runtime code.
#include "strata/deepseek41/layer2_decoder_cuda.hpp"
#include "strata/deepseek41/layer3_attention_cuda.hpp"
#include "strata/deepseek41/expert_source.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>

using namespace strata;
using namespace strata::deepseek41;
namespace {
constexpr int hidden = 5120, residual_size = 4 * hidden;
int checks = 0, failures = 0;
void check(bool ok, const std::string& name) { ++checks; failures += !ok; std::printf("%s %s\n", ok ? "ok  " : "FAIL", name.c_str()); }
void ck(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void rejects(const std::string& name, const std::function<void()>& fn, bool execution = false) {
    try { fn(); check(false, name); }
    catch (const std::invalid_argument&) { check(!execution, name); }
    catch (const std::exception& e) { check(execution, name + ": " + e.what()); }
}
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
Geometry geometry() {
    Geometry g; g.layers = 40; g.hidden = hidden; g.expert_width = 2304; g.experts = 384; g.top_k = 6; g.compress_ratios.assign(40, 0); for (int l=2;l<8;++l) g.compress_ratios[size_t(l)] = 2;
    g.sources.resize(40); for (int l=2;l<8;++l) g.sources[size_t(l)] = {2, 2, -1}; for (int l=8;l<14;++l) { g.compress_ratios[size_t(l)] = 2; g.sources[size_t(l)] = {8,8,-1}; } return g;
}
float bf(float value) { return kernels::f32_from_bf16(kernels::bf16_from_f32(value)); }
void upload(float* dst, const std::vector<float>& input, cudaStream_t stream) {
    ck(cudaMemcpyAsync(dst, input.data(), input.size() * 4, cudaMemcpyHostToDevice, stream));
}
std::vector<float> download(const float* data, size_t n, cudaStream_t stream) {
    std::vector<float> out(n); ck(cudaMemcpyAsync(out.data(), data, n * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); return out;
}
std::vector<float> cache(const uint16_t* data, cudaStream_t stream, size_t n = 128 * 512) {
    std::vector<uint16_t> raw(n); ck(cudaMemcpyAsync(raw.data(), data, raw.size() * 2, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream)); std::vector<float> out; out.reserve(raw.size());
    for (auto v : raw) out.push_back(kernels::f32_from_bf16(v));
    return out;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
inline void compare(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& name, bool coeff = false, bool quantized = false) {
    if (actual.size() != expected.size()) throw std::runtime_error("Reference shape mismatch");
    double error = 0., norm = 0., peak = 0., max_abs = 0.; bool finite = true, rounded = true, pointwise = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i]; error += delta * delta; norm += double(expected[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(delta)); peak = std::max(peak, std::abs(double(expected[i])));
        finite &= std::isfinite(actual[i]) && std::isfinite(expected[i]); rounded &= actual[i] == bf(actual[i]);
        pointwise &= std::abs(delta) <= 2e-6 + 2e-5 * std::abs(double(expected[i]));
    }
    const double rel = std::sqrt(error / std::max(norm, 1e-30));
    std::printf("  %s relL2=%.9g max_abs=%.9g peak=%.9g\n", name.c_str(), rel, max_abs, peak);
    check(finite && (coeff ? pointwise : rounded && rel <= (quantized ? .003 : .005) && max_abs <= 1e-5 + (quantized ? .02 : .03) * peak), name);
}
// CPU residual join; tensors use official source-row/destination-column order.
inline std::vector<float> join(std::span<const float> residual, std::span<const float> sublayer,
                         std::span<const float> post, std::span<const float> comb) {
    const size_t tokens = sublayer.size() / hidden; std::vector<float> out(tokens * residual_size);
    for (size_t t = 0; t < tokens; ++t)
        for (size_t dest = 0; dest < 4; ++dest)
            for (size_t d = 0; d < hidden; ++d) {
                float mixed = 0.f;
                for (size_t source = 0; source < 4; ++source)
                    mixed += comb[t * 16 + source * 4 + dest] * residual[t * residual_size + source * hidden + d];
                out[t * residual_size + dest * hidden + d] = bf(post[t * 4 + dest] * bf(sublayer[t * hidden + d]) + mixed);
            }
    return out;
}
inline std::vector<float> collapse_norm(const std::vector<float>& residual, const std::vector<float>& pre) {
    const size_t tokens = pre.size() / 4; std::vector<float> out(tokens * hidden);
    for (size_t t = 0; t < tokens; ++t) {
        double squares = 0.;
        for (int d = 0; d < hidden; ++d) {
            float sum = 0.f;
            for (int s = 0; s < 4; ++s) sum += pre[t * 4 + s] * residual[t * residual_size + size_t(s) * hidden + d];
            out[t * hidden + d] = bf(sum); squares += double(bf(sum)) * bf(sum);
        }
        const float inv = 1.f / std::sqrt(float(squares / hidden) + 1e-20f);
        for (int d = 0; d < hidden; ++d) out[t * hidden + d] = bf(out[t * hidden + d] * inv);
    }
    return out;
}
template<class Attention> std::array<std::vector<float>, 3> history(const Attention& a, cudaStream_t stream) {
    return {cache(a.kv_cache(), stream), cache(a.compressed_kv_cache(), stream, size_t(a.capacity()) * 512),
            cache(a.index_key_cache(), stream, size_t(a.capacity()) * 128)};
}
struct CoveredSource : core::ExpertSource {
    Mxfp4ExpertSource source;
    std::array<bool, 384> populated{};
    int expected_layer;
    CoveredSource(const Geometry& g, const ExpertIndex& index, const std::array<bool, 384>& ids, int layer = 2)
        : source(g, index), populated(ids), expected_layer(layer) {}
    void validate(int64_t layer, int64_t expert) const {
        if (layer != expected_layer || expert < 0 || expert >= 384 || !populated[size_t(expert)])
            throw std::runtime_error("Attempt to read an unpopulated fixture expert");
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override { validate(layer, expert); return source.blob(layer, expert); }
    bool copy_blob(int64_t layer, int64_t expert, uint8_t* dst) override { validate(layer, expert); return source.copy_blob(layer, expert, dst); }
    const uint8_t* blob_stable(int64_t layer, int64_t expert) override { validate(layer, expert); return source.blob_stable(layer, expert); }
    void note_async_read(const uint8_t* blob, void* stream) override { source.note_async_read(blob, stream); }
    bool transient(int64_t, int64_t) const override { return true; }
    int64_t reads() const override { return source.reads(); }
};
struct Matrix {
    std::vector<uint8_t> values, scales; int rows, cols;
    Matrix(int r, int c, bool diagonal) : values(size_t(r) * c, 0), scales(size_t(r / 32) * (c / 32), 127), rows(r), cols(c) {
        if (diagonal) for (int i = 0; i < r; ++i) values[size_t(i) * c + i % c] = 0x30; // 0.5
    }
    Fp8MatrixView view() const { return {values, scales, rows, cols}; }
};
struct ConstantMhc {
    std::vector<float> fn = std::vector<float>(96 * hidden, 0.f);
    std::array<float, 3> scale{0.f, 0.f, 0.f};
    std::array<float, 24> base{};
    std::vector<uint16_t> norm = std::vector<uint16_t>(hidden, kernels::bf16_from_f32(1.f));
    explicit ConstantMhc(float shift) {
        for (int i = 0; i < 8; ++i) base[size_t(i)] = shift + float(i) * .2f;
    }
    MhcWeights view() const { return {fn, scale, base, norm}; }
    std::array<float, 4> pre() const {
        std::array<float, 4> v{};
        for (size_t i = 0; i < 4; ++i) v[i] = 1.f / (1.f + std::exp(-base[i])) + 1e-6f;
        return v;
    }
    std::vector<float> post(int count) const {
        std::vector<float> v(size_t(count) * 4);
        for (size_t i = 0; i < v.size(); ++i) v[i] = 2.f / (1.f + std::exp(-base[4 + i % 4]));
        return v;
    }
    std::vector<float> comb(int count) const {
        float v = .25f + 1e-6f;
        // All equal logits: initial row softmax+eps, column norm, 19 row/column pairs.
        for (int step = 0; step < 39; ++step) v /= 4.f * v + 1e-6f;
        return std::vector<float>(size_t(count) * 16, v);
    }
};
struct ZeroSource : core::ExpertSource {
    std::vector<uint8_t> bytes;
    int calls = 0; bool missing = false;
    int expected_layer;
    explicit ZeroSource(const Geometry& g, int layer = 2) : bytes(nvfp4_expert_bytes(g), 0), expected_layer(layer) {
        const std::array<float,4> scales{1.f,1.f,1.f,0.f};
        std::memcpy(bytes.data()+bytes.size()-nvfp4_scale_tail_bytes,scales.data(),nvfp4_scale_tail_bytes);
    }
    const uint8_t* blob(int64_t layer, int64_t expert) override {
        if (layer != expected_layer || expert < 0 || expert >= 384) throw std::runtime_error("Unexpected original expert ID");
        ++calls; return missing ? nullptr : bytes.data();
    }
};
struct SyntheticWeights {
    std::array<Matrix, 5> attention{Matrix(1280,5120,true), Matrix(32768,1280,true), Matrix(512,5120,true), Matrix(8192,4096,true), Matrix(5120,8192,true)};
    std::array<Matrix, 3> shared{Matrix(2304,5120,true), Matrix(2304,5120,true), Matrix(5120,2304,true)};
    std::vector<uint16_t> qnorm = std::vector<uint16_t>(1280, kernels::bf16_from_f32(1.f));
    std::vector<uint16_t> kvnorm = std::vector<uint16_t>(512, kernels::bf16_from_f32(1.f));
    std::vector<uint16_t> router = std::vector<uint16_t>(384 * hidden, 0);
    std::vector<float> sinks = std::vector<float>(64, 0.f), bias = std::vector<float>(384);
    std::vector<uint16_t> compkv = std::vector<uint16_t>(512 * hidden, 0), gate = compkv;
    std::vector<uint16_t> compnorm = kvnorm, key = std::vector<uint16_t>(128 * 512, 0);
    std::vector<uint16_t> keynorm = std::vector<uint16_t>(128, 0x3f80), head_weights = std::vector<uint16_t>(32 * hidden, 0);
    Matrix index_query{4096,1280,true};
    CompressorWeights compressor(int layer=2) const { return {compkv, gate, compnorm, layer}; }
    IndexKeyWeights index_key(int layer=2) const { return {key, keynorm, layer}; }
    IndexQueryWeights query(int layer=2) const { return {index_query.view(), head_weights, layer}; }
    ConstantMhc attn_mhc{-.7f}, ffn_mhc{-1.4f};
    SyntheticWeights() {
        for (int i = 0; i < 384; ++i) bias[size_t(i)] = -float(i);
        for (int i = 0; i < 512; ++i) compkv[size_t(i) * hidden + i] = 0x3f80;
        for (int i = 0; i < 128; ++i) key[size_t(i) * 512 + i] = 0x3f80;
        for (int i = 0; i < 32; ++i) head_weights[size_t(i) * hidden + i] = 0x3f80;
    }
    DecoderWeights view(int layer = 2) const {
        DecoderWeights w; w.attention_mhc = attn_mhc.view(); w.ffn_mhc = ffn_mhc.view();
        for (size_t i = 0; i < 5; ++i) w.attention.projections[i] = attention[i].view();
        w.attention.q_norm = qnorm; w.attention.kv_norm = kvnorm; w.attention.sink = sinks;
        for (size_t i = 0; i < 3; ++i) w.shared.parts[i] = shared[i].view();
        w.router = router; w.bias = bias; w.attention.layer = layer; return w;
    }
};
template<class T> std::vector<T> tensor(const GgufModel& model, const std::string& name, uint32_t type, size_t n) {
    size_t shard = 0; const auto* t = model.find(name, &shard);
    if (!t || t->type != type || tensor_payload_bytes(*t) != n * sizeof(T) || !model.in_bounds(*t, shard))
        throw std::runtime_error("Missing/invalid fixture tensor " + name);
    std::vector<T> out(n); std::memcpy(out.data(), model.shard(shard).tensor_data(*t), out.size() * sizeof(T)); return out;
}
// Independent double-precision CPU form of the four delayed coefficients.
// Unlike the full CPU graph reference, this can use the native attention boundary
// as input and retain the original strict, same-input mHC acceptance threshold.
inline std::vector<float> cpu_pre(const std::vector<float>& residual, const MhcWeights& w) {
    const size_t tokens = residual.size() / residual_size; std::vector<float> out(tokens * 4);
    for (size_t t = 0; t < tokens; ++t) {
        double squares = 0.;
        for (int d = 0; d < residual_size; ++d) squares += double(residual[t * residual_size + d]) * residual[t * residual_size + d];
        const double inv = 1. / std::sqrt(squares / residual_size + 1e-20);
        for (int row = 0; row < 4; ++row) {
            double dot = 0.;
            for (int d = 0; d < residual_size; ++d) dot += double(w.projection[size_t(row) * residual_size + d]) * residual[t * residual_size + d];
            out[t * 4 + row] = float(1. / (1. + std::exp(-(dot * inv * w.scale[0] + w.base[size_t(row)]))) + 1e-6);
        }
    }
    return out;
}
// |sigmoid'(z)| <= 1/4. Bound propagation from the independently measured
// attention residual difference, after each input's own full-stream RMSNorm:
// |delta pre_i| <= |scale|/4 * |sum_j W_ij (x_j/rms(x)-ref_j/rms(ref))|.
// The original mHC numerical tolerance is added separately, not replaced.
inline void compare_propagated(const std::vector<float>& actual, const std::vector<float>& expected,
                        const std::vector<float>& residual, const std::vector<float>& reference,
                        const MhcWeights& w, const std::string& name) {
    if (actual.size() != expected.size() || residual.size() != reference.size() || residual.size() != actual.size() * hidden)
        throw std::runtime_error("Propagation reference shape mismatch");
    bool valid = true; double maximum = 0., max_bound = 0., max_ratio = 0.;
    for (size_t t = 0; t < actual.size() / 4; ++t) {
        double a2 = 0., r2 = 0.;
        for (int d = 0; d < residual_size; ++d) {
            const size_t j = t * residual_size + d;
            a2 += double(residual[j]) * residual[j]; r2 += double(reference[j]) * reference[j];
        }
        const double ai = 1. / std::sqrt(a2 / residual_size + 1e-20), ri = 1. / std::sqrt(r2 / residual_size + 1e-20);
        for (int row = 0; row < 4; ++row) {
            double propagated = 0.;
            for (int d = 0; d < residual_size; ++d) {
                const size_t j = t * residual_size + d;
                propagated += double(w.projection[size_t(row) * residual_size + d]) * (residual[j] * ai - reference[j] * ri);
            }
            const size_t i = t * 4 + row;
            const double bound = .25 * std::abs(double(w.scale[0]) * propagated) + 2e-6 + 2e-5 * std::abs(double(expected[i]));
            const double delta = std::abs(double(actual[i]) - expected[i]);
            valid &= std::isfinite(delta) && std::isfinite(bound) && delta <= bound;
            maximum = std::max(maximum, delta); max_bound = std::max(max_bound, bound); max_ratio = std::max(max_ratio, delta / bound);
        }
    }
    std::printf("  %s max_abs=%.9g max_bound=%.9g max_fraction=%.9g\n", name.c_str(), maximum, max_bound, max_ratio);
    check(valid, name);
}
} // namespace
