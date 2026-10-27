#include "strata/deepseek41/layer2_decoder_cuda.hpp"

#include <cuda_runtime.h>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
const Geometry& checked(const Geometry& g, const DecoderWeights& w, const CompressorWeights& c,
                        const IndexKeyWeights& k, const IndexQueryWeights& q, int layer) {
    if (g.layers != 40 || g.hidden != hidden || g.expert_width != 2304 || g.experts != 384 || g.top_k != 6 ||
        (layer != 2 && layer != 8) || w.attention.layer != layer || c.layer != layer || k.layer != layer || q.layer != layer ||
        g.compress_ratios.size() != 40 || g.compress_ratios[size_t(layer)] != 2 || g.sources.size() != 40 ||
        g.sources[size_t(layer)].kv != layer || g.sources[size_t(layer)].index != layer || g.sources[size_t(layer)].candidates != -1)
        throw std::invalid_argument("Decoder requires matching fixed 1017 source-2/source-8 weights and owned KV/index geometry");
    return g;
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn;
}
} // namespace
CudaSourceDecoder::CudaSourceDecoder(const Geometry& g, const DecoderWeights& w,
                                    const CompressorWeights& c, const IndexKeyWeights& k,
                                    const IndexQueryWeights& q, core::ExpertSource& source,
                                    int slots, int capacity_groups, int device, int shared_prefill_capacity, int layer)
    : device_(device), attention_mhc_(checked(g, w, c, k, q, layer), w.attention_mhc, device),
      attention_(w.attention, c, k, q, capacity_groups, device, shared_prefill_capacity, layer),
      ffn_mhc_(g, w.ffn_mhc, device), moe_(g, layer, w.router, w.bias, source, slots, w.shared, device),
      arena_(size_t(hidden) * 20, device) {
    attention_output_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 4));
    middle_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 16));
}
void CudaSourceDecoder::check_stream(void* stream) const {
    if (failed_) throw std::logic_error("Source decoder must be reconstructed after execution failure");
    if (!stream || (stream_ && stream_ != stream)) throw std::invalid_argument("Source decoder requires one explicit stream per instance");
    int device = -1; ck(cudaGetDevice(&device), "get source decoder device");
    if (device != device_) throw std::invalid_argument("Source decoder called on wrong device");
}
void CudaSourceDecoder::forward(const float* input, const float* incoming, float* output, float* next_pre,
                               int tokens, int start_pos, void* stream, std::span<Route> routes) {
    check_stream(stream);
    attention_.validate_forward(tokens, start_pos, stream);
    if (!routes.empty() && routes.size() != size_t(tokens))
        throw std::invalid_argument("Source decoder requires dimension-matched routes");
    const void* pointers[] = {input, incoming, output, next_pre};
    const size_t bytes[] = {size_t(tokens) * hidden * 16, size_t(tokens) * 16,
                           size_t(tokens) * hidden * 16, size_t(tokens) * 16};
    for (int i = 0; i < 4; ++i) {
        attention_.check_external(pointers[i], bytes[i]);
        if (overlaps(pointers[i], bytes[i], arena_.base(), arena_.capacity()))
            throw std::invalid_argument("Source decoder requires external device buffers");
        for (int j = 0; j < i; ++j)
            if (overlaps(pointers[i], bytes[i], pointers[j], bytes[j]))
                throw std::invalid_argument("Source decoder input/output buffers overlap");
    }
    try {
        stream_ = stream;
        attention_.begin_forward(stream);
        for (int t = 0; t < tokens; ++t) {
            const float* normalized = attention_mhc_.normalize_input(input + size_t(t) * 4 * hidden,
                                                                     incoming + size_t(t) * 4, stream);
            attention_.compress_token(normalized, start_pos + t, stream);
        }
        for (int t = 0; t < tokens; ++t) {
            const auto state = attention_mhc_.prepare(input + size_t(t) * 4 * hidden, incoming + size_t(t) * 4, stream);
            attention_.attend_token(state.normalized, attention_output_, t, tokens, start_pos, stream);
            attention_mhc_.finish(attention_output_, middle_, stream);
            const Route route = ffn_mhc_.forward_moe(moe_, middle_, state.next_pre, output + size_t(t) * 4 * hidden,
                                                    next_pre + size_t(t) * 4, stream);
            if (!routes.empty()) routes[size_t(t)] = route;
        }
        // No publication is visible until residual joins and every MoE finish.
        attention_.finish_forward(tokens, start_pos, stream);
    } catch (...) { failed_ = true; attention_.fail_forward(stream); throw; }
}
void CudaSourceDecoder::reset(void* stream) {
    check_stream(stream);
    try { attention_.reset(stream); stream_ = stream; }
    catch (...) { failed_ = true; attention_.fail_forward(stream); throw; }
}
size_t CudaSourceDecoder::device_bytes() const {
    return attention_mhc_.device_bytes() + attention_.device_bytes() + ffn_mhc_.device_bytes() +
           moe_.device_bytes() + arena_.capacity();
}
} // namespace strata::deepseek41
