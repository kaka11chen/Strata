#include "strata/deepseek41/layer3_decoder_cuda.hpp"

#include <cuda_runtime.h>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
const Geometry& checked(const Geometry& g, const DecoderWeights& w, int layer, bool candidate_source = false) {
    const int source = layer < 8 ? 2 : layer < 14 ? 8 : layer < 20 ? 14 : 20;
    const int index_source = layer >= 24 ? layer / 4 * 4 : source;
    if (g.layers != 40 || g.hidden != hidden || g.expert_width != 2304 || g.experts != 384 || g.top_k != 6 ||
        layer < 3 || layer > 39 || layer == 8 || layer == 14 || layer == 20 ||
        candidate_source != (layer == 24 || layer == 28 || layer == 32 || layer == 36) || w.attention.layer != layer ||
        g.compress_ratios.size() != 40 || g.compress_ratios[size_t(source)] != (source == 20 ? 1 : 2) || g.compress_ratios[size_t(layer)] != (source == 20 ? 1 : 2) ||
        g.sources.size() != 40 || g.sources[size_t(source)].kv != source || g.sources[size_t(source)].index != source || g.sources[size_t(source)].candidates != -1 ||
        g.sources[size_t(layer)].kv != source || g.sources[size_t(layer)].index != index_source ||
        g.sources[size_t(layer)].candidates != ((layer == 24 || layer == 28 || layer == 32 || layer == 36) ? 20 : -1) ||
        (layer >= 24 && (g.sources[size_t(index_source)].kv != 20 || g.sources[size_t(index_source)].index != index_source || g.sources[size_t(index_source)].candidates != 20 || g.compress_ratios[size_t(index_source)] != 1)))
        throw std::invalid_argument("Decoder requires fixed 1017 consumers 3..7/9..13/15..19/21..23 and 24..39 with original KV/index/candidate sources");
    return g;
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn;
}
} // namespace
CudaSharedDecoder::CudaSharedDecoder(const Geometry& g, const DecoderWeights& w,
                                    core::ExpertSource& source, int slots, int device, int layer)
    : device_(device), attention_mhc_(checked(g, w, layer), w.attention_mhc, device),
      attention_(w.attention, device, layer),
      ffn_mhc_(g, w.ffn_mhc, device), moe_(g, layer, w.router, w.bias, source, slots, w.shared, device),
      arena_(size_t(hidden) * 20, device) {
    attention_output_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 4));
    middle_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 16));
}
CudaSharedDecoder::CudaSharedDecoder(const Geometry& g, const DecoderWeights& w,
                                    const CandidateIndexQueryWeights& query, core::ExpertSource& source, int slots,
                                    int capacity, int prefill, int device)
    : device_(device), attention_mhc_(checked(g, w, w.attention.layer, true), w.attention_mhc, device),
      attention_(w.attention, query, capacity, prefill, device), ffn_mhc_(g, w.ffn_mhc, device),
      moe_(g, w.attention.layer, w.router, w.bias, source, slots, w.shared, device), arena_(size_t(hidden) * 20, device) {
    attention_output_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 4));
    middle_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 16));
}
void CudaSharedDecoder::check_stream(void* stream) const {
    if (failed_) throw std::logic_error("Shared decoder must be reconstructed after execution failure");
    if (!stream || (stream_ && stream_ != stream)) throw std::invalid_argument("Shared decoder requires one explicit stream per instance");
    int device = -1; ck(cudaGetDevice(&device), "get shared decoder device");
    if (device != device_) throw std::invalid_argument("Shared decoder called on wrong device");
}
void CudaSharedDecoder::forward(const float* input, const float* incoming, float* output, float* next_pre,
                               int tokens, int start_pos, const SharedAttentionBatch& batch,
                               void* stream, std::span<Route> routes) {
    check_stream(stream);
    attention_.validate_forward(tokens, start_pos, batch, stream);
    if (!routes.empty() && routes.size() != size_t(tokens))
        throw std::invalid_argument("Shared decoder requires dimension-matched routes");
    const void* pointers[] = {input, incoming, output, next_pre};
    const size_t bytes[] = {size_t(tokens) * hidden * 16, size_t(tokens) * 16,
                           size_t(tokens) * hidden * 16, size_t(tokens) * 16};
    for (int i = 0; i < 4; ++i) {
        attention_.check_external(pointers[i], bytes[i], batch);
        if (overlaps(pointers[i], bytes[i], arena_.base(), arena_.capacity()))
            throw std::invalid_argument("Shared decoder requires external device buffers");
        for (int j = 0; j < i; ++j)
            if (overlaps(pointers[i], bytes[i], pointers[j], bytes[j]))
                throw std::invalid_argument("Shared decoder input/output buffers overlap");
    }
    try {
        stream_ = stream;
        attention_.begin_forward(stream);
        for (int t = 0; t < tokens; ++t) {
            const auto state = attention_mhc_.prepare(input + size_t(t) * 4 * hidden, incoming + size_t(t) * 4, stream);
            attention_.attend_token(state.normalized, attention_output_, t, tokens, start_pos, batch, stream);
            attention_mhc_.finish(attention_output_, middle_, stream);
            const Route route = ffn_mhc_.forward_moe(moe_, middle_, state.next_pre, output + size_t(t) * 4 * hidden,
                                                    next_pre + size_t(t) * 4, stream);
            if (!routes.empty()) routes[size_t(t)] = route;
        }
        // Only complete the consumer after every residual join and MoE succeeds.
        attention_.finish_forward(tokens, batch, stream);
    } catch (...) { failed_ = true; attention_.fail_forward(stream); throw; }
}
void CudaSharedDecoder::forward_source20(const float* input, const float* incoming, float* output, float* next_pre,
                               int tokens, int start_pos, const Source20AttentionBatch& batch,
                               void* stream, std::span<Route> routes) {
    forward(input, incoming, output, next_pre, tokens, start_pos, batch.attention_batch(), stream, routes);
}
void CudaSharedDecoder::reset(void* stream) {
    check_stream(stream);
    try { attention_.reset(stream); stream_ = stream; }
    catch (...) { failed_ = true; attention_.fail_forward(stream); throw; }
}
size_t CudaSharedDecoder::device_bytes() const {
    return attention_mhc_.device_bytes() + attention_.device_bytes() + ffn_mhc_.device_bytes() +
           moe_.device_bytes() + arena_.capacity();
}
} // namespace strata::deepseek41
