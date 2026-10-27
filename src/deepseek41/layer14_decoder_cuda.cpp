#include "strata/deepseek41/layer14_decoder_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120, max_position = 1048576;
constexpr size_t lookup_size = 6144, residual_size = 4 * hidden;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(e), int(e));
}
int checked(const Geometry& g, const DecoderWeights& w, const EngramWeights& e,
            const CompressorWeights& c, const IndexKeyWeights& k, const IndexQueryWeights& q,
            int capacity, int publication_capacity, int device) {
    if (g.layers != 40 || g.hidden != hidden || g.expert_width != 2304 || g.experts != 384 || g.top_k != 6 ||
        w.attention.layer != 14 || e.layer != 14 || c.layer != 14 || k.layer != 14 || q.layer != 14 ||
        g.compress_ratios.size() != 40 || g.compress_ratios[14] != 2 || g.sources.size() != 40 ||
        g.sources[14].kv != 14 || g.sources[14].index != 14 || g.sources[14].candidates != -1 ||
        capacity <= 0 || capacity > max_position / 2 || publication_capacity < 0 || publication_capacity > max_position)
        throw std::invalid_argument("Layer-14 decoder requires matching Engram/source weights, owned KV/index geometry and bounded capacities");
    return device;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    return an && bn && (uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn);
}
} // namespace
CudaLayer14Decoder::CudaLayer14Decoder(const Geometry& g, const DecoderWeights& w, const EngramWeights& e,
    const CompressorWeights& c, const IndexKeyWeights& k, const IndexQueryWeights& q,
    std::span<const uint32_t> map, const std::string& path, core::ExpertSource& source, int slots,
    int capacity, const EngramIoOptions& io, int device, int publication_capacity)
    : device_(checked(g, w, e, c, k, q, capacity, publication_capacity, device)),
      compressor_hash_(14, map), hash_(14, map), table_(14, path, io), engram_(e, device),
      attention_mhc_(g, w.attention_mhc, device), attention_(w.attention, c, k, q, capacity, device, publication_capacity, 14),
      ffn_mhc_(g, w.ffn_mhc, device), moe_(g, 14, w.router, w.bias, source, slots, w.shared, device),
      arena_((lookup_size + 2 * residual_size + hidden) * 4, device) {
    lookup_ = static_cast<float*>(arena_.alloc(lookup_size * 4));
    engram_residual_ = static_cast<float*>(arena_.alloc(residual_size * 4));
    attention_output_ = static_cast<float*>(arena_.alloc(hidden * 4));
    middle_ = static_cast<float*>(arena_.alloc(residual_size * 4));
}
void CudaLayer14Decoder::check_stream(void* stream) const {
    if (failed_) throw std::logic_error("Layer-14 decoder must be reconstructed after execution failure");
    if (!stream || (stream_ && stream != stream_)) throw std::invalid_argument("Layer-14 decoder requires one explicit stream");
    int device = -1; ck(cudaGetDevice(&device), "get layer-14 decoder device");
    if (device != device_) throw std::invalid_argument("Layer-14 decoder called on wrong device");
}
void CudaLayer14Decoder::prepare_engram(EngramHash& hash, std::span<const int32_t> token,
                                       const float* residual, int position, void* stream) {
    hash.forward(token, rows_, position);
    table_.gather(rows_, lookup_bits_);
    for (size_t i = 0; i < lookup_size; ++i) lookup_values_[i] = kernels::f32_from_bf16(lookup_bits_[i]);
    ck(cudaMemcpyAsync(lookup_, lookup_values_.data(), lookup_size * 4, cudaMemcpyHostToDevice,
                       static_cast<cudaStream_t>(stream)), "upload layer-14 Engram lookup");
    engram_.forward(lookup_, residual, engram_residual_, stream);
}
void CudaLayer14Decoder::forward(std::span<const int32_t> tokens, const float* input, const float* incoming,
    float* output, float* next_pre, int start, void* stream, std::span<Route> routes) {
    check_stream(stream);
    const size_t count = tokens.size();
    if (!count || count > max_position || (!routes.empty() && routes.size() != count))
        throw std::invalid_argument("Layer-14 decoder requires text tokens and matching routes within the context limit");
    attention_.validate_forward(int(count), start, stream);
    const void* pointers[] = {input, incoming, output, next_pre, tokens.data(), routes.data()};
    const size_t bytes[] = {count * residual_size * 4, count * 16, count * residual_size * 4,
                           count * 16, tokens.size_bytes(), routes.size_bytes()};
    for (size_t i = 0; i < 6; ++i) {
        if (!bytes[i]) continue;
        attention_.check_external(pointers[i], bytes[i]);
        if (uintptr_t(pointers[i]) % (i == 5 ? alignof(Route) : 4) ||
            overlap(pointers[i], bytes[i], arena_.base(), arena_.capacity()))
            throw std::invalid_argument("Layer-14 decoder requires aligned external non-cache buffers");
        for (size_t j = 0; j < i; ++j)
            if (overlap(pointers[i], bytes[i], pointers[j], bytes[j])) throw std::invalid_argument("Layer-14 decoder caller buffers overlap");
    }
    for (int32_t id : tokens)
        if (id < 0 || size_t(id) >= engram_token_count) throw std::invalid_argument("Layer-14 original token ID out of range");
    try {
        stream_ = stream;
        attention_.begin_forward(stream);
        for (size_t t = 0; t < count; ++t) {
            prepare_engram(compressor_hash_, tokens.subspan(t, 1), input + t * residual_size, start + int(t), stream);
            const float* normalized = attention_mhc_.normalize_input(engram_residual_, incoming + t * 4, stream);
            attention_.compress_token(normalized, start + int(t), stream);
        }
        for (size_t t = 0; t < count; ++t) {
            prepare_engram(hash_, tokens.subspan(t, 1), input + t * residual_size, start + int(t), stream);
            const auto state = attention_mhc_.prepare(engram_residual_, incoming + t * 4, stream);
            attention_.attend_token(state.normalized, attention_output_, int(t), int(count), start, stream);
            attention_mhc_.finish(attention_output_, middle_, stream);
            const Route route = ffn_mhc_.forward_moe(moe_, middle_, state.next_pre, output + t * residual_size, next_pre + t * 4, stream);
            if (!routes.empty()) routes[t] = route;
        }
        attention_.finish_forward(int(count), start, stream);
    } catch (...) { failed_ = true; attention_.fail_forward(stream); throw; }
}
void CudaLayer14Decoder::reset(void* stream) {
    check_stream(stream);
    try { attention_.reset(stream); compressor_hash_.reset(); hash_.reset(); stream_ = stream; }
    catch (...) { failed_ = true; attention_.fail_forward(stream); throw; }
}
size_t CudaLayer14Decoder::device_bytes() const {
    return engram_.device_bytes() + attention_mhc_.device_bytes() + attention_.device_bytes() +
           ffn_mhc_.device_bytes() + moe_.device_bytes() + arena_.capacity();
}
} // namespace strata::deepseek41
