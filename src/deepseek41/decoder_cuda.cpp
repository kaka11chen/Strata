#include "strata/deepseek41/decoder_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120, max_position = 1048576;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess) throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
const Geometry& checked(const Geometry& g) {
    if (g.layers != 40 || g.hidden != hidden || g.expert_width != 2304 || g.experts != 384 || g.top_k != 6 ||
        g.compress_ratios.empty() || g.compress_ratios[0] != 0)
        throw std::invalid_argument("Decoder requires the fixed 1017 layer-0 text geometry");
    return g;
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn;
}
} // namespace
CudaDecoder::CudaDecoder(const Geometry& g, const DecoderWeights& w, core::ExpertSource& source, int slots, int device)
    : device_(device), attention_mhc_(checked(g), w.attention_mhc, device), attention_(w.attention, device),
      ffn_mhc_(g, w.ffn_mhc, device), moe_(g, 0, w.router, w.bias, source, slots, w.shared, device),
      arena_(size_t(hidden) * 20, device) {
    attention_output_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 4));
    middle_ = static_cast<float*>(arena_.alloc(size_t(hidden) * 16));
}
void CudaDecoder::check_stream(void* stream) const {
    if (failed_) throw std::logic_error("Decoder must be reconstructed after an execution failure");
    if (!stream || (stream_ && stream_ != stream)) throw std::invalid_argument("Decoder requires one explicit stream per instance");
    int device = -1; ck(cudaGetDevice(&device), "get decoder device");
    if (device != device_) throw std::invalid_argument("Decoder called on wrong device");
}
void CudaDecoder::forward(const float* input, const float* incoming, float* output, float* next_pre,
                          int tokens, int start_pos, void* stream, std::span<Route> routes) {
    check_stream(stream);
    if (tokens <= 0 || tokens > max_position || start_pos < 0 || start_pos != next_ ||
        start_pos > max_position - tokens || (start_pos != 0 && tokens != 1) ||
        (!routes.empty() && routes.size() != size_t(tokens)))
        throw std::invalid_argument("Decoder requires initial prefill then contiguous one-token decode and dimension-matched routes");
    const void* pointers[] = {input, incoming, output, next_pre};
    const size_t bytes[] = {size_t(tokens) * hidden * 16, size_t(tokens) * 16,
                           size_t(tokens) * hidden * 16, size_t(tokens) * 16};
    for (int i = 0; i < 4; ++i) {
        if (!pointers[i] || uintptr_t(pointers[i]) % 4 || bytes[i] > UINTPTR_MAX - uintptr_t(pointers[i]) ||
            overlaps(pointers[i], bytes[i], arena_.base(), arena_.capacity()))
            throw std::invalid_argument("Decoder requires aligned external device buffers");
        for (int j = 0; j < i; ++j)
            if (overlaps(pointers[i], bytes[i], pointers[j], bytes[j]))
                throw std::invalid_argument("Decoder input/output buffers overlap");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        stream_ = stream;
        const int prefill_width = start_pos == 0 ? std::min(tokens, 128) : 0;
        for (int t = 0; t < tokens; ++t) {
            const auto state = attention_mhc_.prepare(input + size_t(t) * 4 * hidden, incoming + size_t(t) * 4, stream);
            attention_.forward_token(state.normalized, attention_output_, start_pos + t, prefill_width, stream);
            attention_mhc_.finish(attention_output_, middle_, stream);
            const Route route = ffn_mhc_.forward_moe(moe_, middle_, state.next_pre, output + size_t(t) * 4 * hidden,
                                                    next_pre + size_t(t) * 4, stream);
            if (!routes.empty()) routes[size_t(t)] = route;
        }
        ck(cudaStreamSynchronize(s), "wait decoder execution");
        next_ += tokens;
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaDecoder::reset(void* stream) {
    check_stream(stream);
    try { attention_.reset(stream); next_ = 0; stream_ = stream; }
    catch (...) { failed_ = true; (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); throw; }
}
size_t CudaDecoder::device_bytes() const {
    return attention_mhc_.device_bytes() + attention_.device_bytes() + ffn_mhc_.device_bytes() +
           moe_.device_bytes() + arena_.capacity();
}
} // namespace strata::deepseek41
