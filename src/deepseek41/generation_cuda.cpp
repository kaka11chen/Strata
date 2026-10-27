#include "strata/deepseek41/generation_cuda.hpp"
#include "strata/kernels/sampler.hpp"
#include <cuda_runtime.h>
#include <algorithm>

namespace strata::deepseek41 {
namespace {
constexpr size_t residual_size = 4 * 5120;
void ck(cudaError_t e, const char* label) {
    if (e != cudaSuccess) throw core::CudaError(std::string(label) + ": " + cudaGetErrorString(e), int(e));
}
size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
} // namespace
CudaGreedyGeneration::CudaGreedyGeneration(CudaTextPrefix& prefix, CudaTextIo& io, int device)
    : prefix_(prefix), io_(io), device_(device),
      prefill_(std::min(prefix.prefill_capacity(), io.prefill_capacity())),
      context_(std::min(1048576, 2 * prefix.capacity() + 1)) {
    if (io.hidden_size() != 5120 || io.vocab_size() < 2 || io.vocab_size() > 129280)
        throw std::invalid_argument("Generation requires the confirmed 1017 hidden size and vocabulary");
    if (prefix.layer_count() >= 24) context_ = std::min(context_, prefix.capacity(20));
    const size_t residual_bytes = size_t(prefill_) * residual_size * sizeof(float);
    const size_t pre_bytes = size_t(prefill_) * 4 * sizeof(float);
    const size_t logits_bytes = size_t(io.vocab_size()) * sizeof(float);
    arena_ = std::make_unique<core::DeviceArena>(2 * aligned(residual_bytes) + 2 * aligned(pre_bytes) +
                                              aligned(logits_bytes) + sizeof(int), device);
    input_ = static_cast<float*>(arena_->alloc(residual_bytes));
    initial_ = static_cast<float*>(arena_->alloc(pre_bytes));
    output_ = static_cast<float*>(arena_->alloc(residual_bytes));
    final_ = static_cast<float*>(arena_->alloc(pre_bytes));
    logits_ = static_cast<float*>(arena_->alloc(logits_bytes));
    selected_ = static_cast<int*>(arena_->alloc(sizeof(int)));
}
GenerationResult CudaGreedyGeneration::generate(std::span<const int32_t> prompt, int max_new_tokens, void* stream) {
    if (failed_) throw std::logic_error("Generation must be reconstructed after an execution failure");
    if (!stream || (stream_ && stream != stream_))
        throw std::invalid_argument("Generation requires one explicit stream");
    int device = -1; ck(cudaGetDevice(&device), "get generation device");
    if (device != device_) throw std::invalid_argument("Generation called on wrong device");
    if (prompt.empty() || !prompt.data() || prompt.size() > size_t(prefill_) || max_new_tokens < 0 ||
        max_new_tokens > context_ || prompt.size() > size_t(context_) ||
        (max_new_tokens && prompt.size() + size_t(max_new_tokens - 1) > size_t(context_)))
        throw std::invalid_argument("Generation prompt or token budget exceeds bounded capacity");
    if (uintptr_t(prompt.data()) % alignof(int32_t) ||
        prompt.size_bytes() > UINTPTR_MAX - uintptr_t(prompt.data()))
        throw std::invalid_argument("Generation prompt must be an aligned host token span");
    for (int32_t id : prompt)
        if (id < 0 || id >= io_.vocab_size()) throw std::invalid_argument("Generation token outside vocabulary");
    GenerationResult result;
    if (!max_new_tokens) return result;
    result.tokens.reserve(size_t(max_new_tokens));
    kernels::SamplerParams params;
    params.greedy = true; params.temperature = 0.f; params.top_k = 0; params.top_p = 1.f;
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        prefix_.reset(stream);
        stream_ = stream;
        for (int step = 0; step < max_new_tokens; ++step) {
            const int32_t last = step ? result.tokens.back() : 0;
            const std::span<const int32_t> input = step ? std::span(&last, 1) : prompt;
            io_.embed(input, input_, initial_, stream);
            prefix_.forward(input, input_, initial_, output_, final_, prefix_.next_position(), stream);
            io_.head(output_, final_, int(input.size()), logits_, stream);
            kernels::sample_tokens(logits_, 1, io_.vocab_size(), nullptr, 0, params, selected_, stream);
            int next = -1;
            ck(cudaMemcpyAsync(&next, selected_, sizeof(next), cudaMemcpyDeviceToHost, s), "read greedy token");
            ck(cudaStreamSynchronize(s), "finish greedy token");
            if (next < 0 || next >= io_.vocab_size()) throw std::runtime_error("Sampler returned an invalid token");
            result.tokens.push_back(int32_t(next));
            if (next == 1) { result.stopped_on_eos = true; break; }
        }
    } catch (...) {
        failed_ = true;
        (void)cudaStreamSynchronize(s);
        throw;
    }
    return result;
}
} // namespace strata::deepseek41
