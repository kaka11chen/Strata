#pragma once

#include "strata/deepseek41/text_io_cuda.hpp"
#include "strata/deepseek41/text_prefix_cuda.hpp"

namespace strata::deepseek41 {
struct GenerationResult {
    std::vector<int32_t> tokens; // Generated IDs only, including EOS when selected.
    bool stopped_on_eos = false;
};

// Bounded, batch-one greedy completion. Borrows the prefix and text boundaries;
// both must outlive this object and use the same device and explicit stream.
// The CLI uses the full 40-layer backbone; shorter confirmed prefixes support tests.
// No tokenizer, chat template, sampling penalties or MTP. Reuses Strata's sampler.
class CudaGreedyGeneration {
  public:
    CudaGreedyGeneration(CudaTextPrefix& prefix, CudaTextIo& io, int device = 0);
    // Each positive-budget request resets sequence histories, retaining weight caches.
    // Validate all IDs and worst-case evaluated length BEFORE resetting anything.
    // At most prompt.size()+max_new_tokens-1 tokens enter the backbone: the final
    // sampled ID (including EOS=1) is returned without an unnecessary forward.
    // max_new_tokens=0 is a no-op. No partial result is returned after a failure;
    // an execution failure requires reconstruction, as for the borrowed objects.
    GenerationResult generate(std::span<const int32_t> prompt, int max_new_tokens, void* stream);
    int context_capacity() const { return context_; }
    int prefill_capacity() const { return prefill_; }
    size_t device_bytes() const { return arena_->capacity(); } // Own workspace only.
  private:
    CudaTextPrefix& prefix_;
    CudaTextIo& io_;
    int device_, prefill_, context_;
    void* stream_ = nullptr;
    bool failed_ = false;
    std::unique_ptr<core::DeviceArena> arena_;
    float *input_, *initial_, *output_, *final_, *logits_;
    int* selected_;
};
} // namespace strata::deepseek41
