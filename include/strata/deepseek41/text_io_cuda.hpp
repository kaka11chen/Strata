#pragma once

#include "strata/deepseek41/text_io_weights.hpp"
#include "strata/core/device.hpp"
#include <memory>

namespace strata::deepseek41 {
// Fixed 1017 text boundaries, batch size one. Embedding stays mapped on the host;
// only requested rows enter bounded staging. The untied head is copied to device
// in original BF16 storage; Strata's existing GEMV produces FP32 logits.
// The mapped embedding must outlive this object. Single caller, one explicit
// stream, synchronous calls. No tokenization, sampling, KV state or generation.
class CudaTextIo {
  public:
    CudaTextIo(const TextIoWeights& weights, int prefill_capacity, int device = 0);
    // Host original token IDs [T] -> device BF16-valued F32 [T,4,hidden], with
    // four identical streams and F32 initial coefficients [T,4] = [1,0,0,0].
    // All IDs and selected row values are checked before writing caller output.
    void embed(std::span<const int32_t> tokens, float* residual, float* initial_pre, void* stream);
    // Device residual [T,4,hidden] and delayed coefficients [T,4] from the final
    // backbone layer. Select the LAST token, round residual to BF16, apply the
    // existing weighted hc_pre, RMSNorm(eps=1e-20), BF16 rounding and head GEMV.
    // Device output [vocab] remains F32. Inputs must be finite with finite F32
    // intermediates; buffers must be aligned, disjoint and outside owned storage.
    void head(const float* residual, const float* final_pre, int tokens, float* logits, void* stream);
    int hidden_size() const { return hidden_; }
    int vocab_size() const { return vocab_; }
    int prefill_capacity() const { return capacity_; }
    size_t device_bytes() const { return arena_->capacity(); }
    size_t host_staging_bytes() const { return (rows_.capacity() + initial_.capacity()) * sizeof(float); }
    // Invalid arguments leave caller output unchanged. Execution failure poisons
    // this instance: discard that call's output and reconstruct before reuse.
  private:
    int hidden_, vocab_, capacity_, device_;
    std::span<const uint16_t> embedding_;
    std::vector<float> rows_, initial_;
    std::unique_ptr<core::DeviceArena> arena_;
    uint16_t* head_ = nullptr;
    float *norm_ = nullptr, *residual_ = nullptr, *normalized_ = nullptr;
    void* stream_ = nullptr;
    bool failed_ = false;
    void check_stream(void* stream) const;
    void check_external(const void* pointer, size_t bytes) const;
};
} // namespace strata::deepseek41
