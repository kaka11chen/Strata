#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/attention_weights.hpp"

#include <memory>

namespace strata::deepseek41 {
// The fixed 1017 pure-SWA core: 64 queries x 512, one shared KV x 512,
// adjacent-pair RoPE on the final 64 values, FP8 KV, sink, inverse output RoPE.
// Uses cuBLAS FP32 dots, not the official TileLang reduction order.
class CudaWindowAttention {
  public:
    explicit CudaWindowAttention(std::span<const float> sink, int device = 0);
    ~CudaWindowAttention();
    // Device F32 arrays [tokens,64,512], [tokens,512], [tokens,64,512].
    // Values are rounded to BF16; must be finite, |x| <= 2^120. Buffers must be
    // disjoint and 4-byte aligned. Explicit owning-device stream, one caller.
    // First call: start_pos=0, tokens>=1 (prefill). Later calls: tokens=1,
    // start_pos=next_position(). Total positions cannot exceed 1048576.
    // Synchronous; no per-token allocation. An execution failure poisons state.
    void forward(const float* query, const float* kv, float* output, int tokens, int start_pos, void* stream);
    void reset(void* stream); // same stream; clears the ring and position
    int next_position() const;
    size_t device_bytes() const; // owned arena, excludes cuBLAS/context
    const uint16_t* kv_cache() const; // borrowed device BF16 [128,512], read-only
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class CudaAttention;
};

// Complete attention projections around the window core, fixed to layer 0 or 1.
// FP8 weights expand losslessly to BF16 once at startup. Matrix products and
// RMSNorm reuse Strata. No compressed KV, indexer, mHC or residual connection.
class CudaAttention {
  public:
    // Layer 1 is explicit; existing callers retain the layer-0 profile check.
    explicit CudaAttention(const AttentionWeights& weights, int device = 0, int layer = 0);
    ~CudaAttention();
    // Same sequence/stream contract as CudaWindowAttention. Disjoint F32 device
    // input/output [tokens,5120], finite BF16-representable inputs with |x|<=2^120.
    // Prefill is serialized using bounded scratch; not a throughput optimization.
    void forward(const float* input, float* output, int tokens, int start_pos, void* stream);
    void reset(void* stream);
    int next_position() const;
    size_t weight_bytes() const;
    size_t device_bytes() const;
    const uint16_t* kv_cache() const;
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    // Decoder streams a prefill through mHC one token at a time while retaining
    // the original prefill candidate grouping. Only its validated outer call
    // supplies prefill_width (min(initial_tokens,128), or zero during decode).
    void forward_token(const float* input, float* output, int position, int prefill_width, void* stream);
    friend class CudaDecoder;
    friend class CudaLayer1Decoder;
};
} // namespace strata::deepseek41
