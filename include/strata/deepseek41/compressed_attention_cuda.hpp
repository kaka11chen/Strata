#pragma once

#include "strata/core/device.hpp"
#include <memory>
#include <span>

namespace strata::deepseek41 {
// Borrowed device buffers. KV is already compressed-RoPE/FP4-processed BF16.
// Indices are the 6d indexer's shifted group IDs, strictly increasing followed
// by -1 padding, with no future groups. Subsets are allowed; selection policy is
// the caller's responsibility (the indexer supplies min(512,groups) slots).
struct CompressedAttentionSelection {
    const uint16_t* kv = nullptr; // [groups,512]
    int groups = 0; // public ratio-2 forward: (start_pos + tokens) / 2, including future prefill groups
    const int32_t* indices = nullptr; // [tokens,count]
    int count = 0; // 0..min(512,groups); zero permits null indices
    int offset = 0; // initial prefill: tokens; decode: 128
};

// Fixed 1017 layer-2 core: window128 + selected compressed KV, one joint sparse
// softmax in 64-candidate chunks, sink and inverse output RoPE. cuBLAS FP32 dots;
// no TileLang reduction-order claim. Q/output projections are outside this API.
class CudaCompressedAttention {
  public:
    explicit CudaCompressedAttention(std::span<const float> sink, int device = 0);
    ~CudaCompressedAttention();
    // Device F32: projected query [tokens,64,512], normalized window KV
    // [tokens,512], output [tokens,64,512]. Inputs are pre-RoPE, rounded to BF16.
    // Finite inputs/intermediate dot products, |x|<=2^120; finite compressed KV.
    // Aligned, disjoint
    // buffers on the owning device, including no aliases into this instance.
    // First call is prefill at 0, then contiguous single-token decode, <=1048576
    // positions. One caller, one explicit stream; synchronous. No token-time
    // allocation. Bad arguments/indices preserve output, ring and position;
    // an execution failure poisons the instance. Caller owns external caches.
    void forward(const float* query, const float* kv, float* output, int tokens,
                 int start_pos, CompressedAttentionSelection selection, void* stream);
    void reset(void* stream);
    int next_position() const;
    size_t device_bytes() const; // owned arena only, excludes cuBLAS/context/external KV
    const uint16_t* kv_cache() const; // borrowed read-only BF16 ring [128,512]
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class CudaSource20Attention; // typed ratio-1 owner reuses the private arithmetic
    friend class CudaSharedAttention;
    friend class CudaSourceAttention; // preserves original prefill grouping with one-query scratch
};
} // namespace strata::deepseek41
