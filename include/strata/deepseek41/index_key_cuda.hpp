#pragma once

#include "strata/deepseek41/compressor_cuda.hpp"
#include "strata/deepseek41/index_key_weights.hpp"

namespace strata::deepseek41 {
// Fixed 1017 index keys: normalized F32 [groups,128] rounded to BF16, adjacent
// RoPE on the final 64 channels, then FP4/E2M1 with group-32 E8M0 scales restored
// to BF16 [groups,128]. Shares the compressed KV RoPE. Input is unchanged.
// Finite |x|<=2^120 required. Nonempty groups at even positions 0..1048574.
// Aligned, disjoint device buffers and an explicit owning-device stream required.
// Asynchronous; caller synchronizes before reuse/destruction. No allocation.
void postprocess_index_keys(const float* normalized, uint16_t* output, int groups,
                            int first_position, void* stream);

// Ratio-2 source key projection + normalization + post-processing and bounded storage.
// Owns original BF16 weights, one-group scratch and BF16 cache (not packed FP4).
// No query projection, selection or shared-layer ownership is implemented here.
class CudaIndexKeyCache {
  public:
    CudaIndexKeyCache(const IndexKeyWeights& weights, int capacity_groups, int device = 0);
    ~CudaIndexKeyCache();
    // Input is the ORIGINAL pre-RoPE compressor latent, F32 [batch.count,512]
    // rounded to BF16. Finite input/intermediates and |normalized key|<=2^120
    // are caller preconditions. The input is never modified.
    // Nonempty batches start at 2*size(); empty batches are {0,-1}, ignoring input.
    // Whole-batch capacity check occurs before writing; no eviction. Orchestration
    // must preflight BOTH caches before advancing the compressor (no joint rollback).
    // Synchronous, single caller and one explicit stream. Invalid arguments leave
    // state intact; execution failure requires reconstruction, not reset.
    void append(const float* latent, CompressedBatch batch, void* stream);
    void reset(void* stream); // clear cache/size, retaining weights and allocation
    int size() const;
    int capacity() const;
    size_t device_bytes() const;
    const uint16_t* data() const; // read-only device [size(),128], until reset/destruction
  private:
    friend class CudaSource20IndexKeyCache;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::deepseek41
