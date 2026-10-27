#pragma once

#include "strata/core/device.hpp"
#include "strata/deepseek41/source20_contract.hpp"

#include <memory>

namespace strata::deepseek41 {

// Source-20 ratio-1 version of the existing compressed-KV transform. Reuses
// adjacent-pair YaRN RoPE on the last 64 channels and group-16 FP4/E4M3 scales.
// Input F32 [tokens,512] rounds to BF16; output is restored BF16 [tokens,512],
// not packed FP4. Input remains intact for the later index-key path.
// Positions are first_position+i, including odd positions, up to 1048575.
// Requires tokens>0, finite |input|<=1024, aligned disjoint device buffers and
// an explicit owning-device stream. Asynchronous, with no allocation.
void postprocess_source20_kv(const float* latent, uint16_t* output, int tokens,
                             int first_position, void* stream);

// Append-only original source-20 cache. Explicit capacity in tokens; no eviction.
// Owns exactly capacity_tokens*512*sizeof(uint16_t) device bytes. Does not own
// the compressor, index keys, candidate selection or shared-state publication.
class CudaSource20KvCache {
  public:
    explicit CudaSource20KvCache(int capacity_tokens, int device = 0);
    ~CudaSource20KvCache();

    // Only ratio-1 descriptors: batch.first_position() must equal size().
    // Rejects the entire append before writes if it exceeds capacity. Callers
    // must preflight capacity before advancing their compressor; a rejected
    // append cannot roll back an already advanced producer.
    // Synchronous, one caller and one explicit stream per instance. Never
    // mutates latent. Invalid arguments preserve state; execution exceptions
    // require reconstruction. The low-level transform's input contract applies.
    void append(const float* latent, const Source20Batch& batch, void* stream);
    void reset(void* stream); // Same stream; clears all storage, retains allocation.
    int size() const;
    int capacity() const;
    size_t device_bytes() const; // Excludes caller buffers and CUDA runtime resources.
    const uint16_t* data() const; // Read-only device [size(),512]; invalidated by reset/destruction.

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::deepseek41
