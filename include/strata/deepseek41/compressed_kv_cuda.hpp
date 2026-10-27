#pragma once

#include "strata/deepseek41/compressor_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 compressed KV: adjacent-pair YaRN RoPE on the final 64 channels,
// then FP4/E2M1 with E4M3 scales per 16 values, restored to BF16 like the official
// cache. Input [groups,512] is F32 rounded to BF16, output is BF16 [groups,512].
// Leaves pre-RoPE input intact for the indexer. Finite input |x|<=1024 required.
// first_position must be even, >=0; groups>0; final group position<=1048574.
// Disjoint aligned device buffers and an explicit owning-device stream required.
// Asynchronous; caller synchronizes before reuse/destruction. No allocation.
void postprocess_compressed_kv(const float* latent, uint16_t* output, int groups,
                              int first_position, void* stream);

// Append-only cache of completed ratio-2 groups. Capacity is explicit at startup;
// reaching it rejects the entire append before writing. History is never evicted.
// Stores BF16 values after an FP4 round trip, not packed FP4 bytes.
class CudaCompressedKvCache {
  public:
    explicit CudaCompressedKvCache(int capacity_groups, int device = 0);
    ~CudaCompressedKvCache();
    // Accepts CudaCompressor's emission descriptor. Nonempty batches start at
    // position 2*size(); empty batches must have first_position=-1, ignore latent.
    // Synchronous, one caller and one explicit stream per instance. Never changes
    // latent. Invalid arguments leave state unchanged; execution failures poison
    // the cache and require reconstruction (reset cannot recover a failed cache).
    void append(const float* latent, CompressedBatch batch, void* stream);
    void reset(void* stream); // clears storage and size, retaining allocation
    int size() const;
    int capacity() const;
    size_t device_bytes() const;
    const uint16_t* data() const; // read-only device [size(),512]; valid until reset/destruction
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::deepseek41
