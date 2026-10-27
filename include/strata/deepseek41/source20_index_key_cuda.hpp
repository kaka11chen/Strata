#pragma once

#include "strata/deepseek41/index_key_cuda.hpp"
#include "strata/deepseek41/source20_contract.hpp"

namespace strata::deepseek41 {
// Same RoPE/FP4/E8M0 math as ratio-2 index keys, at every token position.
// F32 [tokens,128] -> BF16 [tokens,128]. Explicit stream, disjoint aligned
// device buffers; asynchronous and allocation-free. Input is unchanged.
// Finite |x|<=2^120; 1..1048576 tokens, first_position+tokens<=1048576.
void postprocess_source20_index_keys(const float* normalized, uint16_t* output,
                                    int tokens, int first_position, void* stream);

// Original source-20 weights, shared projection/RMSNorm/quantization pipeline,
// bounded BF16 key cache. No queries, selection or shared-state publication.
class CudaSource20IndexKeyCache {
  public:
    CudaSource20IndexKeyCache(const Source20IndexKeyWeights& weights, int capacity_tokens, int device = 0);
    ~CudaSource20IndexKeyCache();
    // ORIGINAL pre-RoPE compressor latent: F32 [batch.latent_count(),512],
    // BF16-valued. Finite input/intermediates and |normalized key|<=2^120.
    // Requires batch.first_position()==size(); whole batch must fit. Preserves
    // latent. Preflight BOTH caches before advancing compressor; no joint rollback.
    // Synchronous, single caller/device/explicit stream. Argument errors preserve
    // state. Execution failures poison the instance and require reconstruction.
    void append(const float* latent, const Source20Batch& batch, void* stream);
    void reset(void* stream); // zeros cache, retains weights, allocation and stream
    int size() const;
    int capacity() const;
    size_t device_bytes() const;
    const uint16_t* data() const; // device [size(),128], until reset/destruction
    const float* last_normalized() const; // diagnostic [128], null until append/after reset
  private:
    std::unique_ptr<CudaIndexKeyCache::Impl> impl_;
};
} // namespace strata::deepseek41
