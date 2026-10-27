#pragma once

#include "strata/deepseek41/index_query_weights.hpp"
#include "strata/core/device.hpp"

#include <memory>

namespace strata::deepseek41 {
// One token's 32 query heads. BF16 boundary, compressed YaRN RoPE at this token's
// position (including odd positions), group-32 FP4/E8M0 round trip to BF16.
// Input F32 [32,128], output BF16 [32,128], disjoint aligned device buffers.
// Finite |input|<=2^120, position 0..1048575, explicit owning-device stream.
// Async, no allocation, does not modify input.
void postprocess_index_query(const float* projected, uint16_t* output, int position, void* stream);

struct IndexSelection {
    const int32_t* indices; // [count], valid IDs ascending + offset, then -1 padding
    const float* scores;   // [key_count], BF16-valued F32; masked future groups are -inf
    int count;            // min(512,key_count), including unreachable padding
};
// Query projection, weighted scores and causal top-512 for ratio-2 sources. Consumes the
// confirmed key cache without modifying it. No shared-layer/candidate-block path.
class CudaIndexQuery {
  public:
    CudaIndexQuery(const IndexQueryWeights& weights, int capacity_groups, int device = 0);
    ~CudaIndexQuery();
    // x F32[5120], qr F32[1280] (attention's normalized low-rank query), both
    // rounded to BF16; keys BF16[key_count,128]. Finite intermediate arithmetic
    // required. All inputs external to this object's arena and left unchanged.
    // key_count >= floor((position+1)/2). For prefill pass the WHOLE prefill's key
    // count to every query; future groups are masked. Decode normally uses equality.
    // key_count==0 permits null keys. offset>=0 must keep selected IDs within int32.
    // Ties at the cutoff choose the lowest original group ID deterministically;
    // torch.topk does not specify tied indices. Returns position-sorted IDs.
    // Synchronous, one explicit stream and caller. Invalid arguments preserve the
    // last result; execution failure poisons the object and requires reconstruction.
    // Results/diagnostic views remain valid until the next successful call/destruction.
    IndexSelection forward(const float* x, const float* qr, const uint16_t* keys,
                           int key_count, int position, int offset, void* stream);
    int capacity() const;
    size_t device_bytes() const; // owned weights, scratch and preallocated CUB workspace
    size_t sort_workspace_bytes() const;
    const uint16_t* query() const; // last quantized query [32,128]
    const float* head_weights() const; // last scaled BF16-valued weights [32]
  private:
    friend class CudaSource20IndexQuery;
    friend class CudaCandidateIndexQuery;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::deepseek41
