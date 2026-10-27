#pragma once

#include "strata/deepseek41/source20_index_query_cuda.hpp"
#include <utility>

namespace strata::deepseek41 {
// Source-24/source-28/source-32/source-36 query projection and top-512 inside source-20 candidate blocks.
// Reuses CudaIndexQuery's ratio-1 arithmetic and CUB selection. Owns no keys,
// candidates or sequence history. Same finite-intermediate contract as it.
class CudaCandidateIndexQuery {
  public:
    CudaCandidateIndexQuery(const CandidateIndexQueryWeights& weights, int capacity_tokens, int device = 0);
    ~CudaCandidateIndexQuery();
    // candidates is one source-20 row: sorted unique visible block IDs, newest
    // block included, then -1 padding. Finite source scores imply exactly
    // min(2048,ceil(visible_keys/8)) selected blocks. Shape/content are checked
    // before changing the previous result. keys and candidates remain unchanged.
    // Explicit single stream/device; synchronous, no per-call allocation.
    // Execution failures require reconstruction. Views expire at the next call.
    IndexSelection forward(const float* x, const float* qr, const uint16_t* keys,
                           const Source20Batch& batch, int query, Source20CandidateSelection candidates,
                           int offset, void* stream);
    int capacity() const;
    size_t device_bytes() const;
    size_t sort_workspace_bytes() const;
    const uint16_t* query() const;
    const float* head_weights() const;
  private:
    friend class CudaSharedAttention;
    std::pair<const void*, size_t> storage() const;
    std::unique_ptr<CudaIndexQuery::Impl> impl_;
};
} // namespace strata::deepseek41
