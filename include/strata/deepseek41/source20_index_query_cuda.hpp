#pragma once

#include "strata/deepseek41/index_query_cuda.hpp"
#include "strata/deepseek41/source20_contract.hpp"

namespace strata::deepseek41 {
struct Source20CandidateSelection {
    const int32_t* blocks; // original block IDs ascending, then -1 padding
    int count;            // min(2048,ceil(batch.key_count()/8)), includes padding
};

// One query's candidate blocks. Owns bounded scratch and CUB workspace, not
// sequence history. No allocation per call. Shared publication is separate.
class CudaSource20Candidates {
  public:
    explicit CudaSource20Candidates(int capacity_tokens, int device = 0);
    ~CudaSource20Candidates();
    // External F32 scores[batch.key_count()], finite or -inf; no NaN/+inf.
    // Already causal scores are accepted; future positions are also masked here.
    // Block score=max of 8 positions; newest block pinned; ties choose low IDs.
    // A selected partial block includes future positions, so downstream users
    // must still apply batch.visible_keys(query). Input remains unchanged.
    // Synchronous, one explicit stream/device/caller; invalid arguments preserve
    // last output, execution failure requires reconstruction. View lasts until
    // next successful call/destruction; a later owner must copy rows for prefill.
    Source20CandidateSelection forward(const float* scores, const Source20Batch& batch,
                                       int query, void* stream);
    int capacity() const;
    size_t device_bytes() const;
    size_t sort_workspace_bytes() const;
  private:
    friend class CudaSource20IndexQuery;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Source20IndexSelection {
    IndexSelection index; // unfiltered causal top-512; source 20 does not consume candidates
    Source20CandidateSelection candidates; // for later index sources, without offset
};
class CudaSource20IndexQuery {
  public:
    CudaSource20IndexQuery(const Source20IndexQueryWeights& weights, int capacity_tokens, int device = 0);
    ~CudaSource20IndexQuery();
    // Shared projection/rounding/scoring pipeline; head-weight GEMV uses the
    // existing 32-lane reduction to avoid serial cancellation. One key per token. x F32[5120], qr F32[1280], keys BF16[batch.key_count(),128].
    // Inputs must be aligned external device buffers with finite arithmetic;
    // unchanged by forward. offset affects top-512 indices only, must fit int32.
    // For prefill pass the whole batch/key history for every query row. No key
    // generation, sequence-state ownership, candidate filtering or publication.
    // Same stream/failure/view lifetime contract as CudaSource20Candidates.
    Source20IndexSelection forward(const float* x, const float* qr, const uint16_t* keys,
                                   const Source20Batch& batch, int query, int offset, void* stream);
    int capacity() const;
    size_t device_bytes() const; // both computation and candidate scratch/workspaces
    size_t sort_workspace_bytes() const;
    const uint16_t* query() const;
    const float* head_weights() const;
  private:
    std::unique_ptr<CudaIndexQuery::Impl> impl_;
    CudaSource20Candidates candidates_;
};
} // namespace strata::deepseek41
