#pragma once

#include "strata/deepseek41/source20_attention_cuda.hpp"
#include "strata/deepseek41/candidate_index_query_cuda.hpp"

namespace strata::deepseek41 {
// Fixed 1017 shared-KV layers through 39. Consumers reuse index sources 2/8/14/20/24/28/32/36. Own Q/KV/output
// projections, sink and window ring; borrow the original source's compressed KV
// and per-query indices. Ordinary consumers have no compressor, indexer or
// duplicate compressed/key cache. mHC/MoE/residual joins are outside this class.
class CudaSharedAttention {
  public:
    explicit CudaSharedAttention(const AttentionWeights& weights, int device = 0, int layer = 3);
    // Index sources 24/28/32/36 borrow source20 KV/keys/candidates, own only their query
    // indexer and bounded per-query index publication. Each following three-layer group consumes its own index source.
    CudaSharedAttention(const AttentionWeights& weights, const CandidateIndexQueryWeights& query,
                        int capacity_tokens, int prefill_capacity, int device = 0);
    ~CudaSharedAttention();
    SharedAttentionBatch shared_batch() const; // index sources 24/28/32/36 only, after the whole call succeeds
    // Aligned disjoint external F32 [tokens,5120]; component finite-input/range
    // preconditions apply. Publication must be current and match this exact
    // start/count, device and explicit stream. Consume initial prefill then
    // contiguous single-token decode, from one source and one source sequence.
    // Source reset requires consumer reset before starting the new sequence.
    // Bad arguments/stale publication preserve output and ring. GPU execution
    // failure poisons only this consumer; reset does not recover it. Synchronous,
    // fixed allocation; one caller serializes both source and consumer lifetimes.
    void forward(const float* input, float* output, int tokens, int start_pos,
                 const SharedAttentionBatch& shared, void* stream);
    void forward_source20(const float* input, float* output, int tokens, int start_pos,
                 const Source20AttentionBatch& shared, void* stream);
    void reset(void* stream); // only own ring/binding; never resets the source
    int next_position() const;
    size_t device_bytes() const; // own weights/scratch/window, excludes source
    const uint16_t* kv_cache() const; // read-only BF16 [128,512]
  private:
    friend class CudaSharedDecoder;
    // Decoder orchestration retains the whole publication's prefill layout and
    // advances the completed position only after every residual/MoE step succeeds.
    void validate_forward(int tokens, int start_pos, const SharedAttentionBatch& batch, void* stream) const;
    void check_external(const void* pointer, size_t bytes, const SharedAttentionBatch& batch) const;
    void begin_forward(void* stream);
    void attend_token(const float* input, float* output, int token, int tokens, int start_pos,
                      const SharedAttentionBatch& batch, void* stream);
    void finish_forward(int tokens, const SharedAttentionBatch& batch, void* stream);
    void fail_forward(void* stream) noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Preserve the step-6g entry point; default layer remains 3.
using CudaLayer3Attention = CudaSharedAttention;
} // namespace strata::deepseek41
