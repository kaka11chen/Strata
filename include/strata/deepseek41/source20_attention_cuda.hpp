#pragma once

#include "strata/deepseek41/layer2_attention_cuda.hpp"
#include "strata/deepseek41/source20_compressor_cuda.hpp"
#include "strata/deepseek41/source20_kv_cuda.hpp"
#include "strata/deepseek41/source20_index_key_cuda.hpp"
#include "strata/deepseek41/source20_index_query_cuda.hpp"

namespace strata::deepseek41 {
// Borrowed read-only device views, valid only until the owner advances, resets,
// fails or is destroyed. No KV/key copy. Candidate IDs have no attention offset;
// a partial selected block includes future positions, so causality is separate.
struct Source20AttentionView {
    int start, tokens, key_count, index_count, offset, candidate_stride;
    uint64_t sequence; // changes on reset, independent of call position
    const uint16_t* kv; // [key_count,512]
    const uint16_t* keys; // [key_count,128]
    const int32_t* indices; // [tokens,index_count], shifted, sorted then -1
    const int32_t* candidates; // [tokens,candidate_stride], block IDs then -1
};
// Distinct from ratio-2 SharedAttentionBatch. Does not keep GPU storage alive.
// Obtain a fresh view for each use; caching a raw view does not extend lifetime.
// One caller/device/stream; concurrent owner mutation is unsupported.
class Source20AttentionBatch {
  public:
    Source20AttentionBatch() = default;
    Source20AttentionView view(int expected_start, int expected_tokens, void* stream) const;
  private:
    std::weak_ptr<SharedAttentionBatch::State> state_;
    uint64_t publication_ = 0;
    Source20AttentionBatch(const std::shared_ptr<SharedAttentionBatch::State>& state, uint64_t publication)
        : state_(state), publication_(publication) {}
    friend class CudaSource20Attention;
    friend class CudaSharedAttention;
    friend class CudaSharedDecoder;
    SharedAttentionBatch attention_batch() const; // private borrowed view; no KV/index copy
};
// Complete original source-20 attention, batch one. Reuses projection and
// sparse-attention arithmetic plus validated ratio-1 compressor/cache/indexer.
// No decoder, residual, MoE or downstream candidate consumer in this class.
class CudaSource20Attention {
  public:
    CudaSource20Attention(const AttentionWeights& attention, const Source20CompressorWeights& compressor,
                         const Source20IndexKeyWeights& key, const Source20IndexQueryWeights& query,
                         int context_capacity, int prefill_capacity, int device = 0);
    ~CudaSource20Attention();
    // Aligned disjoint external F32 [tokens,5120], finite component arithmetic.
    // Initial prefill then contiguous single-token decode. Both capacities are
    // explicit in tokens, 1<=prefill<=context<=1048576. All argument checks precede
    // producer advance/publication invalidation; rejection preserves state/output.
    // Two passes retain whole-prefill semantics with one-query compute scratch.
    // Startup allocates indices/candidates for prefill_capacity rows, no per-call
    // GPU allocation. Synchronous, one explicit stream. Execution failure poisons
    // the complete owner; reset does not recover it. Consume publications before
    // advancing or resetting this owner.
    void forward(const float* input, float* output, int tokens, int start, void* stream);
    void reset(void* stream);
    Source20AttentionBatch shared_batch() const;
    int next_position() const;
    int capacity() const;
    int size() const; // one compressed KV and index key per token
    size_t device_bytes() const; // all owned arenas; excludes cuBLAS/runtime/caller
    const uint16_t* kv_cache() const; // window ring [128,512]
    const uint16_t* compressed_kv_cache() const;
    const uint16_t* index_key_cache() const;
  private:
    friend class CudaSource20Decoder;
    // Decoder-owned two-pass transaction. Only finish publishes, after every
    // residual/MoE succeeds. Partial operations stay outside the public API.
    void validate_forward(int tokens, int start, void* stream) const;
    void check_external(const void* pointer, size_t bytes) const;
    void begin_forward(void* stream);
    void compress_token(const float* input, int position, void* stream);
    void attend_token(const float* input, float* output, int token, int tokens, int start, void* stream);
    void finish_forward(int tokens, int start, void* stream);
    void fail_forward(void* stream) noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::deepseek41
