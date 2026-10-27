#pragma once

#include "strata/deepseek41/attention_weights.hpp"
#include "strata/deepseek41/compressed_attention_cuda.hpp"
#include "strata/deepseek41/compressed_kv_cuda.hpp"
#include "strata/deepseek41/index_key_cuda.hpp"
#include "strata/deepseek41/index_query_cuda.hpp"

namespace strata::deepseek41 {
// An opaque borrowed publication for one completed source-layer call. It does not
// keep GPU storage alive. Copies expire together when the source advances,
// resets, fails or is destroyed. A rejected source call preserves them.
class SharedAttentionBatch {
  public:
    SharedAttentionBatch() = default; // invalid until obtained from the source
    int source_layer() const; // valid current publication only; rejects expired/stale batches
  private:
    struct State;
    std::weak_ptr<State> state_;
    uint64_t publication_ = 0;
    SharedAttentionBatch(const std::shared_ptr<State>& state, uint64_t publication)
        : state_(state), publication_(publication) {}
    friend class CudaSourceAttention;
    friend class CudaSharedAttention;
    friend class Source20AttentionBatch;
    friend class CudaSource20Attention;
};
// Complete fixed 1017 source-2/source-8/source-14 Attention.forward, batch 1: Q/KV/output projections,
// ratio-2 compressor, owned compressed/index caches and causal indexer. Reuses
// the layer-0 projection implementation and the confirmed 6a-6e components.
// mHC, residual joins and MoE are outside this class. Optional publication lets
// layers 3..7 borrow source 2; layers 9..13 borrow source 8. Source 14 publishes
// its own identity; consumers 15..19 remain a later integration gate.
class CudaSourceAttention {
  public:
    CudaSourceAttention(const AttentionWeights& attention, const CompressorWeights& compressor,
                        const IndexKeyWeights& key, const IndexQueryWeights& query,
                        int capacity_groups, int device = 0, int shared_prefill_capacity = 0, int layer = 2);
    // shared_prefill_capacity=0 disables publication (the step-6f allocation).
    // Otherwise reserve 4 * shared_prefill_capacity * min(512,capacity_groups)
    // bytes, aligned to 256, for the LAST call's indices. Initial prefill must
    // fit this explicit limit, 1..1048576. No per-call allocation or KV copy.
    // Consume every dependent layer before advancing/resetting this source.
    SharedAttentionBatch shared_batch() const;
    ~CudaSourceAttention();
    // Aligned, disjoint F32 device input/output [tokens,5120], external to this
    // instance's scratch and exposed caches. Component finite-input/intermediate
    // range preconditions apply. First call: prefill at 0, then contiguous
    // single-token decode, <=1048576 positions. One explicit stream and caller.
    // All cache capacities are checked BEFORE any component advances. Rejected
    // sequence, capacity, pointer or stream arguments preserve output and state.
    // A full cache permits one incomplete group token, but no further emission.
    // Execution failures poison the complete instance; reset is not recovery.
    // Synchronous, fixed scratch. Prefill first fills the compressed/index caches,
    // then projects/attends one query at a time with the original prefill layout.
    void forward(const float* input, float* output, int tokens, int start_pos, void* stream);
    void reset(void* stream); // same stream; clears all history and partial group
    int next_position() const;
    int capacity() const; // groups, fixed at startup; 1..524288
    int compressed_groups() const;
    size_t device_bytes() const; // all owned weights/caches/scratch, excludes cuBLAS/context/caller buffers
    const uint16_t* kv_cache() const; // read-only BF16 [128,512] window ring
    const uint16_t* compressed_kv_cache() const; // read-only BF16 [compressed_groups(),512]
    const uint16_t* index_key_cache() const; // read-only BF16 [compressed_groups(),128]
  private:
    friend class CudaSourceDecoder;
    friend class CudaLayer14Decoder;
    // Internal two-pass orchestration, shared with the decoder. No public partial
    // calls: only finish publishes, after the enclosing decoder also succeeds.
    void validate_forward(int tokens, int start_pos, void* stream) const;
    void check_external(const void* pointer, size_t bytes) const;
    void begin_forward(void* stream);
    void compress_token(const float* input, int position, void* stream);
    void attend_token(const float* input, float* output, int token, int tokens, int start_pos, void* stream);
    void finish_forward(int tokens, int start_pos, void* stream);
    void fail_forward(void* stream) noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Compatibility name; sources 8/14 always require an explicit layer.
using CudaLayer2Attention = CudaSourceAttention;
} // namespace strata::deepseek41
