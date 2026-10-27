#pragma once

#include "strata/deepseek41/artifact.hpp"

namespace strata::deepseek41 {

// Original source-20 BF16 projection and RMSNorm weights; no gate or pair state.
// Borrows the mapped model, which must outlive these views. Ratio-1 projection
// returns BF16 BEFORE RMSNorm; index keys consume the result BEFORE KV RoPE.
struct Source20CompressorWeights {
    std::span<const uint16_t> kv;   // row-major [512,5120]
    std::span<const uint16_t> norm; // [512]
};
Source20CompressorWeights read_source20_compressor_weights(const GgufModel& model,
                                                         const Geometry& geometry);

// A validated call description, not a cache, publication or forward operation.
// After reset expected_next is zero. Later calls are contiguous single-token
// decode. The caller advances expected_next only after a successful forward.
class Source20Batch {
  public:
    static constexpr int layer = 20;
    static constexpr int max_context = 1048576;
    static constexpr int block_size = 8;
    static constexpr int topk_blocks = 2048;

    Source20Batch(int start, int tokens, int expected_next, int context_capacity, int prefill_capacity);
    int first_position() const { return start_; }
    int latent_count() const { return tokens_; }
    int key_count() const { return start_ + tokens_; }
    int position(int query) const;
    int visible_keys(int query) const { return position(query) + 1; }
    int newest_block(int query) const { return (visible_keys(query) - 1) / block_size; }
    int candidate_stride() const;
    size_t candidate_entries() const { return size_t(tokens_) * size_t(candidate_stride()); }

  private:
    int start_, tokens_;
};

// Host-side fixture/debug validation only; no candidate selection or CPU
// inference fallback. Each query stores original block IDs sorted ascending,
// followed by -1 padding, at candidate_stride(). The corresponding bool mask
// includes ALL positions of selected blocks, including a partial block's future
// positions: consumers must ALSO apply visible_keys(), before their top-512.
// Ties at the cutoff permit any valid tied subset, matching the reference.
//
// causal_logits has key_count() entries: finite or -inf before visible_keys(),
// exactly -inf afterwards. The newest block is always kept. Throws on malformed
// inputs, wrong cardinality or selection below a higher-scoring omitted block.
void validate_source20_candidates(const Source20Batch& batch, int query,
                                  std::span<const float> causal_logits,
                                  std::span<const int32_t> block_ids);

} // namespace strata::deepseek41
