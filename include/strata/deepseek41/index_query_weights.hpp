#pragma once

#include "strata/deepseek41/shared_weights.hpp"

namespace strata::deepseek41 {
struct IndexQueryWeights {
    Fp8MatrixView query; // original E4M3/E8M0 block-32 [4096,1280]
    std::span<const uint16_t> head_weights; // original BF16 [32,5120]
    int layer = 2; // original source identity; owner orchestration rejects mixed layers
};
// Fixed 1017 source layers 2, 8 and 14 only. Borrows mapped tensors; the model must outlive views.
IndexQueryWeights read_index_query_weights(const GgufModel& model, const Geometry& geometry, int layer);
// Original source-20 query/head weights; distinct from ratio-2 source weights.
struct Source20IndexQueryWeights {
    Fp8MatrixView query; // E4M3/E8M0 block-32 [4096,1280]
    std::span<const uint16_t> head_weights; // BF16 [32,5120]
};
Source20IndexQueryWeights read_source20_index_query_weights(const GgufModel& model,
                                                           const Geometry& geometry);
// Candidate-consuming index source; currently original layers 24, 28, 32 and 36. It owns
// query/head weights while borrowing source-20 keys and candidate blocks.
struct CandidateIndexQueryWeights {
    Fp8MatrixView query;
    std::span<const uint16_t> head_weights;
    int layer = 24;
};
CandidateIndexQueryWeights read_candidate_index_query_weights(const GgufModel& model,
                                                              const Geometry& geometry, int layer = 24);
} // namespace strata::deepseek41
