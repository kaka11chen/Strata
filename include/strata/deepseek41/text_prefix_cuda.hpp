#pragma once

#include "strata/deepseek41/layer1_decoder_cuda.hpp"
#include "strata/deepseek41/layer14_decoder_cuda.hpp"
#include "strata/deepseek41/layer2_decoder_cuda.hpp"
#include "strata/deepseek41/layer3_decoder_cuda.hpp"
#include "strata/deepseek41/source20_decoder_cuda.hpp"

namespace strata::deepseek41 {
struct TextPrefixWeights {
    std::vector<DecoderWeights> layers = std::vector<DecoderWeights>(4); // exactly 4, 8, 14, 15, 20, 24, 28, 32, 36 or 40, original order
    Layer1EngramWeights engram;
    CompressorWeights compressor;
    IndexKeyWeights key;
    IndexQueryWeights query;
    CompressorWeights compressor8; // required for prefixes containing original layer 8
    IndexKeyWeights key8;
    IndexQueryWeights query8;
    EngramWeights engram14; // required for prefixes containing original layer 14
    CompressorWeights compressor14;
    IndexKeyWeights key14;
    IndexQueryWeights query14;
    Source20CompressorWeights compressor20; // required for prefixes of at least 24 layers
    Source20IndexKeyWeights key20;
    Source20IndexQueryWeights query20;
    CandidateIndexQueryWeights query24; // original layer 24; 28-element prefix ends at layer 27
    CandidateIndexQueryWeights query28; // original layer 28; 32-element prefix ends at layer 31
    CandidateIndexQueryWeights query32; // original layer 32; 36-element prefix ends at layer 35
    CandidateIndexQueryWeights query36; // original layer 36; 40-element prefix ends at layer 39
};
// Fixed 1017 text prefix of 4, 8, 14, 15, 20, 24, 28, 32, 36 or 40 layers. Owns decoders and two bounded
// residual/coefficient buffers. ExpertSource serves original IDs in all active
// layers and must outlive this object. No embedding, output head or generation.
// A 40-layer instance covers the backbone only. DSpark reads pre-layer stream
// means at 37/38/39; capturing them and executing MTP layers 40..42 are separate
// from this final-residual/delayed-coefficient interface.
class CudaTextPrefix {
  public:
    CudaTextPrefix(const Geometry& geometry, const TextPrefixWeights& weights,
                   std::span<const uint32_t> token_map, const std::string& table_path,
                   core::ExpertSource& source, int slots_per_layer, int prefill_capacity,
                   int capacity_groups, const EngramIoOptions& io = {}, int device = 0,
                   std::span<const uint32_t> token_map14 = {}, const std::string& table_path14 = {});
    // The optional final arguments are required for 15, 20, 24, 28, 32, 36 or 40 layers: an independent
    // original layer-14 Engram table and token map. They never alias layer 1's
    // hash state or row cache. Existing calls retain the four-layer default.
    // Host original tokens [T]; device F32 residual/output [T,4,5120] and
    // incoming/next_pre [T,4]. Coefficients always move to the same token in the
    // next layer. Optional host routes is layer-major [layer_count(),T]. All caller buffers
    // aligned/disjoint, outside owned storage. Existing finite-input contracts apply.
    // First call: prefill up to prefill_capacity. Later: contiguous single-token
    // decode. Cache capacity and every token are checked BEFORE any layer advances.
    // Single caller, synchronous, one explicit stream. Execution/I/O failure
    // poisons the entire prefix; discard that call's output and reconstruct.
    void forward(std::span<const int32_t> tokens, const float* input, const float* incoming_pre,
                 float* output, float* next_pre, int start_pos, void* stream, std::span<Route> routes = {});
    // Reset all histories/publication bindings together, retaining weight caches
    // and counters. Same stream; cannot recover from an execution failure.
    void reset(void* stream);
    int next_position() const { return next_; } // only fully completed prefix calls
    int layer_count() const { return layer_count_; }
    int prefill_capacity() const { return prefill_capacity_; }
    // Cache entries: pairs of tokens for sources 2/8/14, single tokens for 20.
    // Source 20 reserves min(1048576, 2*capacity_groups+1) entries, matching the
    // earlier sources including their final incomplete pair.
    int capacity(int source_layer = 2) const;
    int compressed_groups(int source_layer = 2) const;
    std::vector<CacheStats> stats() const;
    std::array<ngram::ReaderStats, 2> engram_stats(int layer = 1) const;
    size_t device_bytes() const;
    size_t workspace_bytes() const { return arena_.capacity(); }
    // Borrowed read-only state; no shared publication escapes this owner.
    const uint16_t* kv_cache(int layer) const;
    // Source must be active: 2, 8 (>=14 layers), 14 (>=15 layers), or 20 (>=24 layers).
    const uint16_t* compressed_kv_cache(int source_layer = 2) const;
    const uint16_t* index_key_cache(int source_layer = 2) const;
  private:
    int device_, prefill_capacity_, layer_count_, next_ = 0;
    void* stream_ = nullptr;
    bool failed_ = false;
    CudaDecoder layer0_;
    CudaLayer1Decoder layer1_;
    CudaLayer2Decoder layer2_;
    std::unique_ptr<CudaSourceDecoder> layer8_;
    std::unique_ptr<CudaLayer14Decoder> layer14_;
    std::unique_ptr<CudaSource20Decoder> layer20_;
    std::vector<std::unique_ptr<CudaSharedDecoder>> consumers_; // indexed by original layer; destroyed before sources
    core::DeviceArena arena_;
    std::array<float*, 2> residual_, coefficients_;
    void check_stream(void* stream) const;
    const CudaSourceDecoder& source_decoder(int layer) const;
};
} // namespace strata::deepseek41
