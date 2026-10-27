#pragma once

#include "strata/deepseek41/mhc_weights.hpp"
#include "strata/deepseek41/moe_cuda.hpp"

#include <memory>

namespace strata::deepseek41 {
struct MhcPrepared {
    const float* normalized; // [hidden], BF16-valued F32, consumed by this sublayer
    const float* next_pre;   // [4], F32, consumed by the NEXT sublayer
    const float* post;       // [4], F32
    const float* comb;       // [4,4], F32, source rows -> destination columns
};

// One token, fixed 1017 mHC. Owns startup weight copies and fixed scratch.
// FP32 projection uses cuBLAS; RMSNorm uses Strata's existing native QSA kernel.
// Single caller and one explicit stream per instance. Calls may enqueue consecutive
// prepare/finish pairs on that stream. Caller waits before destroying it or buffers.
class CudaMhc {
  public:
    CudaMhc(const Geometry& geometry, const MhcWeights& weights, int device = 0);
    ~CudaMhc();
    // Device residual [4,hidden] is rounded to BF16 into an owned snapshot.
    // incoming_pre [4] is F32; this call never substitutes its newly computed pre.
    // Inputs must be finite, with intermediate arithmetic in F32 range. Returned
    // device views are read-only and valid until the next prepare or destruction.
    MhcPrepared prepare(const float* residual, const float* incoming_pre, void* stream);
    // Pair with prepare on the same stream. BF16-valued sublayer [hidden] is
    // injected into the saved residual; output [4,hidden] is rounded to BF16.
    // Output must be disjoint from sublayer and the owned arena.
    void finish(const float* sublayer, float* output, void* stream);
    // Synchronous complete FFN sublayer, reusing the existing MoE. Device buffers:
    // residual/output [4,hidden], incoming/next_pre [4], all mutually disjoint.
    // The MoE must have the same width/device and its existing valid-input contract.
    // Any execution error invalidates this mHC instance as well as the MoE's own contract.
    Route forward_moe(CudaMoE& moe, const float* residual, const float* incoming_pre,
                      float* output, float* next_pre, void* stream);
    size_t device_bytes() const { return arena_->capacity(); } // excludes cuBLAS-owned resources
  private:
    friend class CudaSourceDecoder;
    friend class CudaSource20Decoder;
    friend class CudaLayer14Decoder;
    // Compressor prepass: reuse hc_pre + norm without calculating delayed mixes
    // or opening a prepare/finish pair. The view lasts until the next mHC call.
    const float* normalize_input(const float* residual, const float* incoming_pre, void* stream);
    void collapse_normalize(const float* incoming_pre, void* stream);
    struct Blas;
    int hidden_, device_;
    std::unique_ptr<core::DeviceArena> arena_;
    std::unique_ptr<Blas> blas_;
    float *projection_, *scale_, *base_, *norm_, *residual_, *mixes_, *coefficients_, *normalized_, *moe_output_;
    void* stream_ = nullptr;
    bool pending_ = false, failed_ = false;
    void check_stream(void* stream) const;
    void check_external(const void* pointer, size_t bytes) const;
};
} // namespace strata::deepseek41
