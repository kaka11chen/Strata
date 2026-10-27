#include "strata/deepseek41/compressed_kv_cuda.hpp"
#include "strata/deepseek41/source20_kv_cuda.hpp"
#include "strata/deepseek41/source20_index_key_cuda.hpp"
#include "strata/deepseek41/index_query_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "compressed_rope.hpp"

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <algorithm>
#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int dim = 512, max_groups = 524288;
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
bool valid(const void* p, size_t bytes, size_t alignment) {
    return p && uintptr_t(p) % alignment == 0 && bytes <= UINTPTR_MAX - uintptr_t(p);
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) < uintptr_t(b) + bn && uintptr_t(b) < uintptr_t(a) + an;
}
struct Frequencies { float values[32]; };
const Frequencies& frequencies() {
    // Fixed 1017: base=160000, original context=65536, factor=16, beta=32/1.
    // Official corrected_dim yields the [15,25] transition band. Preserve its
    // F32 power/reciprocal/ramp/mix boundaries; no ggml magnitude correction.
    static const Frequencies result = [] {
        Frequencies f{};
        for (int i = 0; i < 32; ++i) {
            const float raw = 1.f / float(std::pow(160000., double(i) / 32.));
            const float smooth = 1.f - std::clamp(float(i - 15) / 10.f, 0.f, 1.f);
            const volatile float interpolated = (raw / 16.f) * (1.f - smooth);
            const volatile float extrapolated = raw * smooth;
            f.values[i] = interpolated + extrapolated;
        }
        return f;
    }();
    return result;
}
__device__ float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
__global__ void rotate_rows(float* values, int nheads, int position, bool inverse, Frequencies freq) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= nheads * 32) return;
    const int pair = i % 32, at = (i / 32) * dim + dim - 64 + pair * 2;
    float s, c; sincosf(__fmul_rn(float(position), freq.values[pair]), &s, &c);
    if (inverse) s = -s;
    const float a = values[at], b = values[at + 1];
    values[at] = bf(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s)));
    values[at + 1] = bf(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)));
}
template<int Width, bool E4M3>
__global__ void postprocess(const float* latent, uint16_t* output, int first_position, int position_step, Frequencies freq) {
    const int d = int(threadIdx.x), position = first_position + position_step * int(blockIdx.x);
    const size_t index = size_t(blockIdx.x) * Width + d;
    float x = bf(latent[index]);
    if (d >= Width - 64) {
        const int pair = (d - (Width - 64)) / 2;
        const size_t at = size_t(blockIdx.x) * Width + Width - 64 + pair * 2;
        const float a = bf(latent[at]), b = bf(latent[at + 1]);
        float s, c; sincosf(__fmul_rn(float(position), freq.values[pair]), &s, &c);
        x = d % 2 == 0 ? bf(__fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s))) :
                         bf(__fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)));
    }
    constexpr int group = E4M3 ? 16 : 32;
    float maximum = fmaxf(fabsf(x), E4M3 ? 6.f / 512.f : 6.f * 0x1p-126f);
    for (int mask = group / 2; mask; mask >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, mask, group));
    float scale;
    if constexpr (E4M3) {
        const auto code = __nv_cvt_float_to_fp8(__fdiv_rn(maximum, 6.f), __NV_SATFINITE, __NV_E4M3);
        scale = __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
    } else {
        // Official fast_round_scale: ceil to a power of two AFTER an F32
        // multiplication by 1/6. The floor above keeps this value normal.
        const unsigned bits = __float_as_uint(__fmul_rn(maximum, 1.f / 6.f));
        scale = __uint_as_float((bits + 0x7fffffu) & 0x7f800000u);
    }
    const float scaled = fmaxf(-6.f, fminf(6.f, __fdiv_rn(x, scale)));
    // CUDA supplies a software conversion on SM75; no FP4 Tensor Cores required.
    const auto code = __nv_cvt_float_to_fp4(scaled, __NV_E2M1, cudaRoundNearest);
    const float restored = __half2float(__nv_cvt_fp4_to_halfraw(code, __NV_E2M1)) * scale;
    output[index] = kernels::bf16_from_f32(restored);
}
} // namespace

namespace {
template<int Width, bool E4M3>
void launch_postprocess(const float* latent, uint16_t* output, int groups, int first_position, void* stream, int position_step = 2) {
    const int max_count = position_step == 1 ? Source20Batch::max_context : max_groups;
    if (!stream || groups <= 0 || groups > max_count || first_position < 0 || first_position >= 1048576 ||
        (position_step == 2 && (first_position % 2 || first_position > 2 * (max_groups - groups))) ||
        (position_step == 1 && first_position > Source20Batch::max_context - groups))
        throw std::invalid_argument("DeepSeek compressed post-processing has invalid group positions or count");
    const size_t in_bytes = size_t(groups) * Width * 4, out_bytes = size_t(groups) * Width * 2;
    if (!valid(latent, in_bytes, 4) || !valid(output, out_bytes, 2) || overlaps(latent, in_bytes, output, out_bytes))
        throw std::invalid_argument("DeepSeek compressed post-processing needs aligned disjoint buffers");
    postprocess<Width, E4M3><<<groups, Width, 0, static_cast<cudaStream_t>(stream)>>>(latent, output, first_position, position_step, frequencies());
    ck(cudaGetLastError(), "DeepSeek compressed post-processing");
}
} // namespace
void detail::rotate_compressed(float* values, int nheads, int position, bool inverse, void* stream) {
    rotate_rows<<<(nheads * 32 + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        values, nheads, position, inverse, frequencies());
    ck(cudaGetLastError(), "DeepSeek compressed attention RoPE");
}
void postprocess_compressed_kv(const float* latent, uint16_t* output, int groups,
                              int first_position, void* stream) {
    launch_postprocess<512, true>(latent, output, groups, first_position, stream);
}
void postprocess_source20_kv(const float* latent, uint16_t* output, int tokens,
                             int first_position, void* stream) {
    launch_postprocess<512, true>(latent, output, tokens, first_position, stream, 1);
}
void postprocess_source20_index_keys(const float* normalized, uint16_t* output, int tokens,
                                    int first_position, void* stream) {
    launch_postprocess<128, false>(normalized, output, tokens, first_position, stream, 1);
}
void postprocess_index_keys(const float* normalized, uint16_t* output, int groups,
                            int first_position, void* stream) {
    launch_postprocess<128, false>(normalized, output, groups, first_position, stream);
}

void postprocess_index_query(const float* projected, uint16_t* output, int position, void* stream) {
    launch_postprocess<128, false>(projected, output, 32, position, stream, 0);
}

struct CudaCompressedKvCache::Impl {
    int device, capacity, count = 0;
    bool failed = false;
    void* stream = nullptr;
    std::unique_ptr<core::DeviceArena> arena;
    uint16_t* cache;
    Impl(int groups, int device_id) : device(device_id), capacity(groups) {
        if (groups <= 0 || groups > max_groups) throw std::invalid_argument("DeepSeek compressed KV capacity must be 1..524288 groups");
        arena = std::make_unique<core::DeviceArena>(size_t(groups) * dim * 2, device);
        cache = static_cast<uint16_t*>(arena->alloc(arena->capacity()));
        ck(cudaMemset(cache, 0, arena->capacity()), "initialize compressed KV cache");
        ck(cudaStreamSynchronize(nullptr), "wait for compressed KV initialization");
    }
    void check_stream(void* s) const {
        if (!s || (stream && stream != s)) throw std::invalid_argument("DeepSeek compressed KV cache needs one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get compressed KV device");
        if (current != device) throw std::invalid_argument("DeepSeek compressed KV called on wrong device");
        if (failed) throw std::logic_error("DeepSeek compressed KV cache must be reconstructed after an execution failure");
    }
};
CudaCompressedKvCache::CudaCompressedKvCache(int capacity_groups, int device)
    : impl_(std::make_unique<Impl>(capacity_groups, device)) {}
CudaCompressedKvCache::~CudaCompressedKvCache() = default;
void CudaCompressedKvCache::append(const float* latent, CompressedBatch batch, void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    if (batch.count < 0 || batch.count > p.capacity - p.count ||
        batch.first_position != (batch.count ? 2 * p.count : -1))
        throw std::invalid_argument("DeepSeek compressed KV append is noncontiguous, malformed or exceeds capacity");
    if (batch.count) {
        const size_t bytes = size_t(batch.count) * dim * 4;
        if (!valid(latent, bytes, 4) || overlaps(latent, bytes, p.arena->base(), p.arena->capacity()))
            throw std::invalid_argument("DeepSeek compressed KV cache requires an external aligned latent");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        if (batch.count) postprocess_compressed_kv(latent, p.cache + size_t(p.count) * dim, batch.count, batch.first_position, stream);
        ck(cudaStreamSynchronize(s), "wait for compressed KV append");
        p.count += batch.count;
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaCompressedKvCache::reset(void* stream) {
    auto& p = *impl_; p.check_stream(stream);
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        ck(cudaMemsetAsync(p.cache, 0, p.arena->capacity(), s), "clear compressed KV cache");
        ck(cudaStreamSynchronize(s), "wait for compressed KV reset"); p.count = 0;
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
int CudaCompressedKvCache::size() const { return impl_->count; }
int CudaCompressedKvCache::capacity() const { return impl_->capacity; }
size_t CudaCompressedKvCache::device_bytes() const { return impl_->arena->capacity(); }
const uint16_t* CudaCompressedKvCache::data() const { return impl_->cache; }
} // namespace strata::deepseek41
