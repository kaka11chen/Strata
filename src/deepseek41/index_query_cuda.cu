#include "strata/deepseek41/candidate_index_query_cuda.hpp"
#include "strata/deepseek41/shared_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <climits>
#include <cmath>

namespace strata::deepseek41 {
namespace {
constexpr int heads = 32, dim = 128, qr_dim = 1280, hidden = 5120, topk = 512;
constexpr size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    return uintptr_t(a) < uintptr_t(b) + bn && uintptr_t(b) < uintptr_t(a) + an;
}
size_t sort_bytes(int count, int limit = topk) {
    size_t pairs = 0, ids = 0;
    ck(cub::DeviceRadixSort::SortPairsDescending(nullptr, pairs, static_cast<const float*>(nullptr),
        static_cast<float*>(nullptr), static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr), count), "size index score sort");
    ck(cub::DeviceRadixSort::SortKeys(nullptr, ids, static_cast<const int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr), std::min(count, limit)), "size index position sort");
    return std::max(pairs, ids);
}
__device__ float bf(float x) { return kernels::f32_from_bf16(kernels::bf16_from_f32(x)); }
// Ratio-1 head weights can sit less than one FP32 ULP from a BF16 midpoint
// after cancellation. Neither serial nor split FP32 dots preserve that boundary.
// Only this small 32x5120 projection uses FP64 accumulation; retain the explicit
// FP32 -> BF16 projection and BF16 scale boundaries below. No extra workspace.
__global__ void project_ratio1_head_weights(const uint16_t* x, const uint16_t* w, float* weights) {
    const int head = int(blockIdx.x), lane = int(threadIdx.x);
    double sum = 0.;
    for (int i = lane; i < hidden; i += 32)
        sum = __dadd_rn(sum, double(kernels::f32_from_bf16(x[i])) *
                            double(kernels::f32_from_bf16(w[head * hidden + i])));
    for (int off = 16; off > 0; off >>= 1)
        sum = __dadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, off));
    if (lane == 0) weights[head] = __double2float_rn(sum);
}
__global__ void scale_weights(float* weights) {
    const int head = int(threadIdx.x);
    // sqrt(128)^-1 * sqrt(32)^-1 == 1/64. Projection and multiply both return BF16.
    weights[head] = bf(bf(weights[head]) * (1.f / 64.f));
}
__global__ void accumulate_scores(const float* dot, const float* weights, float* score, int32_t* ids,
                                  int count, int visible, int head) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= count) return;
    // einsum -> BF16, relu, BF16 multiply, F32 sum over heads -> BF16.
    const float product = bf(fmaxf(0.f, bf(dot[i])) * weights[head]);
    const float total = __fadd_rn(head == 0 ? 0.f : score[i], product);
    score[i] = head == heads - 1 ? (i < visible ? bf(total) : -INFINITY) : total;
    if (head == heads - 1) ids[i] = i;
}
__global__ void shift_indices(int32_t* ids, int count, int visible, int offset) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < count) ids[i] = ids[i] < visible ? ids[i] + offset : -1;
}
// Source-20 candidate rule only; query projection/scoring and CUB selection
// are shared with the existing indexer. Invalid blocks sort behind valid IDs.
__global__ void candidate_scores(const float* input, float* scores, int32_t* ids, int width, int visible) {
    const int block = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (block >= (width + Source20Batch::block_size - 1) / Source20Batch::block_size) return;
    float value = -INFINITY;
    for (int j = 0; j < Source20Batch::block_size; ++j) {
        const int position = block * Source20Batch::block_size + j;
        if (position < width && position < visible) value = fmaxf(value, input[position]);
    }
    if (block == (visible - 1) / Source20Batch::block_size) value = INFINITY;
    scores[block] = value;
    ids[block] = value > -INFINITY ? block : INT_MAX;
}
// Validate the compact source-20 mask before mutating query results. The source
// publication guarantees this same finite-score cardinality in the runtime.
__global__ void check_candidates(const int32_t* blocks, int count, int selected, int newest, int* invalid) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= count) return;
    const int id = blocks[i];
    if (i >= selected ? id != -1 : (id < 0 || id > newest || (i && id <= blocks[i-1]) ||
                                    (i + 1 == selected && id != newest))) atomicExch(invalid, 1);
}
__global__ void mask_candidates(float* scores, int32_t* ids, int count, const int32_t* blocks, int selected) {
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= count) return;
    const int block = i / Source20Batch::block_size;
    int lo = 0, hi = selected;
    while (lo < hi) { const int mid = (lo + hi) / 2; if (blocks[mid] < block) lo = mid + 1; else hi = mid; }
    if (lo == selected || blocks[lo] != block) { scores[i] = -INFINITY; ids[i] = INT_MAX; }
}
void select_positions(void* workspace, size_t workspace_bytes, const float* scores, float* sorted_scores,
                      const int32_t* ids, int32_t* sorted_ids, int32_t* selected,
                      int length, int count, int visible, int offset, cudaStream_t stream) {
    size_t available = workspace_bytes;
    // CUB is stable: ascending input IDs give low-ID priority at tied scores.
    ck(cub::DeviceRadixSort::SortPairsDescending(workspace, available, scores, sorted_scores,
        ids, sorted_ids, length, 0, 32, stream), "sort index scores");
    available = workspace_bytes;
    ck(cub::DeviceRadixSort::SortKeys(workspace, available, sorted_ids, selected,
        count, 0, 32, stream), "sort selected positions");
    shift_indices<<<(count + 255) / 256, 256, 0, stream>>>(selected, count, visible, offset);
    ck(cudaGetLastError(), "shift selected positions");
}
} // namespace

struct CudaIndexQuery::Impl {
    int device, capacity, ratio;
    bool candidate_mode;
    int* invalid_candidates = nullptr;
    bool failed = false;
    void* stream = nullptr;
    size_t workspace_bytes = 0;
    std::unique_ptr<core::DeviceArena> arena;
    uint16_t *wq, *ww, *activation, *query;
    float *projected, *weights, *dot, *scores;
    int32_t *ids, *sorted_ids, *selected;
    void* workspace;
    Impl(const IndexQueryWeights& w, int groups, int device_id, int compress_ratio = 2, bool use_candidates = false)
        : device(device_id), capacity(groups), ratio(compress_ratio), candidate_mode(use_candidates) {
        if (groups <= 0 || groups > Source20Batch::max_context / ratio)
            throw std::invalid_argument("Index query capacity exceeds its position range");
        if (w.query.rows != heads * dim || w.query.cols != qr_dim ||
            w.head_weights.size() != heads * hidden || !w.head_weights.data())
            throw std::invalid_argument("Index query needs fixed 1017 projection shapes");
        for (uint16_t bits : w.head_weights)
            if (!std::isfinite(kernels::f32_from_bf16(bits))) throw std::invalid_argument("Nonfinite index head weight");
        const auto decoded = fp8_block32_to_bf16(w.query);
        ck(cudaSetDevice(device), "set index query device");
        workspace_bytes = sort_bytes(groups);
        const size_t fixed = aligned(decoded.size() * 2) + aligned(w.head_weights.size_bytes()) +
            aligned(hidden * 2) + aligned(heads * dim * 4) + aligned(heads * dim * 2) + aligned(heads * 4) + aligned(topk * 4);
        arena = std::make_unique<core::DeviceArena>(fixed + 4 * aligned(size_t(groups) * 4) + aligned(workspace_bytes) + (candidate_mode ? aligned(4) : 0), device);
        wq = static_cast<uint16_t*>(arena->alloc(decoded.size() * 2));
        ww = static_cast<uint16_t*>(arena->alloc(w.head_weights.size_bytes()));
        activation = static_cast<uint16_t*>(arena->alloc(hidden * 2));
        projected = static_cast<float*>(arena->alloc(heads * dim * 4));
        query = static_cast<uint16_t*>(arena->alloc(heads * dim * 2));
        weights = static_cast<float*>(arena->alloc(heads * 4));
        selected = static_cast<int32_t*>(arena->alloc(topk * 4));
        dot = static_cast<float*>(arena->alloc(size_t(groups) * 4));
        scores = static_cast<float*>(arena->alloc(size_t(groups) * 4));
        ids = static_cast<int32_t*>(arena->alloc(size_t(groups) * 4));
        sorted_ids = static_cast<int32_t*>(arena->alloc(size_t(groups) * 4));
        workspace = arena->alloc(workspace_bytes);
        if (candidate_mode) invalid_candidates = static_cast<int*>(arena->alloc(4));
        ck(cudaMemcpy(wq, decoded.data(), decoded.size() * 2, cudaMemcpyHostToDevice), "upload index query weights");
        ck(cudaMemcpy(ww, w.head_weights.data(), w.head_weights.size_bytes(), cudaMemcpyHostToDevice), "upload index head weights");
        ck(cudaStreamSynchronize(nullptr), "wait for index query uploads");
    }
    IndexSelection forward(const float* x, const float* qr, const uint16_t* keys,
                           int key_count, int position, int offset, void* stream,
                           const int32_t* candidates = nullptr, int candidate_count = 0);
    void check_stream(void* s) const {
        if (!s || (stream && stream != s)) throw std::invalid_argument("Index query needs one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get index query device");
        if (current != device) throw std::invalid_argument("Index query called on wrong device");
        if (failed) throw std::logic_error("Index query must be reconstructed after execution failure");
    }
    void external(const void* p, size_t bytes, size_t alignment) const {
        if (!p || uintptr_t(p) % alignment || bytes > UINTPTR_MAX - uintptr_t(p) ||
            overlaps(p, bytes, arena->base(), arena->capacity()))
            throw std::invalid_argument("Index query requires aligned external inputs");
    }
};
CudaIndexQuery::CudaIndexQuery(const IndexQueryWeights& weights, int capacity_groups, int device)
    : impl_(std::make_unique<Impl>(weights, capacity_groups, device)) {}
CudaIndexQuery::~CudaIndexQuery() = default;
IndexSelection CudaIndexQuery::Impl::forward(const float* x, const float* qr, const uint16_t* keys,
                                       int key_count, int position, int offset, void* stream,
                                       const int32_t* candidates, int candidate_count) {
    auto& p = *this; p.check_stream(stream);
    if (position < 0 || position >= 1048576 || key_count < 0 || key_count > p.capacity ||
        key_count < (position + 1) / p.ratio || offset < 0 || int64_t(offset) + std::max(key_count - 1, 0) > INT_MAX)
        throw std::invalid_argument("Invalid index query position, key count or output offset");
    p.external(x, hidden * 4, 4); p.external(qr, qr_dim * 4, 4);
    if (key_count) p.external(keys, size_t(key_count) * dim * 2, 2);
    // Query the library's actual requirement before any output mutation. No device
    // allocation occurs here, including when a different CUB policy is selected.
    if (key_count && sort_bytes(key_count) > p.workspace_bytes)
        throw std::invalid_argument("Index query sort exceeds its startup workspace");
    const int count = std::min(key_count, topk), visible = (position + 1) / p.ratio;
    const auto s = static_cast<cudaStream_t>(stream);
    const int selected_blocks = std::min(Source20Batch::topk_blocks, (visible + 7) / 8);
    if (p.candidate_mode) {
        if (candidate_count != std::min(Source20Batch::topk_blocks, (key_count + 7) / 8))
            throw std::invalid_argument("Candidate mask width does not match the whole source batch");
        p.external(candidates, size_t(candidate_count) * 4, 4);
        int invalid = 0;
        try {
            ck(cudaMemsetAsync(p.invalid_candidates, 0, 4, s), "clear candidate validation");
            check_candidates<<<(candidate_count + 255) / 256, 256, 0, s>>>(candidates, candidate_count,
                selected_blocks, (visible - 1) / 8, p.invalid_candidates);
            ck(cudaGetLastError(), "validate candidate blocks");
            ck(cudaMemcpyAsync(&invalid, p.invalid_candidates, 4, cudaMemcpyDeviceToHost, s), "read candidate validation");
            ck(cudaStreamSynchronize(s), "wait candidate validation");
        } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
        if (invalid) throw std::invalid_argument("Malformed finite-score source-20 candidate row");
    }
    try {
        p.stream = stream;
        quantize_fp8_bf16(qr, p.activation, qr_dim, stream);
        kernels::bf16_gemv(p.activation, p.wq, p.projected, qr_dim, heads * dim, stream);
        postprocess_index_query(p.projected, p.query, position, stream);
        kernels::f32_to_bf16_bulk(x, p.activation, hidden, stream);
        if (p.ratio == 1) {
            project_ratio1_head_weights<<<heads, 32, 0, s>>>(p.activation, p.ww, p.weights);
            ck(cudaGetLastError(), "project ratio1 index head weights");
        } else {
            kernels::bf16_gemv(p.activation, p.ww, p.weights, hidden, heads, stream);
        }
        scale_weights<<<1, heads, 0, s>>>(p.weights);
        ck(cudaGetLastError(), "scale index head weights");
        if (key_count) {
            for (int head = 0; head < heads; ++head) {
                kernels::bf16_gemv(p.query + head * dim, keys, p.dot, dim, key_count, stream);
                accumulate_scores<<<(key_count + 255) / 256, 256, 0, s>>>(p.dot, p.weights, p.scores, p.ids, key_count, visible, head);
                ck(cudaGetLastError(), "accumulate index scores");
            }
            if (p.candidate_mode) {
                mask_candidates<<<(key_count + 255) / 256, 256, 0, s>>>(p.scores, p.ids, key_count, candidates, selected_blocks);
                ck(cudaGetLastError(), "mask index scores by source-20 candidate blocks");
            }
            select_positions(p.workspace, p.workspace_bytes, p.scores, p.dot, p.ids, p.sorted_ids,
                             p.selected, key_count, count, visible, offset, s);
        }
        ck(cudaStreamSynchronize(s), "wait for index query");
        return {p.selected, p.scores, count};
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
IndexSelection CudaIndexQuery::forward(const float* x, const float* qr, const uint16_t* keys,
                                       int key_count, int position, int offset, void* stream) {
    return impl_->forward(x, qr, keys, key_count, position, offset, stream);
}
int CudaIndexQuery::capacity() const { return impl_->capacity; }
size_t CudaIndexQuery::device_bytes() const { return impl_->arena->capacity(); }
size_t CudaIndexQuery::sort_workspace_bytes() const { return impl_->workspace_bytes; }
const uint16_t* CudaIndexQuery::query() const { return impl_->query; }
const float* CudaIndexQuery::head_weights() const { return impl_->weights; }
struct CudaSource20Candidates::Impl {
    int device, capacity;
    bool failed = false;
    void* stream = nullptr;
    size_t workspace_bytes;
    std::unique_ptr<core::DeviceArena> arena;
    float *scores, *sorted_scores;
    int32_t *ids, *sorted_ids, *selected;
    void* workspace;
    Impl(int tokens, int device_id) : device(device_id), capacity(tokens) {
        if (tokens < 1 || tokens > Source20Batch::max_context)
            throw std::invalid_argument("Source-20 candidate capacity must be 1..1048576 tokens");
        ck(cudaSetDevice(device), "set source-20 candidate device");
        const int blocks = (tokens + Source20Batch::block_size - 1) / Source20Batch::block_size;
        const int count = std::min(blocks, Source20Batch::topk_blocks);
        workspace_bytes = sort_bytes(blocks, Source20Batch::topk_blocks);
        arena = std::make_unique<core::DeviceArena>(4 * aligned(size_t(blocks) * 4) +
                    aligned(count * 4) + aligned(workspace_bytes), device);
        scores = static_cast<float*>(arena->alloc(size_t(blocks) * 4));
        sorted_scores = static_cast<float*>(arena->alloc(size_t(blocks) * 4));
        ids = static_cast<int32_t*>(arena->alloc(size_t(blocks) * 4));
        sorted_ids = static_cast<int32_t*>(arena->alloc(size_t(blocks) * 4));
        selected = static_cast<int32_t*>(arena->alloc(count * 4));
        workspace = arena->alloc(workspace_bytes);
    }
    void external(const void* p, size_t bytes, size_t alignment) const {
        if (!p || uintptr_t(p) % alignment || bytes > UINTPTR_MAX - uintptr_t(p) ||
            overlaps(p, bytes, arena->base(), arena->capacity()))
            throw std::invalid_argument("Source-20 candidates require aligned external inputs");
    }
    void check(const float* input, const Source20Batch& batch, int query, void* s) const {
        (void)batch.position(query);
        if (!s || (stream && stream != s)) throw std::invalid_argument("Source-20 candidates need one explicit stream");
        int current = -1; ck(cudaGetDevice(&current), "get source-20 candidate device");
        if (current != device) throw std::invalid_argument("Source-20 candidates called on wrong device");
        if (failed) throw std::logic_error("Source-20 candidates require reconstruction after execution failure");
        if (batch.key_count() > capacity) throw std::invalid_argument("Source-20 candidates exceed startup capacity");
        external(input, size_t(batch.key_count()) * 4, 4);
        const int blocks = (batch.key_count() + Source20Batch::block_size - 1) / Source20Batch::block_size;
        if (sort_bytes(blocks, Source20Batch::topk_blocks) > workspace_bytes)
            throw std::invalid_argument("Source-20 candidates exceed startup sort workspace");
    }
};
CudaSource20Candidates::CudaSource20Candidates(int capacity_tokens, int device)
    : impl_(std::make_unique<Impl>(capacity_tokens, device)) {}
CudaSource20Candidates::~CudaSource20Candidates() = default;
Source20CandidateSelection CudaSource20Candidates::forward(const float* scores, const Source20Batch& batch,
                                                           int query, void* stream) {
    auto& p = *impl_; p.check(scores, batch, query, stream);
    const int blocks = (batch.key_count() + Source20Batch::block_size - 1) / Source20Batch::block_size;
    const int count = batch.candidate_stride();
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        p.stream = stream;
        candidate_scores<<<(blocks + 255) / 256, 256, 0, s>>>(scores, p.scores, p.ids, batch.key_count(), batch.visible_keys(query));
        ck(cudaGetLastError(), "reduce source-20 candidate blocks");
        select_positions(p.workspace, p.workspace_bytes, p.scores, p.sorted_scores, p.ids, p.sorted_ids,
                         p.selected, blocks, count, blocks, 0, s);
        ck(cudaStreamSynchronize(s), "wait for source-20 candidates");
        return {p.selected, count};
    } catch (...) { p.failed = true; (void)cudaStreamSynchronize(s); throw; }
}
int CudaSource20Candidates::capacity() const { return impl_->capacity; }
size_t CudaSource20Candidates::device_bytes() const { return impl_->arena->capacity(); }
size_t CudaSource20Candidates::sort_workspace_bytes() const { return impl_->workspace_bytes; }

CudaSource20IndexQuery::CudaSource20IndexQuery(const Source20IndexQueryWeights& weights, int capacity_tokens, int device)
    : impl_(std::make_unique<CudaIndexQuery::Impl>(IndexQueryWeights{weights.query, weights.head_weights, 20}, capacity_tokens, device, 1)),
      candidates_(capacity_tokens, device) {}
CudaSource20IndexQuery::~CudaSource20IndexQuery() = default;
Source20IndexSelection CudaSource20IndexQuery::forward(const float* x, const float* qr, const uint16_t* keys,
                                                     const Source20Batch& batch, int query, int offset, void* stream) {
    // Preflight candidate storage before query computation changes the last result.
    auto& candidates = *candidates_.impl_;
    candidates.check(impl_->scores, batch, query, stream);
    candidates.external(x, hidden * 4, 4); candidates.external(qr, qr_dim * 4, 4);
    candidates.external(keys, size_t(batch.key_count()) * dim * 2, 2);
    const auto index = impl_->forward(x, qr, keys, batch.key_count(), batch.position(query), offset, stream);
    try {
        return {index, candidates_.forward(index.scores, batch, query, stream)};
    } catch (...) { impl_->failed = true; throw; }
}
int CudaSource20IndexQuery::capacity() const { return impl_->capacity; }
size_t CudaSource20IndexQuery::device_bytes() const { return impl_->arena->capacity() + candidates_.device_bytes(); }
size_t CudaSource20IndexQuery::sort_workspace_bytes() const { return impl_->workspace_bytes + candidates_.sort_workspace_bytes(); }
const uint16_t* CudaSource20IndexQuery::query() const { return impl_->query; }
const float* CudaSource20IndexQuery::head_weights() const { return impl_->weights; }
CudaCandidateIndexQuery::CudaCandidateIndexQuery(const CandidateIndexQueryWeights& w, int capacity_tokens, int device) {
    if (w.layer != 24 && w.layer != 28 && w.layer != 32 && w.layer != 36) throw std::invalid_argument("Candidate index query requires original layer 24, 28, 32 or 36");
    impl_ = std::make_unique<CudaIndexQuery::Impl>(IndexQueryWeights{w.query, w.head_weights, w.layer}, capacity_tokens, device, 1, true);
}
CudaCandidateIndexQuery::~CudaCandidateIndexQuery() = default;
IndexSelection CudaCandidateIndexQuery::forward(const float* x, const float* qr, const uint16_t* keys,
                                               const Source20Batch& batch, int query, Source20CandidateSelection candidates,
                                               int offset, void* stream) {
    return impl_->forward(x, qr, keys, batch.key_count(), batch.position(query), offset, stream,
                          candidates.blocks, candidates.count);
}
int CudaCandidateIndexQuery::capacity() const { return impl_->capacity; }
size_t CudaCandidateIndexQuery::device_bytes() const { return impl_->arena->capacity(); }
size_t CudaCandidateIndexQuery::sort_workspace_bytes() const { return impl_->workspace_bytes; }
const uint16_t* CudaCandidateIndexQuery::query() const { return impl_->query; }
const float* CudaCandidateIndexQuery::head_weights() const { return impl_->weights; }
std::pair<const void*, size_t> CudaCandidateIndexQuery::storage() const { return {impl_->arena->base(), impl_->arena->capacity()}; }
} // namespace strata::deepseek41
