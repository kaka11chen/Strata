#include "strata/deepseek41/routed_cuda.hpp"
#include "strata/deepseek41/fp4_layout.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace strata::deepseek41 {
namespace {
void ck(cudaError_t code, const char* what) {
    if (code != cudaSuccess)
        throw core::CudaError(std::string(what) + ": " + cudaGetErrorString(code), int(code));
}
size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
} // namespace

CudaRoutedLayer::CudaRoutedLayer(const Geometry& g, int layer, std::span<const uint16_t> router,
                                 std::span<const float> bias, core::ExpertSource& source, int slots, int device)
    : layer_(layer), hidden_(g.hidden), device_(device), blob_bytes_(nvfp4_expert_bytes(g)), source_(source) {
    if (g.layers <= 0 || g.layers > 40 || layer < 0 || layer >= g.layers || g.experts != 384 || g.top_k != 6 ||
        slots <= 0 || slots > 384 || router.size() != size_t(g.hidden) * 384 || bias.size() != 384 ||
        !router.data() || !bias.data())
        throw std::invalid_argument("DeepSeek routed layer requires 1017 geometry/router and 1..384 cache slots");
    for (uint16_t value : router)
        if (!std::isfinite(kernels::f32_from_bf16(value)))
            throw std::invalid_argument("Nonfinite DeepSeek router weight");
    for (float value : bias)
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite DeepSeek router bias");
    std::copy(bias.begin(), bias.end(), bias_.begin());
    owners_.assign(size_t(slots), -1);
    ages_.resize(size_t(slots));
    scales_.resize(size_t(slots));
    const size_t router_bytes = router.size_bytes(), input_bytes = size_t(g.hidden) * 2;
    const size_t logits_bytes = 384 * sizeof(float), output_bytes = size_t(g.hidden) * sizeof(float);
    arena_ = std::make_unique<core::DeviceArena>(aligned(router_bytes) + aligned(input_bytes) +
                                                aligned(logits_bytes) + output_bytes, device);
    router_ = static_cast<uint16_t*>(arena_->alloc(router_bytes));
    input_bf16_ = static_cast<uint16_t*>(arena_->alloc(input_bytes));
    logits_ = static_cast<float*>(arena_->alloc(logits_bytes));
    expert_output_ = static_cast<float*>(arena_->alloc(output_bytes));
    ck(cudaMemcpy(router_, router.data(), router_bytes, cudaMemcpyHostToDevice), "upload DeepSeek router");
    expert_ = std::make_unique<CudaExpert>(g, device);
    std::string error;
    if (!cache_.open(slots, g.layers, 384, int64_t(blob_bytes_), error))
        throw std::runtime_error(error);
    // Router upload and ExpertCache's initial zeroing use the legacy stream.
    // They must finish before forward() is allowed on a nonblocking stream.
    ck(cudaStreamSynchronize(nullptr), "wait for DeepSeek routed-layer initialization");
}

int32_t CudaRoutedLayer::ensure_slot(int32_t expert, void* stream) {
    int32_t slot = cache_.slot_of(layer_, expert);
    if (slot != core::kNotResident) {
        ++stats_.hits;
        ages_[size_t(slot)] = ++tick_;
        return slot;
    }
    // Read/convert before touching residency. A missing expert is an error, never
    // a replacement route to whichever experts happen to be available.
    const uint8_t* blob = source_.blob(layer_, expert);
    if (!blob)
        throw std::runtime_error("DeepSeek expert source could not supply selected expert " + std::to_string(expert));
    const auto scales = nvfp4_expert_scales({blob, blob_bytes_});
    const bool replacing = cache_.resident() == cache_.slots();
    slot = replacing ? int32_t(std::min_element(ages_.begin(), ages_.end()) - ages_.begin())
                     : int32_t(cache_.resident());
    std::string error;
    if (!cache_.fill_slot(slot, blob, stream, error))
        throw std::runtime_error(error);
    // Orders reuse of both the device slot and the source's one transient host
    // buffer. Publish replacement only after the copy completes, per ExpertCache.
    ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "fill DeepSeek expert cache");
    if (replacing) {
        cache_.replace(layer_, owners_[size_t(slot)], expert);
        ++stats_.evictions;
    } else if (cache_.admit(layer_, expert) != slot) {
        throw std::logic_error("DeepSeek cache admission disagrees with the filled slot");
    }
    owners_[size_t(slot)] = expert;
    scales_[size_t(slot)] = scales;
    ages_[size_t(slot)] = ++tick_;
    ++stats_.misses;
    stats_.expert_h2d_bytes += blob_bytes_;
    return slot;
}

Route CudaRoutedLayer::forward(const float* input, float* output, void* stream) {
    const uintptr_t in = uintptr_t(input), out = uintptr_t(output);
    const size_t bytes = size_t(hidden_) * sizeof(float);
    if (!input || !output || !stream || in % 4 || out % 4 || (in > out ? in - out : out - in) < bytes)
        throw std::invalid_argument("DeepSeek routed input/output must be aligned, disjoint and use an explicit stream");
    int device = -1;
    ck(cudaGetDevice(&device), "get DeepSeek CUDA device");
    if (device != device_)
        throw std::invalid_argument("DeepSeek routed layer called on the wrong CUDA device");
    if (failed_)
        throw std::logic_error("DeepSeek routed layer must be reconstructed after an execution failure");
    const auto s = static_cast<cudaStream_t>(stream);
    try {
        kernels::f32_to_bf16_bulk(input, input_bf16_, hidden_, stream);
        kernels::bf16_gemv(input_bf16_, router_, logits_, hidden_, 384, stream);
        ck(cudaMemcpyAsync(host_logits_.data(), logits_, sizeof(host_logits_), cudaMemcpyDeviceToHost, s),
           "read DeepSeek router logits");
        ck(cudaStreamSynchronize(s), "wait for DeepSeek router");
        stats_.router_d2h_bytes += sizeof(host_logits_);
        const Route route = route_top6(host_logits_, bias_);
        source_.begin_layer(layer_, route.ids.data(), 6);
        std::array<int, 6> order{};
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return route.ids[a] < route.ids[b]; });
        ck(cudaMemsetAsync(output, 0, bytes, s), "clear DeepSeek routed output");
        for (int rank : order) {
            const int32_t slot = ensure_slot(route.ids[rank], stream);
            expert_->forward_from_blob(cache_.device_slot(slot), blob_bytes_, scales_[size_t(slot)], input, route.weights[rank],
                                       expert_output_, stream);
            kernels::add_inplace(output, expert_output_, hidden_, stream);
        }
        ck(cudaStreamSynchronize(s), "wait for DeepSeek routed experts");
        return route;
    } catch (...) {
        failed_ = true;
        (void) cudaStreamSynchronize(s);
        throw;
    }
}
} // namespace strata::deepseek41
