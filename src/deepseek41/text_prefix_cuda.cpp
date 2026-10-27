#include "strata/deepseek41/text_prefix_cuda.hpp"
#include <cuda_runtime.h>
#include <algorithm>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120, max_position = 1048576;
constexpr size_t residual_size = 4 * hidden;
size_t aligned(size_t n) { return (n + 255) / 256 * 256; }
void ck(cudaError_t e, const char* label) {
    if (e != cudaSuccess) throw core::CudaError(std::string(label) + ": " + cudaGetErrorString(e), int(e));
}
int checked(const Geometry& g, const TextPrefixWeights& w, int slots, int prefill, int groups, int device, std::span<const uint32_t> map14, const std::string& path14) {
    if (g.layers != 40 || g.hidden != hidden || g.expert_width != 2304 || g.experts != 384 || g.top_k != 6 ||
        g.compress_ratios.size() != 40 || g.compress_ratios[0] != 0 || g.compress_ratios[1] != 0 ||
        g.sources.size() != 40 || (w.layers.size() != 4 && w.layers.size() != 8 && w.layers.size() != 14 && w.layers.size() != 15 && w.layers.size() != 20 && w.layers.size() != 24 && w.layers.size() != 28 && w.layers.size() != 32 && w.layers.size() != 36 && w.layers.size() != 40) ||
        slots <= 0 || slots > 384 || prefill <= 0 || prefill > max_position || groups <= 0 || groups > max_position / 2)
        throw std::invalid_argument("Text prefix requires fixed 1017 4/8/14/15/20/24/28/32/36/40-layer prefix and bounded prefill/cache capacities");
    for (int l = 0; l < int(w.layers.size()); ++l)
        if (w.layers[size_t(l)].attention.layer != l)
            throw std::invalid_argument("Text prefix weights must be in original layer order");
    for (size_t l = 2; l < w.layers.size(); ++l) {
        const int source = l < 8 ? 2 : l < 14 ? 8 : l < 20 ? 14 : 20;
        if (g.compress_ratios[l] != (l < 20 ? 2 : 1) || g.sources[l].kv != source || g.sources[l].index != (l >= 24 ? int(l / 4 * 4) : source) || g.sources[l].candidates != ((l == 24 || l == 28 || l == 32 || l == 36) ? 20 : -1))
            throw std::invalid_argument("Text prefix requires the original KV/index/candidate source topology");
    }
    if (w.engram.layer != 1) throw std::invalid_argument("Text prefix requires original layer-1 Engram weights");
    if (w.compressor.layer != 2 || w.key.layer != 2 || w.query.layer != 2 ||
        (w.layers.size() >= 14 && (w.compressor8.layer != 8 || w.key8.layer != 8 || w.query8.layer != 8)))
        throw std::invalid_argument("Text prefix requires matching original source weight identities");
    if (w.layers.size() >= 15 && (w.engram14.layer != 14 || w.compressor14.layer != 14 ||
        w.key14.layer != 14 || w.query14.layer != 14 || map14.size() != engram_token_count || path14.empty()))
        throw std::invalid_argument("Prefixes containing layer 14 require original Engram/source-14 weights, token map and table path");
    if ((w.layers.size() >= 28 && w.query24.layer != 24) || (w.layers.size() >= 32 && w.query28.layer != 28) || (w.layers.size() >= 36 && w.query32.layer != 32) || (w.layers.size() == 40 && w.query36.layer != 36))
        throw std::invalid_argument("Text prefix requires matching original candidate index query weights");
    return device;
}
bool valid(const void* p, size_t n, size_t alignment) {
    return p && uintptr_t(p) % alignment == 0 && n <= UINTPTR_MAX - uintptr_t(p);
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    return an && bn && (uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b) - uintptr_t(a) < an : uintptr_t(a) - uintptr_t(b) < bn);
}
} // namespace
CudaTextPrefix::CudaTextPrefix(const Geometry& g, const TextPrefixWeights& w,
    std::span<const uint32_t> map, const std::string& path, core::ExpertSource& source,
    int slots, int prefill, int groups, const EngramIoOptions& io, int device,
    std::span<const uint32_t> map14, const std::string& path14)
    : device_(checked(g, w, slots, prefill, groups, device, map14, path14)), prefill_capacity_(prefill), layer_count_(int(w.layers.size())),
      layer0_(g, w.layers[0], source, slots, device),
      layer1_(g, w.layers[1], w.engram, map, path, source, slots, io, device),
      layer2_(g, w.layers[2], w.compressor, w.key, w.query, source, slots, groups, device, prefill),
      arena_(2 * (size_t(prefill) * residual_size * 4 + aligned(size_t(prefill) * 16)), device) {
    if (layer_count_ >= 14)
        layer8_ = std::make_unique<CudaSourceDecoder>(g, w.layers[8], w.compressor8, w.key8, w.query8, source, slots, groups, device, prefill, 8);
    if (layer_count_ >= 15)
        layer14_ = std::make_unique<CudaLayer14Decoder>(g, w.layers[14], w.engram14, w.compressor14,
            w.key14, w.query14, map14, path14, source, slots, groups, io, device, prefill);
    if (layer_count_ >= 24) {
        const int context = std::min(max_position, 2 * groups + 1);
        layer20_ = std::make_unique<CudaSource20Decoder>(g, w.layers[20], w.compressor20, w.key20, w.query20,
            source, slots, context, std::min(prefill, context), device);
    }
    consumers_.resize(size_t(layer_count_));
    // 40 entries cover original backbone IDs 0..39, all with 384/top6 MoE.
    // DSpark targets 37..39 are ordinary consumers; MTP IDs 40..42 are excluded.
    for (int l = 3; l < layer_count_; ++l) {
        if (l == 24 || l == 28 || l == 32 || l == 36) {
            const int context = layer20_->capacity();
            consumers_[size_t(l)] = std::make_unique<CudaSharedDecoder>(g, w.layers[size_t(l)], l == 24 ? w.query24 : l == 28 ? w.query28 : l == 32 ? w.query32 : w.query36, source, slots,
                                                                context, std::min(prefill, context), device);
        } else if (l != 8 && l != 14 && l != 20)
            consumers_[size_t(l)] = std::make_unique<CudaSharedDecoder>(g, w.layers[size_t(l)], source, slots, device, l);
    }
    for (size_t i = 0; i < 2; ++i) {
        residual_[i] = static_cast<float*>(arena_.alloc(size_t(prefill) * residual_size * 4));
        coefficients_[i] = static_cast<float*>(arena_.alloc(size_t(prefill) * 16));
    }
}
void CudaTextPrefix::check_stream(void* stream) const {
    if (failed_) throw std::logic_error("Text prefix must be reconstructed after any layer execution failure");
    if (!stream || (stream_ && stream != stream_)) throw std::invalid_argument("Text prefix requires one explicit stream");
    int device = -1; ck(cudaGetDevice(&device), "get text-prefix device");
    if (device != device_) throw std::invalid_argument("Text prefix called on wrong device");
}
void CudaTextPrefix::forward(std::span<const int32_t> tokens, const float* input, const float* incoming,
    float* output, float* next_pre, int start, void* stream, std::span<Route> routes) {
    check_stream(stream);
    const size_t count = tokens.size();
    if (!count || count > size_t(prefill_capacity_) || start < 0 || start != next_ ||
        count > size_t(max_position - start) || (start && count != 1) ||
        (!routes.empty() && routes.size() != size_t(layer_count_) * count) || (size_t(start) + count) / 2 > size_t(capacity()) ||
        (layer20_ && size_t(start) + count > size_t(capacity(20))))
        throw std::invalid_argument("Text prefix needs initial bounded prefill then contiguous single-token decode within cache capacity");
    const void* pointers[] = {input, incoming, output, next_pre, tokens.data(), routes.data()};
    const size_t bytes[] = {count * residual_size * 4, count * 16, count * residual_size * 4, count * 16,
                           tokens.size_bytes(), routes.size_bytes()};
    const size_t alignments[] = {4, 4, 4, 4, alignof(int32_t), alignof(Route)};
    for (size_t i = 0; i < 6; ++i) {
        if (!bytes[i]) continue;
        if (!valid(pointers[i], bytes[i], alignments[i]) || overlap(pointers[i], bytes[i], arena_.base(), arena_.capacity()) ||
            overlap(pointers[i], bytes[i], compressed_kv_cache(), size_t(capacity()) * 512 * 2) ||
            overlap(pointers[i], bytes[i], index_key_cache(), size_t(capacity()) * 128 * 2))
            throw std::invalid_argument("Text prefix requires aligned external non-cache buffers");
        if (layer8_ && (overlap(pointers[i], bytes[i], compressed_kv_cache(8), size_t(capacity()) * 512 * 2) ||
                       overlap(pointers[i], bytes[i], index_key_cache(8), size_t(capacity()) * 128 * 2)))
            throw std::invalid_argument("Text prefix caller buffer overlaps source-8 storage");
        if (layer14_ && (overlap(pointers[i], bytes[i], compressed_kv_cache(14), size_t(capacity()) * 512 * 2) ||
                        overlap(pointers[i], bytes[i], index_key_cache(14), size_t(capacity()) * 128 * 2)))
            throw std::invalid_argument("Text prefix caller buffer overlaps source-14 storage");
        if (layer20_ && (overlap(pointers[i], bytes[i], compressed_kv_cache(20), size_t(capacity(20)) * 512 * 2) ||
                        overlap(pointers[i], bytes[i], index_key_cache(20), size_t(capacity(20)) * 128 * 2)))
            throw std::invalid_argument("Text prefix caller buffer overlaps source-20 storage");
        for (int l = 0; l < layer_count_; ++l)
            if (overlap(pointers[i], bytes[i], kv_cache(l), 128 * 512 * 2))
                throw std::invalid_argument("Text prefix caller buffer overlaps a layer KV ring");
        for (size_t j = 0; j < i; ++j)
            if (overlap(pointers[i], bytes[i], pointers[j], bytes[j]))
                throw std::invalid_argument("Text prefix caller buffers overlap");
    }
    for (int32_t token : tokens)
        if (token < 0 || size_t(token) >= engram_token_count)
            throw std::invalid_argument("Text prefix original token ID out of range");
    const int n = int(count);
    auto layer_routes = [&](size_t l) { return routes.empty() ? std::span<Route>{} : routes.subspan(l * count, count); };
    try {
        stream_ = stream;
        layer0_.forward(input, incoming, residual_[0], coefficients_[0], n, start, stream, layer_routes(0));
        layer1_.forward(tokens, residual_[0], coefficients_[0], residual_[1], coefficients_[1], start, stream, layer_routes(1));
        layer2_.forward(residual_[1], coefficients_[1], residual_[0], coefficients_[0], n, start, stream, layer_routes(2));
        auto shared = layer2_.shared_batch();
        Source20AttentionBatch shared20;
        for (int l = 3; l < layer_count_; ++l) {
            const size_t from = size_t(l - 1) % 2, to = size_t(l) % 2;
            const bool last = l + 1 == layer_count_;
            if (l == 8) {
                layer8_->forward(residual_[from], coefficients_[from], residual_[to], coefficients_[to],
                    n, start, stream, layer_routes(8));
                shared = layer8_->shared_batch();
            } else if (l == 14) {
                layer14_->forward(tokens, residual_[from], coefficients_[from], last ? output : residual_[to],
                    last ? next_pre : coefficients_[to], start, stream, layer_routes(14));
                if (!last) shared = layer14_->shared_batch();
            } else if (l == 20) {
                layer20_->forward(residual_[from], coefficients_[from], residual_[to], coefficients_[to],
                    n, start, stream, layer_routes(20));
                shared20 = layer20_->shared_batch();
            } else if ((l > 20 && l <= 24) || l == 28 || l == 32 || l == 36) {
                consumers_[size_t(l)]->forward_source20(residual_[from], coefficients_[from],
                    last ? output : residual_[to], last ? next_pre : coefficients_[to],
                    n, start, shared20, stream, layer_routes(size_t(l)));
                if (l == 24 || l == 28 || l == 32 || l == 36) shared = consumers_[size_t(l)]->shared_batch();
            } else {
                consumers_[size_t(l)]->forward(residual_[from], coefficients_[from],
                    last ? output : residual_[to], last ? next_pre : coefficients_[to],
                    n, start, shared, stream, layer_routes(size_t(l)));
            }
        }
        ck(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "wait for text prefix");
        next_ += n;
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); throw; }
}
void CudaTextPrefix::reset(void* stream) {
    check_stream(stream);
    try {
        for (auto i = consumers_.rbegin(); i != consumers_.rend(); ++i) if (*i) (*i)->reset(stream);
        if (layer20_) layer20_->reset(stream);
        if (layer14_) layer14_->reset(stream);
        if (layer8_) layer8_->reset(stream);
        layer2_.reset(stream); layer1_.reset(stream); layer0_.reset(stream);
        next_ = 0; stream_ = stream;
    } catch (...) { failed_ = true; (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); throw; }
}
std::vector<CacheStats> CudaTextPrefix::stats() const {
    std::vector<CacheStats> result{layer0_.stats(), layer1_.stats(), layer2_.stats()};
    for (int l = 3; l < layer_count_; ++l)
        result.push_back(l == 20 ? layer20_->stats() : l == 14 ? layer14_->stats() : l == 8 ? layer8_->stats() : consumers_[size_t(l)]->stats());
    return result;
}
const uint16_t* CudaTextPrefix::kv_cache(int layer) const {
    switch (layer) {
        case 0: return layer0_.kv_cache();
        case 1: return layer1_.kv_cache();
        case 2: return layer2_.kv_cache();
        default:
            if (layer < 3 || layer >= layer_count_) throw std::invalid_argument("Layer outside active text prefix");
            return layer == 20 ? layer20_->kv_cache() : layer == 14 ? layer14_->kv_cache() : layer == 8 ? layer8_->kv_cache() : consumers_[size_t(layer)]->kv_cache();
    }
}
const CudaSourceDecoder& CudaTextPrefix::source_decoder(int layer) const {
    if (layer == 2) return layer2_;
    if (layer == 8 && layer8_) return *layer8_;
    throw std::invalid_argument("Source outside active text prefix");
}
int CudaTextPrefix::capacity(int layer) const {
    if (layer == 20 && layer20_) return layer20_->capacity();
    if (layer == 14 && layer14_) return layer14_->capacity();
    return source_decoder(layer).capacity();
}
int CudaTextPrefix::compressed_groups(int layer) const {
    if (layer == 20 && layer20_) return layer20_->size();
    if (layer == 14 && layer14_) return layer14_->compressed_groups();
    return source_decoder(layer).compressed_groups();
}
const uint16_t* CudaTextPrefix::compressed_kv_cache(int layer) const {
    if (layer == 20 && layer20_) return layer20_->compressed_kv_cache();
    if (layer == 14 && layer14_) return layer14_->compressed_kv_cache();
    return source_decoder(layer).compressed_kv_cache();
}
const uint16_t* CudaTextPrefix::index_key_cache(int layer) const {
    if (layer == 20 && layer20_) return layer20_->index_key_cache();
    if (layer == 14 && layer14_) return layer14_->index_key_cache();
    return source_decoder(layer).index_key_cache();
}
std::array<ngram::ReaderStats, 2> CudaTextPrefix::engram_stats(int layer) const {
    if (layer == 1) return layer1_.engram_stats();
    if (layer == 14 && layer14_) return layer14_->engram_stats();
    throw std::invalid_argument("Engram layer outside active text prefix");
}
size_t CudaTextPrefix::device_bytes() const {
    size_t bytes = layer0_.device_bytes() + layer1_.device_bytes() + layer2_.device_bytes() + arena_.capacity();
    if (layer8_) bytes += layer8_->device_bytes();
    if (layer14_) bytes += layer14_->device_bytes();
    if (layer20_) bytes += layer20_->device_bytes();
    for (const auto& layer : consumers_) if (layer) bytes += layer->device_bytes();
    return bytes;
}
} // namespace strata::deepseek41
