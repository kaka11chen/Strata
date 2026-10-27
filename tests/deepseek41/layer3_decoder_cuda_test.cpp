#include "decoder_test_utils.hpp"
#include "strata/deepseek41/layer3_decoder_cuda.hpp"

#include <filesystem>
#include <fstream>

namespace {
std::array<bool, 384> covered(const GgufModel& fixture) {
    std::array<bool, 384> ids{};
    for (int c = 0; c < 3; ++c)
        for (float id : tensor<float>(fixture, "case." + std::to_string(c) + ".route_ids", 0, c ? 6 : 18)) {
            if (id != std::floor(id) || id < 0 || id >= 384) throw std::runtime_error("Invalid original expert ID");
            ids[size_t(id)] = true;
        }
    return ids;
}
DecoderWeights weights(const GgufModel& f, const Geometry& g, int layer,
                       const std::vector<uint16_t>& router, const std::vector<float>& bias) {
    return {read_mhc_weights(f, g, layer, MhcBranch::attention), read_mhc_weights(f, g, layer, MhcBranch::ffn),
            read_attention_weights(f, g, layer), read_shared_expert(f, g, layer), router, bias};
}
// Fixed local oracle interchange: little-endian DS41PAIR + version/count,
// followed by input2, incoming2, output2, next2 as four contiguous F32 arrays.
void capture(const char* fixture_path, const char* routed_path, const char* output_path) {
    if (std::filesystem::exists(output_path)) throw std::runtime_error("Handoff file already exists");
    const auto g = geometry();
    const auto f = GgufModel::open(fixture_path), r = GgufModel::open(routed_path);
    const auto router = tensor<uint16_t>(r, "router.weight", 30, 384 * hidden);
    const auto bias = tensor<float>(r, "router.bias", 0, 384);
    const auto w = weights(f, g, 2, router, bias); const ExpertIndex index(r, g, {2});
    CoveredSource source(g, index, covered(f));
    CudaLayer2Decoder decoder(g, w, read_compressor_weights(f, g, 2), read_index_key_weights(f, g, 2),
                              read_index_query_weights(f, g, 2), source, 6, 3, 0, 3);
    Stream stream; core::DeviceArena io(1024 * 1024);
    auto* x = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* pre = static_cast<float*>(io.alloc(3 * 16));
    auto* y = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* next = static_cast<float*>(io.alloc(3 * 16));
    std::array<std::vector<float>, 4> arrays;
    for (int c = 0; c < 3; ++c) {
        const int n = c ? 1 : 3, pos = c ? c + 2 : 0;
        auto a = tensor<float>(f, "case." + std::to_string(c) + ".input", 0, size_t(n) * residual_size);
        auto b = tensor<float>(f, "case." + std::to_string(c) + ".incoming", 0, size_t(n) * 4);
        upload(x, a, stream.value); upload(pre, b, stream.value);
        decoder.forward(x, pre, y, next, n, pos, stream.value);
        const std::array<std::vector<float>, 4> values{a, b, download(y, size_t(n) * residual_size, stream.value), download(next, size_t(n) * 4, stream.value)};
        for (size_t i = 0; i < 4; ++i) arrays[i].insert(arrays[i].end(), values[i].begin(), values[i].end());
    }
    std::ofstream out(output_path, std::ios::binary); out.exceptions(std::ios::badbit | std::ios::failbit);
    out.write("DS41PAIR", 8); const uint32_t header[]{1, 5}; out.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const auto& a : arrays) out.write(reinterpret_cast<const char*>(a.data()), std::streamsize(a.size() * 4));
    out.close(); check(decoder.next_position() == 5, "captured actual complete layer-2 output and same-token coefficients");
}
void synthetic() {
    const auto g = geometry(); SyntheticWeights storage;
    // Input-dependent FFN coefficients make a token-shift or constant handoff visible.
    storage.ffn_mhc.scale[0] = .15f;
    for (int row = 0; row < 4; ++row) storage.ffn_mhc.fn[size_t(row) * residual_size + row * hidden + 17] = 1.f;
    const auto w2 = storage.view(2), w3 = storage.view(3);
    ZeroSource s2(g), s3(g, 3);
    constexpr int count = 7, first = 5;
    std::vector<float> input(count * residual_size), incoming(count * 4);
    for (int t = 0; t < count; ++t)
        for (int s = 0; s < 4; ++s) {
            incoming[size_t(t) * 4 + s] = .1f + .13f * float((t + s) % 7);
            for (int d = 0; d < hidden; ++d)
                input[size_t(t) * residual_size + s * hidden + d] = bf(std::sin(float(d) * .03f + float(t) * .2f + float(s) * .7f) + .125f * float(s));
        }
    Stream stream, other; core::DeviceArena io(8 * 1024 * 1024);
    auto* x = static_cast<float*>(io.alloc(count * residual_size * 4));
    auto* pre = static_cast<float*>(io.alloc(count * 16));
    auto* y2 = static_cast<float*>(io.alloc(first * residual_size * 4));
    auto* p2 = static_cast<float*>(io.alloc(first * 16));
    auto* y3 = static_cast<float*>(io.alloc(first * residual_size * 4));
    auto* p3 = static_cast<float*>(io.alloc(first * 16));
    auto* norm = static_cast<float*>(io.alloc(first * hidden * 4));
    auto* ay = static_cast<float*>(io.alloc(first * hidden * 4));
    upload(x, input, stream.value); upload(pre, incoming, stream.value);
    CudaLayer2Decoder producer(g, w2, storage.compressor(), storage.index_key(), storage.query(), s2, 6, 4, 0, first);
    CudaLayer3Decoder consumer(g, w3, s3, 2);
    CudaLayer3Attention reference(w3.attention); CudaSharedExpert shared(g, w3.shared);
    const auto bytes2 = producer.device_bytes(), bytes3 = consumer.device_bytes();
    const auto empty = cache(consumer.kv_cache(), stream.value);
    std::vector<std::vector<float>> saved, saved_pre;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) { producer.reset(stream.value); consumer.reset(stream.value); reference.reset(stream.value); }
        SharedAttentionBatch old;
        for (int pos = 0; pos < count;) {
            const int n = pos ? 1 : first;
            producer.forward(x + size_t(pos) * residual_size, pre + size_t(pos) * 4, y2, p2, n, pos, stream.value);
            const auto batch = producer.shared_batch(); const auto source_history = history(producer, stream.value);
            if (pos) rejects("source advance invalidates prior decoder publication", [&] { consumer.forward(y2, p2, y3, p3, n, pos, old, stream.value); });
            if (!pass && !pos) {
                const std::vector<float> sentinel(first * residual_size, -37.f); upload(y3, sentinel, stream.value);
                rejects("unpublished batch rejected before binding", [&] { consumer.forward(y2, p2, y3, p3, n, pos, {}, other.value); });
                rejects("publication stream mismatch", [&] { consumer.forward(y2, p2, y3, p3, n, pos, batch, other.value); });
                rejects("publication count mismatch", [&] { consumer.forward(y2, p2, y3, p3, n - 1, pos, batch, stream.value); });
                rejects("publication position mismatch", [&] { consumer.forward(y2, p2, y3, p3, 1, 1, batch, stream.value); });
                rejects("null stream", [&] { consumer.forward(y2, p2, y3, p3, n, pos, batch, nullptr); });
                rejects("zero tokens", [&] { consumer.forward(y2, p2, y3, p3, 0, pos, batch, stream.value); });
                rejects("sequence overflow", [&] { consumer.forward(y2, p2, y3, p3, 1048577, pos, batch, stream.value); });
                rejects("unaligned input", [&] { consumer.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(y2) + 1), p2, y3, p3, n, pos, batch, stream.value); });
                rejects("pointer overflow", [&] { consumer.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), p2, y3, p3, n, pos, batch, stream.value); });
                std::array<Route, 1> wrong;
                rejects("route span dimensions", [&] { consumer.forward(y2, p2, y3, p3, n, pos, batch, stream.value, wrong); });
                const std::array<float*, 4> pointers{y2, p2, y3, p3};
                for (size_t i = 0; i < 4; ++i) {
                    auto a = pointers; a[i] = nullptr;
                    rejects("null decoder buffer", [&] { consumer.forward(a[0], a[1], a[2], a[3], n, pos, batch, stream.value); });
                    for (size_t j = 0; j < i; ++j) {
                        a = pointers; a[i] = a[j];
                        rejects("all decoder buffer pairs disjoint", [&] { consumer.forward(a[0], a[1], a[2], a[3], n, pos, batch, stream.value); });
                    }
                    for (const auto* view : {consumer.kv_cache(), producer.kv_cache(), producer.compressed_kv_cache(), producer.index_key_cache()}) {
                        a = pointers; a[i] = reinterpret_cast<float*>(const_cast<uint16_t*>(view));
                        rejects("all four buffers checked against consumer/source caches", [&] { consumer.forward(a[0], a[1], a[2], a[3], n, pos, batch, stream.value); });
                    }
                }
                check(consumer.next_position() == 0 && s3.calls == 0 && same(cache(consumer.kv_cache(), stream.value), empty) &&
                      same(download(y3, sentinel.size(), stream.value), sentinel) && history(producer, stream.value) == source_history,
                      "all preflight rejections preserve both layers and output");
            }
            const auto residual = download(y2, size_t(n) * residual_size, stream.value);
            const auto coeff = download(p2, size_t(n) * 4, stream.value);
            upload(norm, collapse_norm(residual, coeff), stream.value);
            reference.forward(norm, ay, n, pos, batch, stream.value);
            const auto middle = join(residual, download(ay, size_t(n) * hidden, stream.value), storage.attn_mhc.post(n), storage.attn_mhc.comb(n));
            std::vector<float> attn_pre(size_t(n) * 4); const auto fixed = storage.attn_mhc.pre();
            for (size_t i = 0; i < attn_pre.size(); ++i) attn_pre[i] = fixed[i % 4];
            upload(norm, collapse_norm(middle, attn_pre), stream.value);
            for (int t = 0; t < n; ++t) shared.forward(norm + size_t(t) * hidden, ay + size_t(t) * hidden, stream.value);
            const auto expected = join(middle, download(ay, size_t(n) * hidden, stream.value), storage.ffn_mhc.post(n), storage.ffn_mhc.comb(n));
            std::vector<Route> routes(static_cast<size_t>(n));
            consumer.forward(y2, p2, y3, p3, n, pos, batch, stream.value, routes);
            const auto actual = download(y3, size_t(n) * residual_size, stream.value), next = download(p3, size_t(n) * 4, stream.value);
            compare(actual, expected, "decoder pair vs CPU joins and standalone components");
            compare(next, cpu_pre(middle, w3.ffn_mhc), "same-token next-layer coefficient handoff", true);
            check(same(cache(consumer.kv_cache(), stream.value), cache(reference.kv_cache(), stream.value)), "consumer window matches standalone batch attention exactly");
            check(source_history == history(producer, stream.value), "consumer does not mutate shared source caches");
            bool ids = true;
            for (const auto& r : routes) for (int k = 0; k < 6; ++k) ids &= r.ids[size_t(k)] == k && std::abs(r.weights[size_t(k)] - .25f) <= 3e-8f;
            check(ids, "layer-3 native MoE keeps original route IDs and weights");
            check(consumer.next_position() == pos + n && producer.next_position() == pos + n &&
                  consumer.device_bytes() == bytes3 && producer.device_bytes() == bytes2, "pair positions and fixed allocation");
            const size_t c = pos ? size_t(pos - first + 1) : 0;
            if (!pass) { saved.push_back(actual); saved_pre.push_back(next); }
            else check(same(actual, saved[c]) && same(next, saved_pre[c]), "pair reset/replay exact across cache eviction");
            old = batch; pos += n;
        }
    }
    check(s3.calls == 84 && consumer.stats().misses == 84 && consumer.stats().evictions == 82, "two-slot consumer exercises Strata cache eviction");
    const auto source_history = history(producer, stream.value); consumer.reset(stream.value);
    check(cache(consumer.kv_cache(), stream.value) == empty && source_history == history(producer, stream.value), "consumer reset clears own ring only");
    const auto stale = producer.shared_batch(); producer.reset(stream.value);
    rejects("source reset invalidates borrowed batch", [&] { consumer.forward(y2, p2, y3, p3, 1, 0, stale, stream.value); });
    producer.forward(x, pre, y2, p2, first, 0, stream.value);
    auto shifted = download(p2, first * 4, stream.value); std::rotate(shifted.begin(), shifted.begin() + 4, shifted.end()); upload(p2, shifted, stream.value);
    consumer.forward(y2, p2, y3, p3, first, 0, producer.shared_batch(), stream.value);
    check(!same(download(y3, first * residual_size, stream.value), saved[0]), "shifting coefficients across tokens changes output");
    rejects("cross-stream reset", [&] { consumer.reset(other.value); });
    rejects("wrong shared-source topology", [&] { auto bad = g; bad.sources[3].kv = 3; CudaLayer3Decoder invalid(bad, w3, s3, 2); });
    rejects("wrong index source", [&] { auto bad = g; bad.sources[3].index = 3; CudaLayer3Decoder invalid(bad, w3, s3, 2); });
    rejects("wrong geometry", [&] { auto bad = g; bad.hidden = 64; CudaLayer3Decoder invalid(bad, w3, s3, 2); });
    rejects("wrong attention layer", [&] { CudaLayer3Decoder invalid(g, w2, s3, 2); });
    {
        ZeroSource missing(g, 3), healthy_source(g, 3);
        CudaLayer3Decoder broken(g, w3, missing, 2), healthy(g, w3, healthy_source, 6);
        producer.reset(stream.value); producer.forward(x, pre, y2, p2, 3, 0, stream.value);
        const auto batch = producer.shared_batch();
        broken.forward(y2, p2, y3, p3, 3, 0, batch, stream.value);
        healthy.forward(y2, p2, y3, p3, 3, 0, batch, stream.value);
        producer.forward(x, pre, y2, p2, 1, 3, stream.value);
        const auto current = producer.shared_batch(), h = current;
        const auto before = history(producer, stream.value); missing.missing = true;
        rejects("consumer expert failure after attention", [&] { broken.forward(y2, p2, y3, p3, 1, 3, current, stream.value); }, true);
        check(broken.next_position() == 3 && producer.next_position() == 4 && before == history(producer, stream.value), "failure never completes consumer or damages source");
        rejects("failed decoder stays poisoned", [&] { broken.forward(y2, p2, y3, p3, 1, 3, current, stream.value); }, true);
        rejects("reset cannot recover failed decoder", [&] { broken.reset(stream.value); }, true);
        healthy.forward(y2, p2, y3, p3, 1, 3, h, stream.value);
        check(healthy.next_position() == 4 && before == history(producer, stream.value), "healthy consumer can still use source publication after peer failure");
    }
    std::printf("synthetic_pair_source_bytes=%zu consumer_bytes=%zu total_bytes=%zu\n", bytes2, bytes3, bytes2 + bytes3);
}
void real(const char* layer2_path, const char* routed2_path, const char* layer3_path, const char* routed3_path) {
    const auto g = geometry();
    const auto f2 = GgufModel::open(layer2_path), r2 = GgufModel::open(routed2_path);
    const auto f3 = GgufModel::open(layer3_path), r3 = GgufModel::open(routed3_path);
    const auto* arch = f3.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-layer3-decoder-fixture") throw std::runtime_error("Expected layer-3 decoder fixture");
    const auto router2 = tensor<uint16_t>(r2, "router.weight", 30, 384 * hidden), router3 = tensor<uint16_t>(r3, "router.weight", 30, 384 * hidden);
    const auto bias2 = tensor<float>(r2, "router.bias", 0, 384), bias3 = tensor<float>(r3, "router.bias", 0, 384);
    const auto w2 = weights(f2, g, 2, router2, bias2), w3 = weights(f3, g, 3, router3, bias3);
    const ExpertIndex i2(r2, g, {2}), i3(r3, g, {3});
    const auto ids2 = covered(f2), ids3 = covered(f3);
    Stream stream; core::DeviceArena io(4 * 1024 * 1024);
    auto* x = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* pre = static_cast<float*>(io.alloc(3 * 16));
    auto* y2 = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* p2 = static_cast<float*>(io.alloc(3 * 16));
    auto* y3 = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* p3 = static_cast<float*>(io.alloc(3 * 16));
    auto* norm = static_cast<float*>(io.alloc(3 * hidden * 4));
    auto* ay = static_cast<float*>(io.alloc(3 * hidden * 4));
    auto* middle = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* zero = static_cast<float*>(io.alloc(hidden * 4)); ck(cudaMemsetAsync(zero, 0, hidden * 4, stream.value));
    auto* dummy = static_cast<float*>(io.alloc(residual_size * 4));
    std::vector<std::vector<float>> expected;
    {
        CoveredSource s(g, i3, ids3, 3); CudaMoE moe(g, 3, router3, bias3, s, 6, w3.shared);
        for (int c = 0; c < 3; ++c) {
            const int n = c ? 1 : 3; const auto prefix = "case." + std::to_string(c) + ".";
            check(same(tensor<float>(f3, prefix + "source_input", 0, size_t(n) * residual_size), tensor<float>(f2, prefix + "input", 0, size_t(n) * residual_size)) &&
                  same(tensor<float>(f3, prefix + "source_incoming", 0, size_t(n) * 4), tensor<float>(f2, prefix + "incoming", 0, size_t(n) * 4)), "handoff uses established layer-2 fixture inputs");
            const auto normalized = tensor<float>(f3, prefix + "ffn_input", 0, size_t(n) * hidden);
            const auto ids = tensor<float>(f3, prefix + "route_ids", 0, size_t(n) * 6);
            const auto mixing = tensor<float>(f3, prefix + "route_weights", 0, size_t(n) * 6);
            std::vector<float> outputs;
            for (int t = 0; t < n; ++t) {
                ck(cudaMemcpyAsync(norm, normalized.data() + size_t(t) * hidden, hidden * 4, cudaMemcpyHostToDevice, stream.value));
                const auto route = moe.forward(norm, ay, stream.value); const auto value = download(ay, hidden, stream.value);
                outputs.insert(outputs.end(), value.begin(), value.end());
                bool correct = true;
                for (size_t k = 0; k < 6; ++k) correct &= route.ids[k] == int32_t(ids[size_t(t) * 6 + k]) && std::abs(route.weights[k] - mixing[size_t(t) * 6 + k]) <= 2e-6f;
                check(correct, prefix + "native layer-3 MoE on oracle input retains original routes");
            }
            const auto residual = tensor<float>(f3, prefix + "after_attention", 0, size_t(n) * residual_size);
            compare(cpu_pre(residual, w3.ffn_mhc), tensor<float>(f3, prefix + "next_pre", 0, size_t(n) * 4), prefix + "independent CPU coefficients vs pinned graph", true);
            expected.push_back(join(residual, outputs, tensor<float>(f3, prefix + "post", 0, size_t(n) * 4), tensor<float>(f3, prefix + "comb", 0, size_t(n) * 16)));
        }
    }
    std::vector<std::vector<float>> saved, saved_pre;
    for (int slots : {6, 2}) {
        CoveredSource s2(g, i2, ids2), s3(g, i3, ids3, 3);
        CudaLayer2Decoder producer(g, w2, read_compressor_weights(f2, g, 2), read_index_key_weights(f2, g, 2),
                                  read_index_query_weights(f2, g, 2), s2, slots, 3, 0, 3);
        CudaLayer3Decoder consumer(g, w3, s3, slots);
        // Standalone primitives independently measure the native attention boundary;
        // the CPU graph still supplies the main expected output and original routes.
        CudaLayer3Attention reference(w3.attention); CudaMhc mhc(g, w3.attention_mhc);
        const size_t bytes2 = producer.device_bytes(), bytes3 = consumer.device_bytes();
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) { producer.reset(stream.value); consumer.reset(stream.value); reference.reset(stream.value); }
            for (int c = 0; c < 3; ++c) {
                const int n = c ? 1 : 3, pos = c ? c + 2 : 0, groups = (pos + n) / 2;
                const auto prefix = "case." + std::to_string(c) + ".";
                const auto label = "real pair slots=" + std::to_string(slots) + " pass=" + std::to_string(pass) + " case=" + std::to_string(c);
                upload(x, tensor<float>(f3, prefix + "source_input", 0, size_t(n) * residual_size), stream.value);
                upload(pre, tensor<float>(f3, prefix + "source_incoming", 0, size_t(n) * 4), stream.value);
                producer.forward(x, pre, y2, p2, n, pos, stream.value);
                const auto batch = producer.shared_batch(); const auto before = history(producer, stream.value);
                check(same(download(y2, size_t(n) * residual_size, stream.value), tensor<float>(f3, prefix + "input", 0, size_t(n) * residual_size)), label + " actual source residual equals oracle handoff byte-for-byte");
                check(same(download(p2, size_t(n) * 4, stream.value), tensor<float>(f3, prefix + "incoming", 0, size_t(n) * 4)), label + " actual same-token source coefficients equal oracle handoff byte-for-byte");
                compare(cache(producer.kv_cache(), stream.value), tensor<float>(f3, prefix + "source_cache", 0, 128 * 512), label + " source window vs CPU graph");
                compare(cache(producer.compressed_kv_cache(), stream.value, size_t(groups) * 512), tensor<float>(f3, prefix + "compressed", 0, size_t(groups) * 512), label + " borrowed compressed KV vs CPU graph", false, true);
                compare(cache(producer.index_key_cache(), stream.value, size_t(groups) * 128), tensor<float>(f3, prefix + "keys", 0, size_t(groups) * 128), label + " borrowed index keys vs CPU graph", false, true);
                for (int t = 0; t < n; ++t) {
                    const auto state = mhc.prepare(y2 + size_t(t) * residual_size, p2 + size_t(t) * 4, stream.value);
                    ck(cudaMemcpyAsync(norm + size_t(t) * hidden, state.normalized, hidden * 4, cudaMemcpyDeviceToDevice, stream.value));
                    mhc.finish(zero, dummy, stream.value);
                }
                reference.forward(norm, ay, n, pos, batch, stream.value);
                for (int t = 0; t < n; ++t) {
                    (void)mhc.prepare(y2 + size_t(t) * residual_size, p2 + size_t(t) * 4, stream.value);
                    mhc.finish(ay + size_t(t) * hidden, middle + size_t(t) * residual_size, stream.value);
                }
                const auto boundary = download(middle, size_t(n) * residual_size, stream.value);
                const auto reference_boundary = tensor<float>(f3, prefix + "after_attention", 0, size_t(n) * residual_size);
                compare(boundary, reference_boundary, label + " standalone native attention residual vs CPU graph");
                std::vector<Route> routes(static_cast<size_t>(n));
                // Direct GPU handoff: no host substitution/copy of residuals or coefficients.
                consumer.forward(y2, p2, y3, p3, n, pos, batch, stream.value, routes);
                const auto actual = download(y3, size_t(n) * residual_size, stream.value), next = download(p3, size_t(n) * 4, stream.value);
                compare(actual, expected[size_t(c)], label + " layer-3 CPU graph + native Q8_1 MoE output");
                compare(next, cpu_pre(boundary, w3.ffn_mhc), label + " strict same-input next-layer coefficients", true);
                compare_propagated(next, tensor<float>(f3, prefix + "next_pre", 0, size_t(n) * 4), boundary, reference_boundary, w3.ffn_mhc, label + " coefficient propagation bound");
                compare(cache(consumer.kv_cache(), stream.value), tensor<float>(f3, prefix + "cache", 0, 128 * 512), label + " consumer window vs CPU graph");
                check(same(cache(consumer.kv_cache(), stream.value), cache(reference.kv_cache(), stream.value)), label + " consumer window equals standalone attention exactly");
                check(before == history(producer, stream.value), label + " consumer leaves all source caches unchanged");
                const auto ids = tensor<float>(f3, prefix + "route_ids", 0, size_t(n) * 6), mixing = tensor<float>(f3, prefix + "route_weights", 0, size_t(n) * 6);
                bool correct = true;
                for (int t = 0; t < n; ++t) for (size_t k = 0; k < 6; ++k)
                    correct &= routes[size_t(t)].ids[k] == int32_t(ids[size_t(t) * 6 + k]) && std::abs(routes[size_t(t)].weights[k] - mixing[size_t(t) * 6 + k]) <= 1e-4f;
                check(correct, label + " original layer-3 expert IDs and routing weights");
                check(producer.next_position() == pos + n && consumer.next_position() == pos + n && producer.compressed_groups() == groups &&
                      producer.device_bytes() == bytes2 && consumer.device_bytes() == bytes3, label + " both completed positions and fixed device memory");
                if (slots == 6 && !pass) { saved.push_back(actual); saved_pre.push_back(next); }
                else check(same(actual, saved[size_t(c)]) && same(next, saved_pre[size_t(c)]), label + " pair output and coefficients invariant to reset/eviction");
            }
        }
        const std::array<CacheStats, 2> stats{producer.stats(), consumer.stats()};
        for (size_t l = 0; l < 2; ++l) {
            const auto& s = stats[l]; const int64_t reads = l ? s3.reads() : s2.reads();
            check(s.hits + s.misses == 60 && s.expert_h2d_bytes == s.misses * nvfp4_expert_bytes(g) &&
                  s.router_d2h_bytes == 10 * 384 * 4 && uint64_t(reads) == s.misses && s.evictions > 0, "pair uses Strata expert cache accounting layer=" + std::to_string(l + 2));
            std::printf("real_pair_slots=%d layer=%zu device_bytes=%zu expert_h2d_bytes=%llu hits=%llu misses=%llu evictions=%llu\n", slots, l + 2,
                        l ? bytes3 : bytes2, (unsigned long long)s.expert_h2d_bytes, (unsigned long long)s.hits, (unsigned long long)s.misses, (unsigned long long)s.evictions);
        }
    }
}
} // namespace
int main(int argc, char** argv) {
    if (!(argc == 1 || (argc == 5 && std::string(argv[1]) == "--capture") || (argc == 6 && std::string(argv[1]) == "--real"))) {
        std::fprintf(stderr, "usage: deepseek41_layer3_decoder_cuda_test [--capture LAYER2.gguf ROUTED2.gguf HANDOFF.bin | --real LAYER2.gguf ROUTED2.gguf LAYER3.gguf ROUTED3.gguf]\n"); return 2;
    }
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 1) synthetic();
        else if (argc == 5) capture(argv[2], argv[3], argv[4]);
        else real(argv[2], argv[3], argv[4], argv[5]);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures); return failures ? 1 : 0;
}
