#include "decoder_test_utils.hpp"

namespace {
void source_groups(const Geometry& g,const SyntheticWeights& w,const float* input,float* output,cudaStream_t stream) {
    rejects("empty publication has no source identity",[&]{(void)SharedAttentionBatch{}.source_layer();});
    for(int owner:{2,8}) {
        const auto a=w.view(owner).attention;
        rejects("mixed compressor identity",[&]{auto c=w.compressor(owner==2?8:2);CudaSourceAttention bad(a,c,w.index_key(owner),w.query(owner),2,0,3,owner);});
        rejects("mixed key identity",[&]{auto k=w.index_key(owner==2?8:2);CudaSourceAttention bad(a,w.compressor(owner),k,w.query(owner),2,0,3,owner);});
        rejects("mixed query identity",[&]{auto q=w.query(owner==2?8:2);CudaSourceAttention bad(a,w.compressor(owner),w.index_key(owner),q,2,0,3,owner);});
        ZeroSource src(g,owner);
        rejects("mixed decoder identity",[&]{CudaSourceDecoder bad(g,w.view(owner),w.compressor(owner==2?8:2),w.index_key(owner),w.query(owner),src,2,2,0,3,owner);});
    }
    rejects("default attention owner remains 2",[&]{CudaSourceAttention bad(w.view(8).attention,w.compressor(8),w.index_key(8),w.query(8),2);});
    rejects("future owner 20 is not enabled",[&]{CudaSourceAttention bad(w.view(20).attention,w.compressor(20),w.index_key(20),w.query(20),2,0,3,20);});
    auto g14=g;g14.compress_ratios[14]=2;g14.sources[14]={14,14,-1};ZeroSource src14(g14,14);
    rejects("source-14 decoder awaits Engram assembly",[&]{CudaSourceDecoder bad(g14,w.view(14),w.compressor(14),w.index_key(14),w.query(14),src14,2,2,0,3,14);});
    check(src14.calls==0,"unsupported source-14 decoder rejects before expert I/O");
    SharedAttentionBatch expired;
    {
        CudaSourceAttention a(w.view(2).attention,w.compressor(),w.index_key(),w.query(),2,0,3);
        CudaSourceAttention b(w.view(8).attention,w.compressor(8),w.index_key(8),w.query(8),2,0,3,8);
        check(a.kv_cache()!=b.kv_cache() && a.compressed_kv_cache()!=b.compressed_kv_cache() && a.index_key_cache()!=b.index_key_cache(),"source groups own distinct window/KV/key buffers");
        check(a.device_bytes()==b.device_bytes(),"source identity adds no device allocation");
        a.forward(input,output,3,0,stream);const auto p2=a.shared_batch();const auto h2=history(a,stream);
        b.forward(input,output,3,0,stream);const auto p8=b.shared_batch();const auto h8=history(b,stream);
        check(p2.source_layer()==2 && p8.source_layer()==8 && h2==history(a,stream),"source-8 publication preserves independent source-2 state");
        rejects("source-8 rejected call preserves current publication",[&]{b.forward(input,output,1,0,stream);});
        check(p8.source_layer()==8 && h8==history(b,stream),"invalid source call preserves all owned histories and identity");
        CudaSourceAttention third(w.view(14).attention,w.compressor(14),w.index_key(14),w.query(14),2,0,3,14);
        third.forward(input,output,3,0,stream);const auto p14=third.shared_batch();const auto h14=history(third,stream);
        check(p14.source_layer()==14 && h2==history(a,stream) && h8==history(b,stream),"source14 publication preserves earlier groups");
        check(third.device_bytes()==a.device_bytes() && third.compressed_kv_cache()!=a.compressed_kv_cache() && third.compressed_kv_cache()!=b.compressed_kv_cache(),"third source identity adds no extra shared cache allocation");
        for(int consumer_layer=3;consumer_layer<=19;++consumer_layer) {
            if(consumer_layer==8 || consumer_layer==14)continue;
            CudaSharedAttention consumer(w.view(consumer_layer).attention,0,consumer_layer);
            const auto initial=cache(consumer.kv_cache(),stream);std::vector<float> sentinel(3*hidden,-53.f);upload(output,sentinel,stream);
            const int owner=consumer_layer<8?2:consumer_layer<14?8:14;
            for(const auto& wrong:{p2,p8,p14}) if(wrong.source_layer()!=owner)
                rejects("cross-group publication rejected consumer="+std::to_string(consumer_layer),[&]{consumer.forward(input,output,3,0,wrong,stream);});
            check(consumer.next_position()==0 && initial==cache(consumer.kv_cache(),stream) && same(download(output,3*hidden,stream),sentinel),"cross-group rejection leaves consumer ring, position and output untouched");
            consumer.forward(input,output,3,0,consumer_layer<8?p2:consumer_layer<14?p8:p14,stream);
            check(consumer.next_position()==3 && h2==history(a,stream) && h8==history(b,stream) && h14==history(third,stream),"consumer accepts its original publication and preserves all source groups");
        }
        {
            CudaSharedAttention tail(w.view(19).attention,0,19);
            tail.forward(input,output,3,0,p14,stream);
            third.forward(input,output,1,3,stream);const auto old14=third.shared_batch();
            rejects("source14 advance expires original publication",[&]{(void)p14.source_layer();});
            third.reset(stream);third.forward(input,output,3,0,stream);third.forward(input,output,1,3,stream);
            const auto own=cache(tail.kv_cache(),stream);std::vector<float> sentinel(hidden,-61.f);upload(output,sentinel,stream);
            rejects("source14 reset expires decode publication",[&]{tail.forward(input,output,1,3,old14,stream);});
            rejects("consumer19 requires reset after source14 sequence change",[&]{tail.forward(input,output,1,3,third.shared_batch(),stream);});
            check(tail.next_position()==3 && own==cache(tail.kv_cache(),stream) && same(download(output,hidden,stream),sentinel),"source14 stale-sequence rejection preserves consumer output and ring");
            tail.reset(stream);third.reset(stream);third.forward(input,output,3,0,stream);tail.forward(input,output,3,0,third.shared_batch(),stream);
            check(h14==history(third,stream) && h2==history(a,stream) && h8==history(b,stream),"source14 and consumer replay without changing earlier groups");
        }
        a.forward(input,output,1,3,stream);const auto fresh2=a.shared_batch();
        rejects("advancing source 2 expires its old publication",[&]{(void)p2.source_layer();});
        check(p8.source_layer()==8 && h8==history(b,stream),"advancing source 2 preserves source 8");
        b.reset(stream);rejects("source-8 reset expires publication",[&]{(void)p8.source_layer();});
        check(fresh2.source_layer()==2 && a.next_position()==4,"source-8 reset preserves source 2");
        b.forward(input,output,3,0,stream);expired=b.shared_batch();a.reset(stream);
        check(expired.source_layer()==8 && b.next_position()==3,"source-2 reset preserves the new source-8 sequence");
    }
    rejects("destroying the owner expires its source identity",[&]{(void)expired.source_layer();});
}
void synthetic(int layer) {
    const auto g = geometry(); SyntheticWeights storage; const auto w = storage.view(layer); ZeroSource source(g,layer);
    constexpr int count = 130, first = count - 1;
    std::vector<float> input(count * residual_size), incoming(count * 4);
    for (int t = 0; t < count; ++t) {
        for (int s = 0; s < 4; ++s) {
            incoming[size_t(t) * 4 + s] = .1f + .13f * float((t + s) % 7);
            for (int d = 0; d < hidden; ++d)
                input[size_t(t) * residual_size + s * hidden + d] = bf(std::sin(float(d) * .03f + float(t) * .2f + float(s) * .7f) + .1f * std::cos(float(d) * .07f) + .125f * float(s));
        }
    }
    Stream stream, other;
    core::DeviceArena io(size_t(count) * hidden * 48 + 8192);
    auto* x = static_cast<float*>(io.alloc(input.size() * 4));
    auto* pre = static_cast<float*>(io.alloc(incoming.size() * 4));
    auto* y = static_cast<float*>(io.alloc(input.size() * 4));
    auto* next = static_cast<float*>(io.alloc(incoming.size() * 4));
    auto* norm = static_cast<float*>(io.alloc(size_t(count) * hidden * 4));
    auto* attn_output = static_cast<float*>(io.alloc(size_t(count) * hidden * 4));
    upload(x, input, stream.value); upload(pre, incoming, stream.value);
    upload(norm, collapse_norm(input, incoming), stream.value);
    std::vector<float> expected, expected_cache; std::array<std::vector<float>, 3> expected_history;
    {
        CudaLayer2Attention attention(w.attention, storage.compressor(layer), storage.index_key(layer), storage.query(layer), count / 2, 0, 0, layer);
        attention.forward(norm, attn_output, first, 0, stream.value);
        attention.forward(norm + first * hidden, attn_output + first * hidden, 1, first, stream.value);
        auto middle = join(input, download(attn_output, size_t(count) * hidden, stream.value), storage.attn_mhc.post(count), storage.attn_mhc.comb(count));
        std::vector<float> ffn_pre(size_t(count) * 4);
        const auto pre_values = storage.attn_mhc.pre();
        for (size_t i = 0; i < ffn_pre.size(); ++i) ffn_pre[i] = pre_values[i % 4];
        upload(norm, collapse_norm(middle, ffn_pre), stream.value);
        CudaSharedExpert shared(g, w.shared);
        for (int t = 0; t < count; ++t) shared.forward(norm + size_t(t) * hidden, attn_output + size_t(t) * hidden, stream.value);
        expected = join(middle, download(attn_output, size_t(count) * hidden, stream.value), storage.ffn_mhc.post(count), storage.ffn_mhc.comb(count));
        upload(norm, collapse_norm(input, incoming), stream.value);
        expected_cache = cache(attention.kv_cache(), stream.value); expected_history = history(attention, stream.value);
    }
    CudaLayer2Decoder decoder(g, w, storage.compressor(layer), storage.index_key(layer), storage.query(layer), source, 6, count / 2, 0, first, layer);
    const auto bytes = decoder.device_bytes();
    std::vector<Route> routes(count);
    const auto initial = history(decoder, stream.value);
    const std::vector<float> sentinel(input.size(), -37.f); upload(y, sentinel, stream.value);
    rejects("shared-index capacity rejected before state change", [&] { decoder.forward(x, pre, y, next, count, 0, other.value); });
    rejects("compressed capacity rejected before state change", [&] { decoder.forward(x, pre, y, next, count + 2, 0, other.value); });
    rejects("unaligned input rejected", [&] { decoder.forward(reinterpret_cast<float*>(reinterpret_cast<char*>(x) + 1), pre, y, next, 1, 0, other.value); });
    rejects("pointer range overflow rejected", [&] { decoder.forward(reinterpret_cast<float*>(UINTPTR_MAX - 3), pre, y, next, 1, 0, other.value); });
    for (const auto* view : {decoder.kv_cache(), decoder.compressed_kv_cache(), decoder.index_key_cache()}) {
        auto* alias = reinterpret_cast<float*>(const_cast<uint16_t*>(view));
        rejects("residual/cache alias", [&] { decoder.forward(alias, pre, y, next, 1, 0, other.value); });
        rejects("incoming/cache alias", [&] { decoder.forward(x, alias, y, next, 1, 0, other.value); });
        rejects("output/cache alias", [&] { decoder.forward(x, pre, alias, next, 1, 0, other.value); });
        rejects("next_pre/cache alias", [&] { decoder.forward(x, pre, y, alias, 1, 0, other.value); });
    }
    rejects("null stream rejected", [&] { decoder.forward(x, pre, y, next, 1, 0, nullptr); });
    rejects("null input rejected", [&] { decoder.forward(nullptr, pre, y, next, 1, 0, stream.value); });
    rejects("residual input/output overlap rejected", [&] { decoder.forward(x, pre, x, next, 1, 0, stream.value); });
    rejects("coefficient input/output overlap rejected", [&] { decoder.forward(x, pre, y, pre, 1, 0, stream.value); });
    rejects("zero tokens rejected", [&] { decoder.forward(x, pre, y, next, 0, 0, stream.value); });
    rejects("missing prefill rejected", [&] { decoder.forward(x, pre, y, next, 1, 1, stream.value); });
    rejects("sequence bound checked before access", [&] { decoder.forward(x, pre, y, next, 1048577, 0, stream.value); });
    rejects("host route span must match tokens", [&] { decoder.forward(x, pre, y, next, 1, 0, stream.value, routes); });
    check(source.calls == 0 && decoder.next_position() == 0 && initial == history(decoder, stream.value) && same(download(y, input.size(), stream.value), sentinel), "validation failures leave cache and position untouched");
    decoder.forward(x, pre, y, next, first, 0, stream.value, std::span(routes).first(first));
    decoder.forward(x + first * residual_size, pre + first * 4, y + first * residual_size, next + first * 4, 1, first, stream.value, std::span(routes).last(1));
    const auto output = download(y, input.size(), stream.value);
    compare(output, expected, "129-token two-pass prefill + decode matches batch-attention/CPU-join composition");
    compare(cache(decoder.kv_cache(), stream.value), expected_cache, "decoder preserves batch-prefill KV state");
    check(expected_history == history(decoder, stream.value), "decoder prepass preserves all standalone attention caches byte-for-byte");
    const auto expected_pre = storage.ffn_mhc.pre();
    std::vector<float> expected_next(incoming.size());
    for (size_t i = 0; i < expected_next.size(); ++i) expected_next[i] = expected_pre[i % 4];
    compare(download(next, incoming.size(), stream.value), expected_next, "returns FFN coefficients for each token's next layer", true);
    bool correct_routes = true;
    for (const auto& route : routes)
        for (int k = 0; k < 6; ++k) correct_routes &= route.ids[size_t(k)] == k && std::abs(route.weights[size_t(k)] - .25f) < 1e-6f;
    check(correct_routes, "all original top-6 IDs and route weights retained");
    check(source.calls == 6 && decoder.stats().misses == 6 && decoder.stats().hits == 6 * count - 6, "expert cache reused throughout prefill and decode");
    const auto before = history(decoder, stream.value);
    rejects("noncontiguous decode rejected", [&] { decoder.forward(x, pre, y, next, 1, count + 1, stream.value); });
    rejects("second prefill rejected", [&] { decoder.forward(x, pre, y, next, 1, 0, stream.value); });
    rejects("multi-token continuation rejected", [&] { decoder.forward(x, pre, y, next, 2, count, stream.value); });
    rejects("stream switch rejected", [&] { decoder.forward(x, pre, y, next, 1, count, other.value); });
    rejects("reset cannot switch streams", [&] { decoder.reset(other.value); });
    check(decoder.next_position() == count && before == history(decoder, stream.value), "invalid continuation preserves live KV and position");
    decoder.forward(x, pre, y, next, 1, count, stream.value);
    const auto full = history(decoder, stream.value); const auto full_output = download(y, input.size(), stream.value);
    check(decoder.next_position() == count + 1 && decoder.compressed_groups() == count / 2, "full cache accepts one incomplete group token");
    rejects("next group emission rejected before mutation", [&] { decoder.forward(x, pre, y, next, 1, count + 1, stream.value); });
    check(full == history(decoder, stream.value) && same(full_output, download(y, input.size(), stream.value)), "capacity rejection preserves all caches and output");
    check(decoder.shared_batch().source_layer()==layer,"completed publication retains original source identity");
    decoder.reset(stream.value);
    check(decoder.next_position() == 0 && history(decoder, stream.value) == initial, "reset clears sequence state");
    decoder.forward(x, pre, y, next, first, 0, stream.value);
    decoder.forward(x + first * residual_size, pre + first * 4, y + first * residual_size, next + first * 4, 1, first, stream.value);
    check(same(output, download(y, input.size(), stream.value)), "reset/replay is byte-identical with warm expert cache");
    check(source.calls == 6 && decoder.stats().hits == 12 * count && decoder.device_bytes() == bytes, "reset retains immutable expert cache and fixed allocation");
    decoder.reset(stream.value);
    auto changed = incoming; std::reverse(changed.begin(), changed.begin() + 4); upload(pre, changed, stream.value);
    decoder.forward(x, pre, y, next, 1, 0, stream.value);
    check(!same(download(y, residual_size, stream.value), std::vector<float>(output.begin(), output.begin() + residual_size)), "incoming mHC coefficients affect attention/output");
    decoder.reset(stream.value);
    check(history(decoder, stream.value) == initial, "reset discards pending compressor group and clears every cache");
    rejects("reset invalidates completed publication", [&] { check(decoder.shared_batch().source_layer()==layer,"completed publication retains original source identity"); });
    rejects("wrong shared-source topology rejected", [&] { auto bad = g; bad.sources[size_t(layer)].kv = 0; CudaSourceDecoder invalid(bad, w, storage.compressor(layer), storage.index_key(layer), storage.query(layer), source, 2, 3, 0, 3, layer); });
    rejects("wrong decoder geometry rejected", [&] { auto bad = g; bad.hidden = 64; CudaSourceDecoder invalid(bad, w, storage.compressor(layer), storage.index_key(layer), storage.query(layer), source, 2, 3, 0, 3, layer); });
    {
        ZeroSource missing(g,layer);
        CudaLayer2Decoder broken(g, w, storage.compressor(layer), storage.index_key(layer), storage.query(layer), missing, 2, 3, 0, 3, layer);
        broken.forward(x, pre, y, next, 3, 0, stream.value);
        const auto published = broken.shared_batch();
        auto w3 = w.attention; w3.layer = 3; CudaLayer3Attention consumer(w3);
        if (layer==2) {
            consumer.forward(norm, attn_output, 3, 0, published, stream.value);
            check(consumer.next_position() == 3, "completed decoder publication is consumable by layer-3 attention");
        } else rejects("layer-3 consumer refuses source 8",[&]{consumer.forward(norm,attn_output,3,0,published,stream.value);});
        check(published.source_layer()==layer,"publication belongs to the explicit owner");
        missing.missing = true;
        rejects("missing original expert fails complete layer after compressor advances", [&] { broken.forward(x, pre, y, next, 1, 3, stream.value); }, true);
        check(broken.next_position() == 3 && broken.compressed_groups() == 2, "failed layer never publishes a completed position");
        rejects("failed decoder has no completed shared batch", [&] { (void)broken.shared_batch(); });
        rejects("failure expires source identity",[&]{(void)published.source_layer();});
        consumer.reset(stream.value);
        rejects("failure invalidates previously borrowed publication", [&] { consumer.forward(norm, attn_output, 3, 0, published, stream.value); });
        rejects("partial execution cannot be reused", [&] { broken.forward(x, pre, y, next, 1, 3, stream.value); }, true);
        rejects("reset cannot revive a poisoned layer", [&] { broken.reset(stream.value); }, true);
    }
    if(layer==8)source_groups(g,storage,norm,attn_output,stream.value);
    std::printf("synthetic_decoder_device_bytes=%zu\n", bytes);
}
std::vector<std::vector<float>> native_attention_boundaries(const Geometry& g, const DecoderWeights& w,
                                                           const GgufModel& fixture, cudaStream_t stream) {
    const int layer=w.attention.layer;
    CudaMhc mhc(g, w.attention_mhc);
    CudaLayer2Attention attention(w.attention, read_compressor_weights(fixture, g, layer),
                                 read_index_key_weights(fixture, g, layer), read_index_query_weights(fixture, g, layer), 3, 0, 0, layer);
    core::DeviceArena io(2 * 1024 * 1024);
    auto* x = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* incoming = static_cast<float*>(io.alloc(3 * 16));
    auto* norm = static_cast<float*>(io.alloc(3 * hidden * 4));
    auto* ay = static_cast<float*>(io.alloc(3 * hidden * 4));
    auto* middle = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* dummy = static_cast<float*>(io.alloc(residual_size * 4));
    auto* zero = static_cast<float*>(io.alloc(hidden * 4)); ck(cudaMemsetAsync(zero, 0, hidden * 4, stream));
    std::vector<std::vector<float>> boundaries;
    for (int c = 0; c < 3; ++c) {
        const int tokens = c == 0 ? 3 : 1, pos = c == 0 ? 0 : c + 2;
        const auto prefix = "case." + std::to_string(c) + ".";
        upload(x, tensor<float>(fixture, prefix + "input", 0, size_t(tokens) * residual_size), stream);
        upload(incoming, tensor<float>(fixture, prefix + "incoming", 0, size_t(tokens) * 4), stream);
        for (int t = 0; t < tokens; ++t) {
            const auto state = mhc.prepare(x + size_t(t) * residual_size, incoming + size_t(t) * 4, stream);
            ck(cudaMemcpyAsync(norm + size_t(t) * hidden, state.normalized, hidden * 4, cudaMemcpyDeviceToDevice, stream));
            mhc.finish(zero, dummy, stream); // close the pair; this dummy join is not a reference output
        }
        attention.forward(norm, ay, tokens, pos, stream);
        for (int t = 0; t < tokens; ++t) {
            (void)mhc.prepare(x + size_t(t) * residual_size, incoming + size_t(t) * 4, stream);
            mhc.finish(ay + size_t(t) * hidden, middle + size_t(t) * residual_size, stream);
        }
        boundaries.push_back(download(middle, size_t(tokens) * residual_size, stream));
        const auto reference = tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size);
        compare(boundaries.back(), reference, prefix + "native attention residual boundary");
        compare(cpu_pre(reference, w.ffn_mhc), tensor<float>(fixture, prefix + "next_pre", 0, size_t(tokens) * 4),
                prefix + "independent CPU coefficients match pinned oracle on same input", true);
    }
    return boundaries;
}

void real(const char* fixture_path, const char* routed_path, int layer) {
    const auto g = geometry();
    const auto fixture = GgufModel::open(fixture_path), routed = GgufModel::open(routed_path);
    const auto* arch = fixture.meta().get("general.architecture");
    if (!arch || arch->s != "deepseek41-layer"+std::to_string(layer)+"-decoder-fixture") throw std::runtime_error("Expected prepared decoder fixture");
    const auto router = tensor<uint16_t>(routed, "router.weight", 30, hidden * 384);
    const auto bias = tensor<float>(routed, "router.bias", 0, 384);
    DecoderWeights weights{read_mhc_weights(fixture, g, layer, MhcBranch::attention), read_mhc_weights(fixture, g, layer, MhcBranch::ffn),
        read_attention_weights(fixture, g, layer), read_shared_expert(fixture, g, layer), router, bias};
    const ExpertIndex index(routed, g, {layer});
    std::array<bool, 384> populated{};
    for (int c = 0; c < 3; ++c)
        for (float id : tensor<float>(fixture, "case." + std::to_string(c) + ".route_ids", 0, c == 0 ? 18 : 6)) {
            if (id != std::floor(id) || id < 0 || id >= 384) throw std::runtime_error("Invalid original expert ID in fixture");
            populated[size_t(id)] = true;
        }
    Stream stream, other;
    core::DeviceArena io(3 * size_t(residual_size) * 12 + 8192);
    auto* x = static_cast<float*>(io.alloc(3 * size_t(residual_size) * 4));
    auto* y = static_cast<float*>(io.alloc(3 * size_t(residual_size) * 4));
    auto* pre = static_cast<float*>(io.alloc(3 * 16));
    auto* next = static_cast<float*>(io.alloc(3 * 16));
    auto* norm = static_cast<float*>(io.alloc(hidden * 4));
    auto* moe_output = static_cast<float*>(io.alloc(hidden * 4));
    const auto boundaries = native_attention_boundaries(g, weights, fixture, stream.value);
    std::vector<std::vector<float>> expected;
    std::vector<std::vector<Route>> expected_routes;
    {
        CoveredSource source(g, index, populated, layer); CudaMoE moe(g, layer, router, bias, source, 6, weights.shared);
        for (int c = 0; c < 3; ++c) {
            const int tokens = c == 0 ? 3 : 1;
            const auto prefix = "case." + std::to_string(c) + ".";
            const auto normalized = tensor<float>(fixture, prefix + "ffn_input", 0, size_t(tokens) * hidden);
            const auto ids = tensor<float>(fixture, prefix + "route_ids", 0, size_t(tokens) * 6);
            const auto mixing = tensor<float>(fixture, prefix + "route_weights", 0, size_t(tokens) * 6);
            std::vector<float> outputs; std::vector<Route> routes;
            for (int t = 0; t < tokens; ++t) {
                ck(cudaMemcpyAsync(norm, normalized.data() + size_t(t) * hidden, hidden * 4, cudaMemcpyHostToDevice, stream.value));
                const auto route = moe.forward(norm, moe_output, stream.value);
                const auto value = download(moe_output, hidden, stream.value); outputs.insert(outputs.end(), value.begin(), value.end());
                bool correct = true;
                for (size_t k = 0; k < 6; ++k)
                    correct &= route.ids[k] == int32_t(ids[size_t(t) * 6 + k]) && std::abs(route.weights[k] - mixing[size_t(t) * 6 + k]) <= 2e-6f;
                check(correct, prefix + "native MoE baseline retains official routes token=" + std::to_string(t));
                routes.push_back(route);
            }
            expected.push_back(join(tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size), outputs,
                                    tensor<float>(fixture, prefix + "post", 0, size_t(tokens) * 4),
                                    tensor<float>(fixture, prefix + "comb", 0, size_t(tokens) * 16)));
            expected_routes.push_back(routes);
        }
    }
    std::vector<std::vector<float>> first_outputs, first_pre;
    for (int slots : {6, 2}) {
        CoveredSource source(g, index, populated, layer); CudaLayer2Decoder decoder(g, weights, read_compressor_weights(fixture, g, layer), read_index_key_weights(fixture, g, layer),
                                  read_index_query_weights(fixture, g, layer), source, slots, 3, 0, 3, layer);
        const size_t bytes = decoder.device_bytes();
        for (int pass = 0; pass < 2; ++pass) {
            if (pass) decoder.reset(stream.value);
            for (int c = 0; c < 3; ++c) {
                const int tokens = c == 0 ? 3 : 1, pos = c == 0 ? 0 : c + 2;
                const auto prefix = "case." + std::to_string(c) + ".";
                const auto label = "real decoder slots=" + std::to_string(slots) + " pass=" + std::to_string(pass) + " case=" + std::to_string(c);
                upload(x, tensor<float>(fixture, prefix + "input", 0, size_t(tokens) * residual_size), stream.value);
                upload(pre, tensor<float>(fixture, prefix + "incoming", 0, size_t(tokens) * 4), stream.value);
                std::vector<Route> routes(static_cast<size_t>(tokens));
                decoder.forward(x, pre, y, next, tokens, pos, stream.value, routes);
                const auto actual = download(y, size_t(tokens) * residual_size, stream.value);
                const auto coefficients = download(next, size_t(tokens) * 4, stream.value);
                compare(actual, expected[size_t(c)], label + " CPU graph + native MoE output");
                compare(coefficients, cpu_pre(boundaries[size_t(c)], weights.ffn_mhc), label + " same-input next-layer coefficients", true);
                compare_propagated(coefficients, tensor<float>(fixture, prefix + "next_pre", 0, size_t(tokens) * 4),
                                    boundaries[size_t(c)], tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size),
                                    weights.ffn_mhc, label + " end-to-end coefficient propagation bound");
                compare(cache(decoder.kv_cache(), stream.value), tensor<float>(fixture, prefix + "cache", 0, 128 * 512), label + " KV state");
                const int groups = (pos + tokens) / 2;
                compare(cache(decoder.compressed_kv_cache(), stream.value, size_t(groups) * 512),
                        tensor<float>(fixture, prefix + "compressed", 0, size_t(groups) * 512), label + " compressed KV state", false, true);
                compare(cache(decoder.index_key_cache(), stream.value, size_t(groups) * 128),
                        tensor<float>(fixture, prefix + "keys", 0, size_t(groups) * 128), label + " index-key state", false, true);
                check(decoder.compressed_groups() == groups, label + " compressed groups");
                check(decoder.shared_batch().source_layer()==layer,"completed publication retains original source identity");
                bool route_ids = true, route_weights = true;
                for (int t = 0; t < tokens; ++t)
                    for (size_t k = 0; k < 6; ++k) {
                        route_ids &= routes[size_t(t)].ids[k] == expected_routes[size_t(c)][size_t(t)].ids[k];
                        route_weights &= std::abs(routes[size_t(t)].weights[k] - expected_routes[size_t(c)][size_t(t)].weights[k]) <= 1e-4f;
                    }
                check(route_ids && route_weights, label + " original routes and weights");
                check(decoder.next_position() == pos + tokens && decoder.device_bytes() == bytes, label + " position and fixed memory");
                if (slots == 6 && pass == 0) { first_outputs.push_back(actual); first_pre.push_back(coefficients); }
                else {
                    check(same(actual, first_outputs[size_t(c)]), label + " output invariant to reset and eviction");
                    check(same(coefficients, first_pre[size_t(c)]), label + " coefficients invariant to reset and eviction");
                }
            }
        }
        const auto stats = decoder.stats();
        check(stats.hits + stats.misses == 60 && stats.expert_h2d_bytes == stats.misses * nvfp4_expert_bytes(g) &&
              stats.router_d2h_bytes == 10 * 384 * 4 && uint64_t(source.reads()) == stats.misses && stats.evictions > 0,
              "complete layer uses existing cache accounting slots=" + std::to_string(slots));
        if (slots == 2) check((layer==2 ? stats.misses==60 : stats.misses>=42) && stats.evictions==stats.misses-2, "two-slot cache forces repeated original expert loads without changing output");
        rejects("real decoder rejects cross-stream reset", [&] { decoder.reset(other.value); });
        std::printf("real_decoder_slots=%d device_bytes=%zu expert_h2d_bytes=%llu hits=%llu misses=%llu evictions=%llu\n", slots, bytes,
                    (unsigned long long)stats.expert_h2d_bytes, (unsigned long long)stats.hits,
                    (unsigned long long)stats.misses, (unsigned long long)stats.evictions);
    }
}
} // namespace
int main(int argc, char** argv) {
    int layer=2;
    if(argc>=3 && std::string(argv[1])=="--layer") {
        const std::string value=argv[2];if(value!="2" && value!="8")return 2;
        layer=value=="2"?2:8;argc-=2;argv+=2;
    }
    if (!(argc == 1 || (argc == 4 && std::string(argv[1]) == "--real"))) {
        std::fprintf(stderr, "usage: deepseek41_layer2_decoder_cuda_test [--layer 2|8] [--real DECODER.gguf ROUTED.gguf]\n"); return 2;
    }
    int count = 0; const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0)) return 77;
    try {
        ck(status); ck(cudaSetDevice(0));
        if (argc == 1) synthetic(layer); else real(argv[2], argv[3],layer);
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); ++failures; }
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
