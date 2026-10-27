#include "decoder_test_utils.hpp"
#include "strata/deepseek41/source20_decoder_cuda.hpp"

namespace {
Geometry source20_geometry(){auto g=geometry();g.compress_ratios[20]=1;g.sources[20]={20,20,-1};return g;}
struct Source20Weights : SyntheticWeights {
    Source20CompressorWeights compressor(int=20) const{return {compkv,compnorm};}
    Source20IndexKeyWeights index_key(int=20) const{return {key,keynorm};}
    Source20IndexQueryWeights query(int=20) const{return {index_query.view(),head_weights};}
};
struct ObservedSource : ZeroSource {
    std::function<void()> observe;
    explicit ObservedSource(const Geometry& g):ZeroSource(g,20){}
    const uint8_t* blob(int64_t layer,int64_t expert) override{if(observe)observe();return ZeroSource::blob(layer,expert);}
};
std::vector<int32_t> integers(const int32_t* p,size_t n,cudaStream_t stream){
    std::vector<int32_t> v(n);ck(cudaMemcpyAsync(v.data(),p,n*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));return v;
}
void check_publication(const CudaSource20Decoder& decoder,const GgufModel& fixture,const std::string& prefix,
                       int tokens,int start,cudaStream_t stream){
    const auto view=decoder.shared_batch().view(start,tokens,stream);
    const auto ids=integers(view.indices,size_t(tokens)*view.index_count,stream);
    const auto expected=tensor<float>(fixture,prefix+"indices",0,ids.size());
    check(std::equal(ids.begin(),ids.end(),expected.begin()),prefix+"decoder publishes exact reference top-512 rows");
    const auto blocks=integers(view.candidates,size_t(tokens)*view.candidate_stride,stream);
    const auto mask=tensor<float>(fixture,prefix+"candidate_mask",0,size_t(tokens)*view.key_count);bool same_mask=true;
    for(int t=0;t<tokens;++t)for(int k=0;k<view.key_count;++k){
        const auto begin=blocks.begin()+size_t(t)*view.candidate_stride,end=begin+view.candidate_stride;
        same_mask&=(std::find(begin,end,k/8)!=end)==(mask[size_t(t)*view.key_count+k]!=0);
    }
    check(same_mask && view.kv==decoder.compressed_kv_cache() && view.keys==decoder.index_key_cache(),prefix+"decoder publishes exact reference candidates and owned caches");
}
void synthetic() {
    constexpr int layer=20;
    const auto g = source20_geometry(); Source20Weights storage; const auto w = storage.view(layer); ZeroSource source(g,layer);
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
        CudaSource20Attention attention(w.attention, storage.compressor(), storage.index_key(), storage.query(), count, first);
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
    CudaSource20Decoder decoder(g, w, storage.compressor(), storage.index_key(), storage.query(), source, 6, count, first);
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
    rejects("null incoming coefficients",[&]{decoder.forward(x,nullptr,y,next,1,0,stream.value);});
    rejects("null next-layer coefficients",[&]{decoder.forward(x,pre,y,nullptr,1,0,stream.value);});
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
    const auto full_output=download(y,input.size(),stream.value);const auto publication=decoder.shared_batch();
    rejects("full ratio-1 cache rejects another token",[&]{decoder.forward(x,pre,y,next,1,count,stream.value);});
    check(history(decoder,stream.value)==before && same(full_output,download(y,input.size(),stream.value)) &&
          publication.view(first,1,stream.value).key_count==count,"capacity rejection preserves output/history/publication");
    decoder.reset(stream.value);
    check(decoder.next_position() == 0 && history(decoder, stream.value) == initial, "reset clears sequence state");
    decoder.forward(x, pre, y, next, first, 0, stream.value);
    decoder.forward(x + first * residual_size, pre + first * 4, y + first * residual_size, next + first * 4, 1, first, stream.value);
    check(same(output, download(y, input.size(), stream.value)), "reset/replay is byte-identical with warm expert cache");
    check(source.calls == 6 && decoder.stats().hits == 12 * count - 6 && decoder.device_bytes() == bytes, "reset retains immutable expert cache and fixed allocation");
    decoder.reset(stream.value);
    auto changed = incoming; std::reverse(changed.begin(), changed.begin() + 4); upload(pre, changed, stream.value);
    decoder.forward(x, pre, y, next, 1, 0, stream.value);
    check(!same(download(y, residual_size, stream.value), std::vector<float>(output.begin(), output.begin() + residual_size)), "incoming mHC coefficients affect attention/output");
    decoder.reset(stream.value);
    check(history(decoder, stream.value) == initial, "reset discards ratio-1 sequence and clears every cache");
    rejects("reset invalidates completed publication", [&] { (void)decoder.shared_batch(); });
    rejects("wrong shared-source topology rejected", [&] { auto bad = g; bad.sources[size_t(layer)].kv = 0; CudaSource20Decoder invalid(bad, w, storage.compressor(), storage.index_key(), storage.query(), source, 2, 5, 3); });
    rejects("wrong decoder geometry rejected", [&] { auto bad = g; bad.hidden = 64; CudaSource20Decoder invalid(bad, w, storage.compressor(), storage.index_key(), storage.query(), source, 2, 5, 3); });
    rejects("wrong ratio rejected",[&]{auto bad=g;bad.compress_ratios[20]=2;CudaSource20Decoder d(bad,w,storage.compressor(),storage.index_key(),storage.query(),source,2,5,3);});
    rejects("wrong index owner rejected",[&]{auto bad=g;bad.sources[20].index=14;CudaSource20Decoder d(bad,w,storage.compressor(),storage.index_key(),storage.query(),source,2,5,3);});
    rejects("source20 does not consume candidates",[&]{auto bad=g;bad.sources[20].candidates=20;CudaSource20Decoder d(bad,w,storage.compressor(),storage.index_key(),storage.query(),source,2,5,3);});
    rejects("other attention owner rejected",[&]{auto bad=w;bad.attention.layer=14;CudaSource20Decoder d(g,bad,storage.compressor(),storage.index_key(),storage.query(),source,2,5,3);});
    {
        ObservedSource missing(g);CudaSource20Decoder broken(g,w,storage.compressor(),storage.index_key(),storage.query(),missing,2,5,3);
        int observations=0;bool premature=false;
        missing.observe=[&]{++observations;try{(void)broken.shared_batch();premature=true;}catch(const std::invalid_argument&) {}};
        broken.forward(x,pre,y,next,3,0,stream.value);const auto published=broken.shared_batch();
        check(observations>0 && !premature && published.view(0,3,stream.value).key_count==3,
              "publication remains inaccessible inside expert reads until entire decoder commits");
        const auto state=history(broken,stream.value);const auto result=download(y,3*residual_size,stream.value);
        const auto view=published.view(0,3,stream.value);
        for(const int32_t* p:{view.indices,view.candidates}){
            auto* alias=reinterpret_cast<float*>(const_cast<int32_t*>(p));
            rejects("residual/publication alias",[&]{broken.forward(alias,pre,y,next,1,3,stream.value);});
            rejects("incoming/publication alias",[&]{broken.forward(x,alias,y,next,1,3,stream.value);});
            rejects("output/publication alias",[&]{broken.forward(x,pre,alias,next,1,3,stream.value);});
            rejects("next_pre/publication alias",[&]{broken.forward(x,pre,y,alias,1,3,stream.value);});
        }
        rejects("rejected call retains committed decoder publication",[&]{broken.forward(x,pre,y,next,1,4,stream.value);});
        check(published.view(0,3,stream.value).tokens==3 && history(broken,stream.value)==state && same(result,download(y,3*residual_size,stream.value)),"rejection preserves decoder transaction");
        missing.missing=true;
        rejects("missing original expert fails after ratio-1 producer advance",[&]{broken.forward(x,pre,y,next,1,3,stream.value);},true);
        check(broken.next_position()==3 && broken.size()==4,"failed call does not commit completed position");
        rejects("failure prevents fresh publication",[&]{(void)broken.shared_batch();});
        rejects("failure expires old publication",[&]{published.view(0,3,stream.value);});
        rejects("failed decoder cannot continue",[&]{broken.forward(x,pre,y,next,1,3,stream.value);},true);
        rejects("reset cannot recover partial execution",[&]{broken.reset(stream.value);},true);
    }
    std::printf("synthetic_decoder_device_bytes=%zu\n", bytes);
}
struct AttentionBoundary { std::vector<float> residual, normalized; };
struct CpuRouter {
    std::array<double,384> logits{}, scores{};
    Route route;
};
// Independent FP64 gate oracle; never calls the runtime route_top6 implementation.
CpuRouter cpu_router(std::span<const float> input, std::span<const uint16_t> weight,
                     std::span<const float> bias) {
    CpuRouter out;
    std::array<int32_t,384> order{};std::iota(order.begin(),order.end(),0);
    for(size_t row=0;row<384;++row){
        double z=0.;
        for(size_t d=0;d<hidden;++d)z+=double(kernels::f32_from_bf16(weight[row*hidden+d]))*input[d];
        out.logits[row]=z;out.scores[row]=std::sqrt(z>20.?z:std::log1p(std::exp(z)));
    }
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return out.scores[size_t(a)]+bias[size_t(a)]>out.scores[size_t(b)]+bias[size_t(b)];});
    double sum=1e-20;for(size_t k=0;k<6;++k)sum+=out.scores[size_t(order[k])];
    for(size_t k=0;k<6;++k){out.route.ids[k]=order[k];out.route.weights[k]=float(1.5*out.scores[size_t(order[k])]/sum);}
    return out;
}
// For s(z)=sqrt(softplus(z)), |s'(z)| <= 1/2. With the same selected IDs,
// E_i=|delta logit_i|/2 bounds each score change. For sums S (reference), A
// (native input), |delta weight_i| <= 1.5*(E_i/A + s_i*sum(E)/(S*A)).
// Add the original 2e-6 same-input numerical allowance for each endpoint.
void compare_route_propagation(std::span<const Route> actual, std::span<const float> expected,
                               const std::vector<CpuRouter>& native, const std::vector<CpuRouter>& reference,
                               const std::string& name) {
    bool valid=true;double maximum=0.,max_bound=0.,max_fraction=0.;
    for(size_t t=0;t<actual.size();++t){
        const auto& a=native[t];const auto& r=reference[t];
        double sum_a=1e-20,sum_r=1e-20,sum_e=0.;std::array<double,6> errors{};
        for(size_t k=0;k<6;++k){
            valid&=a.route.ids[k]==r.route.ids[k] && actual[t].ids[k]==r.route.ids[k];
            const size_t id=size_t(r.route.ids[k]);
            errors[k]=.5*std::abs(a.logits[id]-r.logits[id]);sum_e+=errors[k];
            sum_a+=a.scores[id];sum_r+=r.scores[id];
        }
        for(size_t k=0;k<6;++k){
            const size_t id=size_t(r.route.ids[k]);
            const double bound=1.5*(errors[k]/sum_a+r.scores[id]*sum_e/(sum_r*sum_a))+4e-6;
            const double delta=std::abs(double(actual[t].weights[k])-expected[t*6+k]);
            valid&=std::isfinite(delta) && std::isfinite(bound) && delta<=bound;
            maximum=std::max(maximum,delta);max_bound=std::max(max_bound,bound);max_fraction=std::max(max_fraction,delta/bound);
        }
    }
    std::printf("  %s max_abs=%.9g max_bound=%.9g max_fraction=%.9g\n",name.c_str(),maximum,max_bound,max_fraction);
    check(valid,name);
}
std::vector<AttentionBoundary> native_attention_boundaries(const Geometry& g, const DecoderWeights& w,
                                                           const GgufModel& fixture, cudaStream_t stream) {
    CudaMhc mhc(g, w.attention_mhc), ffn(g,w.ffn_mhc);
    CudaSource20Attention attention(w.attention, read_source20_compressor_weights(fixture, g),
                                 read_source20_index_key_weights(fixture, g), read_source20_index_query_weights(fixture, g), 5, 3);
    core::DeviceArena io(2 * 1024 * 1024);
    auto* x = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* incoming = static_cast<float*>(io.alloc(3 * 16));
    auto* norm = static_cast<float*>(io.alloc(3 * hidden * 4));
    auto* ay = static_cast<float*>(io.alloc(3 * hidden * 4));
    auto* middle = static_cast<float*>(io.alloc(3 * residual_size * 4));
    auto* dummy = static_cast<float*>(io.alloc(residual_size * 4));
    auto* zero = static_cast<float*>(io.alloc(hidden * 4)); ck(cudaMemsetAsync(zero, 0, hidden * 4, stream));
    std::vector<AttentionBoundary> boundaries;
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
        std::vector<float> normalized;
        for (int t = 0; t < tokens; ++t) {
            const auto prepared=mhc.prepare(x + size_t(t) * residual_size, incoming + size_t(t) * 4, stream);
            mhc.finish(ay + size_t(t) * hidden, middle + size_t(t) * residual_size, stream);
            const auto stage=ffn.prepare(middle + size_t(t)*residual_size,prepared.next_pre,stream);
            const auto input=download(stage.normalized,hidden,stream);
            normalized.insert(normalized.end(),input.begin(),input.end());
            ffn.finish(zero,dummy,stream); // close the pair without running MoE
        }
        boundaries.push_back({download(middle, size_t(tokens) * residual_size, stream),std::move(normalized)});
        const auto reference = tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size);
        compare(boundaries.back().residual, reference, prefix + "native attention residual boundary");
        compare(boundaries.back().normalized,tensor<float>(fixture,prefix+"ffn_input",0,size_t(tokens)*hidden),
                prefix+"independent native FFN input boundary");
        compare(cpu_pre(reference, w.ffn_mhc), tensor<float>(fixture, prefix + "next_pre", 0, size_t(tokens) * 4),
                prefix + "independent CPU coefficients match pinned oracle on same input", true);
    }
    return boundaries;
}

void real(const char* fixture_path, const char* routed_path, int layer) {
    const auto g = source20_geometry();
    const auto fixture = GgufModel::open(fixture_path), routed = GgufModel::open(routed_path);
    const auto* oracle=fixture.meta().get("fixture.linear_oracle");
    if(!oracle || oracle->type!=MetaType::STRING || oracle->s!="fp64-linears-fp32-boundaries-v1")throw std::runtime_error("Source20 requires the declared FP64 attention reference");
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
    std::vector<std::vector<CpuRouter>> reference_routers, native_routers;
    {
        CoveredSource source(g, index, populated, layer); CudaMoE moe(g, layer, router, bias, source, 6, weights.shared);
        for (int c = 0; c < 3; ++c) {
            const int tokens = c == 0 ? 3 : 1;
            const auto prefix = "case." + std::to_string(c) + ".";
            const auto normalized = tensor<float>(fixture, prefix + "ffn_input", 0, size_t(tokens) * hidden);
            const auto ids = tensor<float>(fixture, prefix + "route_ids", 0, size_t(tokens) * 6);
            const auto mixing = tensor<float>(fixture, prefix + "route_weights", 0, size_t(tokens) * 6);
            std::vector<float> outputs; std::vector<CpuRouter> reference_rows,native_rows;
            for (int t = 0; t < tokens; ++t) {
                ck(cudaMemcpyAsync(norm, normalized.data() + size_t(t) * hidden, hidden * 4, cudaMemcpyHostToDevice, stream.value));
                const auto route = moe.forward(norm, moe_output, stream.value);
                const auto value = download(moe_output, hidden, stream.value); outputs.insert(outputs.end(), value.begin(), value.end());
                bool correct = true;
                for (size_t k = 0; k < 6; ++k)
                    correct &= route.ids[k] == int32_t(ids[size_t(t) * 6 + k]) && std::abs(route.weights[k] - mixing[size_t(t) * 6 + k]) <= 2e-6f;
                check(correct, prefix + "native MoE baseline retains official routes token=" + std::to_string(t));
                reference_rows.push_back(cpu_router(std::span(normalized).subspan(size_t(t)*hidden,hidden),router,bias));
                native_rows.push_back(cpu_router(std::span(boundaries[size_t(c)].normalized).subspan(size_t(t)*hidden,hidden),router,bias));
                bool oracle=true;
                for(size_t k=0;k<6;++k)oracle&=reference_rows.back().route.ids[k]==int32_t(ids[size_t(t)*6+k]) &&
                    std::abs(reference_rows.back().route.weights[k]-mixing[size_t(t)*6+k])<=2e-6f;
                check(oracle,prefix+"independent CPU gate matches pinned oracle token="+std::to_string(t));
            }
            expected.push_back(join(tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size), outputs,
                                    tensor<float>(fixture, prefix + "post", 0, size_t(tokens) * 4),
                                    tensor<float>(fixture, prefix + "comb", 0, size_t(tokens) * 16)));
            reference_routers.push_back(std::move(reference_rows));native_routers.push_back(std::move(native_rows));
        }
    }
    std::vector<std::vector<float>> first_outputs, first_pre;
    for (int slots : {6, 2}) {
        CoveredSource source(g, index, populated, layer); CudaSource20Decoder decoder(g, weights, read_source20_compressor_weights(fixture, g), read_source20_index_key_weights(fixture, g),
                                  read_source20_index_query_weights(fixture, g), source, slots, 5, 3);
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
                compare(coefficients, cpu_pre(boundaries[size_t(c)].residual, weights.ffn_mhc), label + " same-input next-layer coefficients", true);
                compare_propagated(coefficients, tensor<float>(fixture, prefix + "next_pre", 0, size_t(tokens) * 4),
                                    boundaries[size_t(c)].residual, tensor<float>(fixture, prefix + "after_attention", 0, size_t(tokens) * residual_size),
                                    weights.ffn_mhc, label + " end-to-end coefficient propagation bound");
                compare(cache(decoder.kv_cache(), stream.value), tensor<float>(fixture, prefix + "cache", 0, 128 * 512), label + " KV state");
                const int groups = pos + tokens;
                compare(cache(decoder.compressed_kv_cache(), stream.value, size_t(groups) * 512),
                        tensor<float>(fixture, prefix + "compressed", 0, size_t(groups) * 512), label + " compressed KV state", false, true);
                compare(cache(decoder.index_key_cache(), stream.value, size_t(groups) * 128),
                        tensor<float>(fixture, prefix + "keys", 0, size_t(groups) * 128), label + " index-key state", false, true);
                check(decoder.size() == groups, label + " compressed groups");
                check_publication(decoder,fixture,prefix,tokens,pos,stream.value);
                bool route_ids=true,route_weights=true;
                for(int t=0;t<tokens;++t)for(size_t k=0;k<6;++k){
                    const auto& expected_route=native_routers[size_t(c)][size_t(t)].route;
                    route_ids&=routes[size_t(t)].ids[k]==reference_routers[size_t(c)][size_t(t)].route.ids[k] &&
                               routes[size_t(t)].ids[k]==expected_route.ids[k];
                    route_weights&=std::isfinite(routes[size_t(t)].weights[k]) &&
                        std::abs(routes[size_t(t)].weights[k]-expected_route.weights[k])<=2e-6f;
                }
                check(route_ids,label+" original expert IDs on both reference and native inputs");
                check(route_weights,label+" same-input independent CPU gate weights");
                compare_route_propagation(routes,tensor<float>(fixture,prefix+"route_weights",0,size_t(tokens)*6),
                    native_routers[size_t(c)],reference_routers[size_t(c)],label+" end-to-end route-weight propagation bound");
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
        if (slots == 2) check(stats.misses>=42 && stats.evictions==stats.misses-2, "two-slot cache forces repeated original expert loads without changing output");
        rejects("real decoder rejects cross-stream reset", [&] { decoder.reset(other.value); });
        std::printf("real_decoder_slots=%d device_bytes=%zu expert_h2d_bytes=%llu hits=%llu misses=%llu evictions=%llu\n", slots, bytes,
                    (unsigned long long)stats.expert_h2d_bytes, (unsigned long long)stats.hits,
                    (unsigned long long)stats.misses, (unsigned long long)stats.evictions);
    }
}
} // namespace
int main(int argc,char** argv){
    if(!(argc==1 || (argc==4 && std::string(argv[1])=="--real")))return 2;
    int count=0;const auto status=cudaGetDeviceCount(&count);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && count==0))return 77;
    try{ck(status);ck(cudaSetDevice(0));if(argc==1)synthetic();else real(argv[2],argv[3],20);}
    catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
