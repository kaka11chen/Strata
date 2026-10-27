#include "strata/deepseek41/layer1_decoder_cuda.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_runtime.h>

namespace strata::deepseek41 {
namespace {
constexpr int hidden = 5120, max_position = 1048576;
constexpr size_t lookup_size = 6144, residual_size = 4*hidden;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw core::CudaError(std::string(what)+": "+cudaGetErrorString(e),int(e));
}
int checked(const Geometry& g, const DecoderWeights& w, const Layer1EngramWeights& e, int device) {
    if (g.layers != 40 || g.hidden != hidden || g.expert_width != 2304 || g.experts != 384 || g.top_k != 6 ||
        g.compress_ratios.size() != 40 || g.compress_ratios[1] != 0 || w.attention.layer != 1 || e.layer != 1)
        throw std::invalid_argument("Layer-1 decoder needs fixed 1017 geometry and layer-1 pure-SWA and Engram weights");
    return device;
}
bool valid(const void* p, size_t n, size_t alignment) { return p && uintptr_t(p)%alignment == 0 && n <= UINTPTR_MAX-uintptr_t(p); }
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    return an && bn && (uintptr_t(a) <= uintptr_t(b) ? uintptr_t(b)-uintptr_t(a)<an : uintptr_t(a)-uintptr_t(b)<bn);
}
} // namespace
CudaLayer1Decoder::CudaLayer1Decoder(const Geometry& g, const DecoderWeights& w, const Layer1EngramWeights& e,
    std::span<const uint32_t> map, const std::string& path, core::ExpertSource& source, int slots,
    const EngramIoOptions& io, int device)
    : device_(checked(g,w,e,device)), hash_(map), table_(path,io), engram_(e,device),
      attention_mhc_(g,w.attention_mhc,device), attention_(w.attention,device,1),
      ffn_mhc_(g,w.ffn_mhc,device), moe_(g,1,w.router,w.bias,source,slots,w.shared,device),
      arena_((lookup_size+2*residual_size+hidden)*4,device) {
    lookup_ = static_cast<float*>(arena_.alloc(lookup_size*4));
    engram_residual_ = static_cast<float*>(arena_.alloc(residual_size*4));
    attention_output_ = static_cast<float*>(arena_.alloc(hidden*4));
    middle_ = static_cast<float*>(arena_.alloc(residual_size*4));
}
void CudaLayer1Decoder::check_stream(void* stream) const {
    if (failed_) throw std::logic_error("Layer-1 decoder must be reconstructed after execution failure");
    if (!stream || (stream_ && stream != stream_)) throw std::invalid_argument("Layer-1 decoder requires one explicit stream");
    int device=-1; ck(cudaGetDevice(&device),"get layer-1 decoder device");
    if (device != device_) throw std::invalid_argument("Layer-1 decoder called on wrong device");
}
void CudaLayer1Decoder::forward(std::span<const int32_t> tokens, const float* input, const float* incoming,
    float* output, float* next_pre, int start, void* stream, std::span<Route> routes) {
    check_stream(stream);
    const size_t count=tokens.size();
    if (!count || count>max_position || start<0 || start!=next_ || count>size_t(max_position-start) ||
        (start && count!=1) || (!routes.empty() && routes.size()!=count))
        throw std::invalid_argument("Layer-1 decoder needs initial prefill then contiguous one-token decode and matching routes");
    const void* pointers[]={input,incoming,output,next_pre,tokens.data(),routes.data()};
    const size_t bytes[]={count*residual_size*4,count*16,count*residual_size*4,count*16,tokens.size_bytes(),routes.size_bytes()};
    const size_t alignments[]={4,4,4,4,alignof(int32_t),alignof(Route)};
    for (size_t i=0;i<6;++i) {
        if (!bytes[i]) continue;
        if (!valid(pointers[i],bytes[i],alignments[i]) || overlap(pointers[i],bytes[i],arena_.base(),arena_.capacity()) ||
            overlap(pointers[i],bytes[i],attention_.kv_cache(),128*512*2))
            throw std::invalid_argument("Layer-1 decoder needs aligned external non-cache buffers");
        for (size_t j=0;j<i;++j)
            if (overlap(pointers[i],bytes[i],pointers[j],bytes[j])) throw std::invalid_argument("Layer-1 decoder caller buffers overlap");
    }
    for (int32_t id:tokens)
        if (id<0 || size_t(id)>=engram_token_count) throw std::invalid_argument("Layer-1 original token ID out of range");
    const auto s=static_cast<cudaStream_t>(stream);
    try {
        stream_=stream;
        const int prefill_width=start==0?int(std::min(count,size_t(128))):0;
        for (size_t t=0;t<count;++t) {
            hash_.forward(tokens.subspan(t,1),rows_,start+int(t));
            table_.gather(rows_,lookup_bits_);
            for (size_t i=0;i<lookup_size;++i) lookup_values_[i]=kernels::f32_from_bf16(lookup_bits_[i]);
            ck(cudaMemcpyAsync(lookup_,lookup_values_.data(),lookup_size*4,cudaMemcpyHostToDevice,s),"upload Engram lookup");
            engram_.forward(lookup_,input+t*residual_size,engram_residual_,stream);
            const auto state=attention_mhc_.prepare(engram_residual_,incoming+t*4,stream);
            attention_.forward_token(state.normalized,attention_output_,start+int(t),prefill_width,stream);
            attention_mhc_.finish(attention_output_,middle_,stream);
            const Route route=ffn_mhc_.forward_moe(moe_,middle_,state.next_pre,output+t*residual_size,next_pre+t*4,stream);
            if (!routes.empty()) routes[t]=route;
        }
        ck(cudaStreamSynchronize(s),"wait for complete layer-1 decoder");
        next_+=int(count);
    } catch (...) { failed_=true; (void)cudaStreamSynchronize(s); throw; }
}
void CudaLayer1Decoder::reset(void* stream) {
    check_stream(stream);
    try { attention_.reset(stream); hash_.reset(); next_=0; stream_=stream; }
    catch (...) { failed_=true; (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); throw; }
}
size_t CudaLayer1Decoder::device_bytes() const {
    return engram_.device_bytes()+attention_mhc_.device_bytes()+attention_.device_bytes()+ffn_mhc_.device_bytes()+moe_.device_bytes()+arena_.capacity();
}
} // namespace strata::deepseek41
