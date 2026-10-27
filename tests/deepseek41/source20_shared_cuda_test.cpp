#include "decoder_test_utils.hpp"
#include "strata/deepseek41/source20_attention_cuda.hpp"
#include "strata/deepseek41/layer3_decoder_cuda.hpp"

namespace {
std::vector<int32_t> integers(const int32_t* pointer,size_t n,cudaStream_t stream){
    std::vector<int32_t> result(n);ck(cudaMemcpyAsync(result.data(),pointer,n*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));return result;
}
void run(int prefill){
    const int capacity=prefill+2;
    SyntheticWeights weights;const auto source_weights=weights.view(20).attention;
    const Source20CompressorWeights compressor{weights.compkv,weights.compnorm};
    const Source20IndexKeyWeights key{weights.key,weights.keynorm};
    const Source20IndexQueryWeights query{weights.index_query.view(),weights.head_weights};
    Stream stream,other;
    core::DeviceArena io(size_t(prefill)*hidden*12);
    auto* input=static_cast<float*>(io.alloc(size_t(prefill)*hidden*4));
    auto* output=static_cast<float*>(io.alloc(size_t(prefill)*hidden*4));
    auto* control=static_cast<float*>(io.alloc(size_t(prefill)*hidden*4));
    std::vector<float> x(size_t(prefill)*hidden);
    for(size_t i=0;i<x.size();++i)x[i]=bf(std::sin(float(i%997)*.037f)+.3f*std::cos(float(i/hidden)*.17f));
    upload(input,x,stream.value);
    CudaSource20Attention source(source_weights,compressor,key,query,capacity,prefill);
    std::array<std::unique_ptr<CudaSharedAttention>,3> consumers;
    for(int l=21;l<=23;++l)consumers[size_t(l-21)]=std::make_unique<CudaSharedAttention>(weights.view(l).attention,0,l);
    ZeroSource experts(geometry(),21);
    rejects("source20 consumer rejects ratio2 decoder geometry",[&]{CudaSharedDecoder bad(geometry(),weights.view(21),experts,2,0,21);});
    rejects("source20 cannot be constructed as a consumer",[&]{CudaSharedAttention bad(source_weights,0,20);});
    rejects("candidate index source24 remains unsupported",[&]{CudaSharedAttention bad(weights.view(24).attention,0,24);});
    rejects("implicit consumer still requires layer3",[&]{CudaSharedAttention bad(weights.view(21).attention);});
    rejects("source20 publication is unavailable before completion",[&]{(void)source.shared_batch();});
    source.forward(input,control,prefill,0,stream.value);
    const auto batch=source.shared_batch();const auto view=batch.view(0,prefill,stream.value);
    const auto expected=download(control,x.size(),stream.value);
    const auto saved=history(source,stream.value);
    const auto indices=integers(view.indices,size_t(prefill)*view.index_count,stream.value);
    const auto candidates=integers(view.candidates,size_t(prefill)*view.candidate_stride,stream.value);
    {
        CudaSourceAttention old(weights.view(2).attention,weights.compressor(),weights.index_key(),weights.query(),2,0,3);
        old.forward(input,control,3,0,stream.value);
        rejects("source20 consumer rejects ratio2 publication",[&]{consumers[0]->forward(input,output,3,0,old.shared_batch(),stream.value);});
        CudaSharedAttention legacy(weights.view(3).attention);
        rejects("ratio2 consumer rejects source20 publication",[&]{legacy.forward_source20(input,output,prefill,0,batch,stream.value);});
    }
    for(size_t i=0;i<consumers.size();++i){
        auto& consumer=*consumers[i];const auto label="consumer "+std::to_string(21+i);
        const auto before=cache(consumer.kv_cache(),stream.value);
        upload(output,std::vector<float>(x.size(),-19.f),stream.value);
        rejects(label+" rejects absent typed publication",[&]{consumer.forward_source20(input,output,prefill,0,Source20AttentionBatch{},stream.value);});
        rejects(label+" rejects mismatched prefill shape",[&]{consumer.forward_source20(input,output,prefill-1,0,batch,stream.value);});
        rejects(label+" rejects foreign stream",[&]{consumer.forward_source20(input,output,prefill,0,batch,other.value);});
        rejects(label+" rejects overlapping caller buffers",[&]{consumer.forward_source20(input,input,prefill,0,batch,stream.value);});
        const void* aliases[]={source.kv_cache(),source.compressed_kv_cache(),source.index_key_cache(),view.indices,view.candidates,consumer.kv_cache()};
        for(const void* alias:aliases){
            auto* p=reinterpret_cast<float*>(const_cast<void*>(alias));
            rejects(label+" rejects input alias with owner/consumer storage",[&]{consumer.forward_source20(p,output,prefill,0,batch,stream.value);});
            rejects(label+" rejects output alias with owner/consumer storage",[&]{consumer.forward_source20(input,p,prefill,0,batch,stream.value);});
        }
        check(consumer.next_position()==0 && before==cache(consumer.kv_cache(),stream.value) &&
              download(output,x.size(),stream.value)==std::vector<float>(x.size(),-19.f),label+" rejection preserves output and state");
        const auto bytes=consumer.device_bytes();
        consumer.forward_source20(input,output,prefill,0,batch,stream.value);
        check(same(download(output,x.size(),stream.value),expected),label+" borrowed per-query indices match identical-weight source attention exactly");
        check(consumer.next_position()==prefill && same(cache(consumer.kv_cache(),stream.value),cache(source.kv_cache(),stream.value)),label+" owns correct window history");
        check(history(source,stream.value)==saved && indices==integers(view.indices,size_t(prefill)*view.index_count,stream.value) &&
              candidates==integers(view.candidates,size_t(prefill)*view.candidate_stride,stream.value),label+" leaves owner KV/keys/indices/candidates unchanged");
        check(consumer.device_bytes()==bytes && bytes<source.device_bytes(),label+" fixed allocation excludes source compressor and caches");
    }
    // Same position and shape from a different owner must not rebind live consumers.
    {
        CudaSource20Attention replacement(source_weights,compressor,key,query,capacity,prefill);
        replacement.forward(input,control,prefill,0,stream.value);replacement.forward(input,control,1,prefill,stream.value);
        for(auto& consumer:consumers)rejects("matching position from a different source20 owner",[&]{consumer->forward_source20(input,output,1,prefill,replacement.shared_batch(),stream.value);});
    }
    source.forward(input,control,1,prefill,stream.value);
    const auto decode=source.shared_batch();const auto expected_decode=download(control,hidden,stream.value);
    for(auto& consumer:consumers){
        rejects("advanced source expires old publication",[&]{consumer->forward_source20(input,output,prefill,0,batch,stream.value);});
        consumer->forward_source20(input,output,1,prefill,decode,stream.value);
        check(same(download(output,hidden,stream.value),expected_decode) && consumer->next_position()==prefill+1,"single-token decode reuses source20 indices and window offset");
    }
    // Reach exactly the next consumer position on a new sequence from the same owner.
    source.reset(stream.value);source.forward(input,control,prefill,0,stream.value);
    source.forward(input,control,1,prefill,stream.value);source.forward(input,control,1,prefill+1,stream.value);
    for(auto& consumer:consumers)rejects("same owner/position after source reset requires consumer reset",[&]{consumer->forward_source20(input,output,1,prefill+1,source.shared_batch(),stream.value);});
    source.reset(stream.value);source.forward(input,control,prefill,0,stream.value);
    for(auto& consumer:consumers){
        consumer->reset(stream.value);consumer->forward_source20(input,output,prefill,0,source.shared_batch(),stream.value);
        check(same(download(output,x.size(),stream.value),expected),"reset/replay retains exact source20 consumer output");
    }
    Source20AttentionBatch expired;
    {
        CudaSource20Attention temporary(source_weights,compressor,key,query,3,3);
        temporary.forward(input,control,3,0,stream.value);expired=temporary.shared_batch();
    }
    consumers[0]->reset(stream.value);
    rejects("destroyed owner invalidates typed publication",[&]{consumers[0]->forward_source20(input,output,3,0,expired,stream.value);});
    check(consumers[0]->next_position()==0,"destroyed-owner rejection preserves reset consumer");
    std::printf("source20_device_bytes=%zu shared_consumer_device_bytes=%zu\n",source.device_bytes(),consumers[0]->device_bytes());
}
}
int main(int argc,char** argv){
    if(!(argc==1 || (argc==2 && std::string(argv[1])=="--short")))return 2;
    int n=0;const auto status=cudaGetDeviceCount(&n);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess && !n))return 77;
    try{ck(status);ck(cudaSetDevice(0));run(argc==1?129:9);}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
