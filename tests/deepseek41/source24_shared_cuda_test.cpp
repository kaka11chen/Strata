#include "decoder_test_utils.hpp"
#include "strata/deepseek41/source20_attention_cuda.hpp"
#include "strata/deepseek41/layer3_decoder_cuda.hpp"
namespace {
int index_layer=24;
std::vector<int32_t> ids(const int32_t* p,size_t n,cudaStream_t s){std::vector<int32_t> out(n);ck(cudaMemcpyAsync(out.data(),p,n*4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));return out;}
struct Weights {
    SyntheticWeights base;
    Source20CompressorWeights compressor{base.compkv,base.compnorm};
    Source20IndexKeyWeights key{base.key,base.keynorm};
    Source20IndexQueryWeights query20{base.index_query.view(),base.head_weights};
    CandidateIndexQueryWeights query{base.index_query.view(),base.head_weights,index_layer};
};
std::vector<float> input_values(int tokens){std::vector<float> x(size_t(tokens)*hidden);for(size_t i=0;i<x.size();++i)x[i]=bf(std::sin(float(i%997)*.037f)+.3f*std::cos(float(i/hidden)*.17f));return x;}
void lifecycle(){
    constexpr int n=9,cap=11;Weights w;Stream stream,other;core::DeviceArena io(size_t(n)*hidden*12);
    auto* x=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* y=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* control=static_cast<float*>(io.alloc(size_t(n)*hidden*4));
    const auto input=input_values(n);upload(x,input,stream.value);
    CudaSource20Attention owner(w.base.view(20).attention,w.compressor,w.key,w.query20,cap,n);
    CudaSharedAttention index(w.base.view(index_layer).attention,w.query,cap,n);
    std::array<std::unique_ptr<CudaSharedAttention>,3> consumers;
    for(int l=(index_layer+1);l<=(index_layer+3);++l)consumers[size_t(l-(index_layer+1))]=std::make_unique<CudaSharedAttention>(w.base.view(l).attention,0,l);
    ZeroSource experts(geometry(),index_layer);
    rejects("candidate source decoder requires candidate topology",[&]{CudaSharedDecoder bad(geometry(),w.base.view(index_layer),w.query,experts,2,cap,n);});
    rejects("candidate index owner requires explicit index query weights",[&]{CudaSharedAttention bad(w.base.view(index_layer).attention,0,index_layer);});
    rejects("MTP layer40 cannot be a backbone consumer",[&]{CudaSharedAttention bad(w.base.view(40).attention,0,40);});
    rejects("candidate constructor rejects other layer identity",[&]{CudaSharedAttention bad(w.base.view((index_layer+1)).attention,w.query,cap,n);});
    rejects("candidate query identity must match its index owner",[&]{auto q=w.query;q.layer=(index_layer==24?28:24);CudaSharedAttention bad(w.base.view(index_layer).attention,q,cap,n);});
    rejects("candidate source has no publication before completion",[&]{(void)index.shared_batch();});
    rejects("ordinary consumer cannot publish indices",[&]{(void)consumers[0]->shared_batch();});
    owner.forward(x,control,n,0,stream.value);const auto original=owner.shared_batch();const auto view=original.view(0,n,stream.value);
    const auto expected=download(control,input.size(),stream.value);const auto owner_state=history(owner,stream.value);
    {
        CudaSharedAttention small(w.base.view(index_layer).attention,w.query,n-1,n-1);
        rejects("candidate source rejects oversized source before advancing",[&]{small.forward_source20(x,y,n,0,original,stream.value);});
        check(small.next_position()==0,"candidate source capacity failure is transactional");
    }
    rejects("candidate source needs source20 publication",[&]{index.forward_source20(x,y,n,0,{},stream.value);});
    rejects("candidate source rejects wrong stream",[&]{index.forward_source20(x,y,n,0,original,other.value);});
    index.forward_source20(x,y,n,0,original,stream.value);const auto publication=index.shared_batch();
    check(publication.source_layer()==index_layer && index.next_position()==n,"candidate source owns completed index publication");
    check(same(download(y,input.size(),stream.value),expected),"all-visible candidate source attention matches existing source20 math");
    const auto bytes=index.device_bytes();
    for(size_t i=0;i<consumers.size();++i){auto& c=*consumers[i];
        rejects("consumer cannot reuse source20 top512",[&]{c.forward_source20(x,y,n,0,original,stream.value);});
        rejects("consumer needs exact publication shape",[&]{c.forward(x,y,n-1,0,publication,stream.value);});
        rejects("consumer rejects foreign stream",[&]{c.forward(x,y,n,0,publication,other.value);});
        const void* aliases[]={owner.kv_cache(),owner.compressed_kv_cache(),owner.index_key_cache(),view.indices,view.candidates,index.kv_cache(),c.kv_cache()};
        for(const void* alias:aliases){auto* p=reinterpret_cast<float*>(const_cast<void*>(alias));
            rejects("consumer protects both source owners from alias",[&]{c.forward(x,p,n,0,publication,stream.value);});}
        check(c.next_position()==0,"bad consumer calls preserve position");
        c.forward(x,y,n,0,publication,stream.value);
        check(same(download(y,input.size(),stream.value),expected) && same(cache(c.kv_cache(),stream.value),cache(index.kv_cache(),stream.value)),"consumer owns window ring and reuses candidate source indices");
        check(history(owner,stream.value)==owner_state && publication.source_layer()==index_layer,"consumers preserve upstream source20 and candidate source");
    }
    {
        CudaSource20Attention replacement(w.base.view(20).attention,w.compressor,w.key,w.query20,cap,n);
        replacement.forward(x,control,n,0,stream.value);replacement.forward(x,control,1,n,stream.value);
        rejects("candidate source cannot rebind another source20 at same position",[&]{index.forward_source20(x,y,1,n,replacement.shared_batch(),stream.value);});
        check(publication.source_layer()==index_layer,"rejected rebind retains candidate source publication");
    }
    {
        CudaSharedAttention replacement(w.base.view(index_layer).attention,w.query,cap,n);
        replacement.forward_source20(x,y,n,0,original,stream.value);
        owner.forward(x,control,1,n,stream.value);const auto next=owner.shared_batch();
        rejects("advancing KV owner expires dependent candidate source publication",[&]{(void)publication.source_layer();});
        rejects("candidate source getter cannot republish stale upstream KV",[&]{(void)index.shared_batch();});
        CudaSharedAttention pending(w.base.view((index_layer+1)).attention,0,(index_layer+1));
        rejects("pending consumer rejects stale parent at matching position",[&]{pending.forward(x,y,n,0,publication,stream.value);});
        check(pending.next_position()==0,"stale-parent rejection leaves pending consumer unchanged");
        replacement.forward_source20(x,y,1,n,next,stream.value);
        rejects("consumer cannot rebind a different index owner",[&]{consumers[0]->forward(x,y,1,n,replacement.shared_batch(),stream.value);});
        index.forward_source20(x,control,1,n,next,stream.value);const auto expected_decode=download(control,hidden,stream.value);
        for(auto& c:consumers){c->forward(x,y,1,n,index.shared_batch(),stream.value);check(same(download(y,hidden,stream.value),expected_decode),"candidate source decode publication retains original window offset");}
    }
    const auto before_reset=index.shared_batch();index.reset(stream.value);
    rejects("index owner reset invalidates its own publication",[&]{(void)before_reset.source_layer();});
    check(owner.shared_batch().view(n,1,stream.value).key_count==n+1,"index reset never resets KV owner");
    owner.reset(stream.value);owner.forward(x,control,n,0,stream.value);index.forward_source20(x,y,n,0,owner.shared_batch(),stream.value);
    for(int pos:{n,n+1}){owner.forward(x,control,1,pos,stream.value);index.forward_source20(x,y,1,pos,owner.shared_batch(),stream.value);}
    for(auto& c:consumers)rejects("matching position after index reset needs consumer reset",[&]{c->forward(x,y,1,n+1,index.shared_batch(),stream.value);});
    for(auto& c:consumers)c->reset(stream.value);
    index.reset(stream.value);owner.reset(stream.value);owner.forward(x,control,n,0,stream.value);index.forward_source20(x,y,n,0,owner.shared_batch(),stream.value);
    for(auto& c:consumers){c->forward(x,y,n,0,index.shared_batch(),stream.value);check(same(download(y,input.size(),stream.value),expected),"whole dependency chain resets and replays exactly");}
    {
        auto parent=std::make_unique<CudaSource20Attention>(w.base.view(20).attention,w.compressor,w.key,w.query20,3,3);
        CudaSharedAttention child(w.base.view(index_layer).attention,w.query,3,3),pending(w.base.view((index_layer+1)).attention,0,(index_layer+1));
        parent->forward(x,control,3,0,stream.value);child.forward_source20(x,y,3,0,parent->shared_batch(),stream.value);
        const auto stale=child.shared_batch();parent.reset();
        rejects("destroyed KV owner invalidates surviving index owner",[&]{(void)stale.source_layer();});
        rejects("pending consumer cannot use freed borrowed KV",[&]{pending.forward(x,y,3,0,stale,stream.value);});
        rejects("index getter detects destroyed upstream owner",[&]{(void)child.shared_batch();});
        check(pending.next_position()==0,"destroyed-parent rejection preserves consumer");
    }
    {
        SharedAttentionBatch stale;
        {CudaSharedAttention temporary(w.base.view(index_layer).attention,w.query,cap,n);temporary.forward_source20(x,y,n,0,owner.shared_batch(),stream.value);stale=temporary.shared_batch();}
        rejects("destroyed index owner invalidates borrowed indices",[&]{(void)stale.source_layer();});
        check(owner.shared_batch().view(0,n,stream.value).key_count==n,"destroying index owner preserves source20");
    }
    check(index.device_bytes()==bytes,"candidate source retains fixed allocation across reset");
    std::printf("source%d_device_bytes=%zu consumer%d_device_bytes=%zu\n",index_layer,bytes,index_layer+1,consumers[0]->device_bytes());
}
void distinct_query(){
    constexpr int n=513;Weights w;Stream stream;auto negative=w.base.head_weights;
    for(auto& b:negative)b^=0x8000;
    const Source20IndexQueryWeights control_query{w.base.index_query.view(),negative};
    const CandidateIndexQueryWeights candidate_query{w.base.index_query.view(),negative,index_layer};
    CudaSource20Attention source(w.base.view(20).attention,w.compressor,w.key,w.query20,n,n);
    CudaSource20Attention reference(w.base.view(20).attention,w.compressor,w.key,control_query,n,n);
    CudaSharedAttention index(w.base.view(index_layer).attention,candidate_query,n,n),consumer(w.base.view((index_layer+1)).attention,0,(index_layer+1));
    core::DeviceArena io(size_t(n)*hidden*12);auto* x=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* y=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* expected=static_cast<float*>(io.alloc(size_t(n)*hidden*4));
    const auto input=input_values(n);upload(x,input,stream.value);source.forward(x,y,n,0,stream.value);
    const auto original=download(y,input.size(),stream.value);reference.forward(x,expected,n,0,stream.value);
    const auto truth=download(expected,input.size(),stream.value);
    const auto a=source.shared_batch().view(0,n,stream.value),b=reference.shared_batch().view(0,n,stream.value);
    check(ids(a.indices+size_t(n-1)*a.index_count,a.index_count,stream.value)!=ids(b.indices+size_t(n-1)*b.index_count,b.index_count,stream.value),"513-position control selects different candidate source top512");
    check(!same(original,truth),"different query weights change actual attention output");
    check(cache(source.compressed_kv_cache(),stream.value,size_t(n)*512)==cache(reference.compressed_kv_cache(),stream.value,size_t(n)*512) && cache(source.index_key_cache(),stream.value,size_t(n)*128)==cache(reference.index_key_cache(),stream.value,size_t(n)*128),"control uses identical KV and keys");
    const auto preserved=history(source,stream.value);
    index.forward_source20(x,y,n,0,source.shared_batch(),stream.value);
    check(same(download(y,input.size(),stream.value),truth),"candidate source uses own query and matches existing independent-source control");
    consumer.forward(x,y,n,0,index.shared_batch(),stream.value);
    check(same(download(y,input.size(),stream.value),truth),"consumer uses candidate source top512 rather than source20 top512");
    check(history(source,stream.value)==preserved && same(cache(index.kv_cache(),stream.value),cache(reference.kv_cache(),stream.value)),"long prefill preserves source20 and wraps owned window correctly");
}
void independent_groups(){
    constexpr int n=3;Weights w;Stream stream;core::DeviceArena io(size_t(n)*hidden*8);
    auto* x=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* y=static_cast<float*>(io.alloc(size_t(n)*hidden*4));
    upload(x,input_values(n),stream.value);
    const CandidateIndexQueryWeights q24{w.base.index_query.view(),w.base.head_weights,24},q28{w.base.index_query.view(),w.base.head_weights,28};
    CudaSource20Attention owner(w.base.view(20).attention,w.compressor,w.key,w.query20,n,n);
    CudaSharedAttention a(w.base.view(24).attention,q24,n,n),b(w.base.view(28).attention,q28,n,n),consumer(w.base.view(29).attention,0,29);
    owner.forward(x,y,n,0,stream.value);const auto root=owner.shared_batch();const auto saved=history(owner,stream.value);
    a.forward_source20(x,y,n,0,root,stream.value);const auto pa=a.shared_batch();
    b.forward_source20(x,y,n,0,root,stream.value);const auto pb=b.shared_batch();
    check(pa.source_layer()==24 && pb.source_layer()==28,"index owners24/28 publish independently from the same source20 batch");
    rejects("consumer29 rejects current source24 at matching position",[&]{consumer.forward(x,y,n,0,pa,stream.value);});
    check(consumer.next_position()==0,"cross-index-group rejection preserves consumer29");
    a.reset(stream.value);
    rejects("reset24 invalidates only its own publication",[&]{(void)pa.source_layer();});
    check(pb.source_layer()==28,"reset24 leaves source28 publication valid");
    consumer.forward(x,y,n,0,pb,stream.value);
    check(consumer.next_position()==n,"consumer29 can finish after source24 reset");
    a.forward_source20(x,y,n,0,root,stream.value);const auto again=a.shared_batch();b.reset(stream.value);
    check(again.source_layer()==24 && history(owner,stream.value)==saved,"reset28 preserves source24 publication and source20 storage");
    owner.reset(stream.value);
    rejects("resetting common KV owner expires surviving index publication",[&]{(void)again.source_layer();});
}

void independent_three_groups(){
    constexpr int n=3;Weights w;Stream stream;core::DeviceArena io(size_t(n)*hidden*8);
    auto* x=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* y=static_cast<float*>(io.alloc(size_t(n)*hidden*4));
    upload(x,input_values(n),stream.value);
    const CandidateIndexQueryWeights q24{w.base.index_query.view(),w.base.head_weights,24},q28{w.base.index_query.view(),w.base.head_weights,28},q32{w.base.index_query.view(),w.base.head_weights,32};
    CudaSource20Attention owner(w.base.view(20).attention,w.compressor,w.key,w.query20,n,n);
    CudaSharedAttention a(w.base.view(24).attention,q24,n,n),b(w.base.view(28).attention,q28,n,n),c(w.base.view(32).attention,q32,n,n),consumer(w.base.view(33).attention,0,33);
    owner.forward(x,y,n,0,stream.value);const auto root=owner.shared_batch();const auto saved=history(owner,stream.value);
    a.forward_source20(x,y,n,0,root,stream.value);const auto pa=a.shared_batch();
    b.forward_source20(x,y,n,0,root,stream.value);const auto pb=b.shared_batch();
    c.forward_source20(x,y,n,0,root,stream.value);const auto pc=c.shared_batch();
    check(pa.source_layer()==24 && pb.source_layer()==28 && pc.source_layer()==32,"three index owners publish independently from source20");
    for(const auto& wrong:{pa,pb})rejects("consumer33 rejects either earlier index owner at matching position",[&]{consumer.forward(x,y,n,0,wrong,stream.value);});
    check(consumer.next_position()==0,"both cross-group rejections preserve consumer33");
    a.reset(stream.value);rejects("reset24 expires only its own publication",[&]{(void)pa.source_layer();});
    check(pb.source_layer()==28 && pc.source_layer()==32 && history(owner,stream.value)==saved,"reset24 preserves both later groups and source20");
    consumer.forward(x,y,n,0,pc,stream.value);check(consumer.next_position()==n,"consumer33 can finish after source24 reset");
    a.forward_source20(x,y,n,0,root,stream.value);const auto again24=a.shared_batch();
    b.reset(stream.value);rejects("reset28 expires its own publication",[&]{(void)pb.source_layer();});
    check(again24.source_layer()==24 && pc.source_layer()==32,"reset28 preserves both sibling groups");
    b.forward_source20(x,y,n,0,root,stream.value);const auto again28=b.shared_batch();
    c.reset(stream.value);rejects("reset32 expires its own publication",[&]{(void)pc.source_layer();});
    check(again24.source_layer()==24 && again28.source_layer()==28 && history(owner,stream.value)==saved,"reset32 preserves both earlier groups and source20");
    c.forward_source20(x,y,n,0,root,stream.value);const auto again32=c.shared_batch();owner.reset(stream.value);
    for(const auto& stale:{again24,again28,again32})rejects("resetting common source20 expires all three index publications",[&]{(void)stale.source_layer();});
}

void independent_four_groups(){
    constexpr int n=3;Weights w;Stream stream;core::DeviceArena io(size_t(n)*hidden*8);
    auto* x=static_cast<float*>(io.alloc(size_t(n)*hidden*4));auto* y=static_cast<float*>(io.alloc(size_t(n)*hidden*4));
    upload(x,input_values(n),stream.value);
    CudaSource20Attention owner(w.base.view(20).attention,w.compressor,w.key,w.query20,n,n);
    constexpr std::array<int,4> layers{24,28,32,36};
    std::array<std::unique_ptr<CudaSharedAttention>,4> groups;
    std::array<SharedAttentionBatch,4> publications;
    CudaSharedAttention consumer(w.base.view(37).attention,0,37);
    owner.forward(x,y,n,0,stream.value);const auto root=owner.shared_batch();const auto saved=history(owner,stream.value);
    bool identities=true;
    for(size_t i=0;i<layers.size();++i){
        const CandidateIndexQueryWeights query{w.base.index_query.view(),w.base.head_weights,layers[i]};
        groups[i]=std::make_unique<CudaSharedAttention>(w.base.view(layers[i]).attention,query,n,n);
        groups[i]->forward_source20(x,y,n,0,root,stream.value);publications[i]=groups[i]->shared_batch();
        identities=identities && publications[i].source_layer()==layers[i];
    }
    check(identities,"four index owners publish independently from source20");
    for(size_t i=0;i<3;++i)rejects("consumer37 rejects earlier index owner at matching position",[&]{consumer.forward(x,y,n,0,publications[i],stream.value);});
    check(consumer.next_position()==0,"cross-group rejections preserve consumer37");
    for(size_t i=0;i<layers.size();++i){
        groups[i]->reset(stream.value);
        rejects("reset expires only its own index publication",[&]{(void)publications[i].source_layer();});
        bool siblings=history(owner,stream.value)==saved;
        for(size_t j=0;j<layers.size();++j)if(j!=i)siblings=siblings && publications[j].source_layer()==layers[j];
        check(siblings,"reset preserves all three sibling groups and source20");
        if(i==0){consumer.forward(x,y,n,0,publications[3],stream.value);check(consumer.next_position()==n,"consumer37 can finish after source24 reset");}
        groups[i]->forward_source20(x,y,n,0,root,stream.value);publications[i]=groups[i]->shared_batch();
    }
    owner.reset(stream.value);
    for(const auto& stale:publications)rejects("resetting common source20 expires all four index publications",[&]{(void)stale.source_layer();});
}

}
int main(int argc,char** argv){
    if(argc>=3 && std::string(argv[1])=="--layer"){
        const std::string value=argv[2];if(value!="24" && value!="28" && value!="32" && value!="36")return 2;
        index_layer=std::stoi(value);argc-=2;argv+=2;
    }
    if(!(argc==1 || (argc==2 && std::string(argv[1])=="--short")))return 2;
    int n=0;const auto status=cudaGetDeviceCount(&n);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || (status==cudaSuccess&&!n))return 77;
    try{ck(status);ck(cudaSetDevice(0));lifecycle();if(index_layer==28)independent_groups();if(index_layer==32)independent_three_groups();if(index_layer==36)independent_four_groups();if(argc==1)distinct_query();}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());++failures;}
    std::printf("%d checks, %d failures\n",checks,failures);return failures?1:0;
}
