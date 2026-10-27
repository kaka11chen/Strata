#pragma once
// Included inside text_prefix_cuda_test.cpp's fixture namespace. Production
// generation contains no test sources, tokenization or full-logit D2H copies.
void generation_synthetic(const fs::path& dir) {
    const auto g=geometry();const Synthetic weights;const auto w=weights.view(4);
    const auto map=engram_fixture::synthetic_map();Layer1EngramHash hash(map);
    const std::array<int32_t,5> evaluated{0,455,0,0,0};
    std::vector<uint32_t> rows(evaluated.size()*24);hash.forward(evaluated,rows,0);
    std::sort(rows.begin(),rows.end());rows.erase(std::unique(rows.begin(),rows.end()),rows.end());
    const auto table=dir/"generation-rows.gguf";
    engram_fixture::sparse(table,engram_fixture::table_meta(true,rows),engram_fixture::table_tensors(),rows);
    constexpr int vocab=456;
    std::vector<uint16_t> embedding(size_t(vocab)*hidden),norm(hidden,0x3f80),head(embedding.size(),0);
    for(size_t i=0;i<embedding.size();++i)embedding[i]=kernels::bf16_from_f32(float(int((i*17+3)%67)-33)/128.f);
    TextIoWeights boundary{hidden,vocab,embedding,norm,head};
    Stream stream,other;ZeroLayers source(g);EngramIoOptions options;options.allow_partial=true;
    CudaTextPrefix prefix(g,w,map,table.string(),source,2,2,2,options);
    CudaTextIo io(boundary,2);CudaGreedyGeneration generation(prefix,io);
    const std::array<int32_t,2> prompt{0,455};
    const size_t bytes=prefix.device_bytes()+io.device_bytes()+generation.device_bytes();
    check(generation.context_capacity()==5 && generation.prefill_capacity()==2,"generation reports evaluated-token and prefill capacities");
    const auto result=generation.generate(prompt,4,stream.value);
    check(result.tokens==std::vector<int32_t>(4,0) && !result.stopped_on_eos,"GPU greedy tie chooses lowest ID and reaches length limit");
    check(prefix.next_position()==5 && prefix.compressed_groups()==2,"last sampled token is returned without an extra forward at capacity");
    const auto before=snapshot(prefix,stream.value);const auto loads=source.loads;
    const auto zero=generation.generate(prompt,0,stream.value);
    check(zero.tokens.empty() && !zero.stopped_on_eos && prefix.next_position()==5 && source.loads==loads && snapshot(prefix,stream.value)==before,"zero-token budget preserves sequence and caches");
    rejects("generation rejects empty prompt",[&]{generation.generate({},1,stream.value);});
    rejects("generation rejects prefill overflow",[&]{generation.generate(evaluated,1,stream.value);});
    rejects("generation rejects evaluated-length overflow before reset",[&]{generation.generate(prompt,5,stream.value);});
    rejects("generation rejects negative budget",[&]{generation.generate(prompt,-1,stream.value);});
    for(int32_t id:{-1,vocab})rejects("generation rejects out-of-vocabulary input before reset",[&]{generation.generate(std::span(&id,1),1,stream.value);});
    alignas(int32_t) std::array<uint8_t,12> misaligned{};
    rejects("generation rejects misaligned tokens before reading IDs",[&]{generation.generate(std::span(reinterpret_cast<const int32_t*>(misaligned.data()+1),2),1,stream.value);});
    rejects("generation rejects null stream",[&]{generation.generate(prompt,1,nullptr);});
    rejects("generation rejects changed stream before reset",[&]{generation.generate(prompt,1,other.value);});
    check(prefix.next_position()==5 && source.loads==loads && snapshot(prefix,stream.value)==before,"all invalid requests preserve prior sequence and caches");
    const auto repeated=generation.generate(prompt,4,stream.value);
    check(repeated.tokens==result.tokens && snapshot(prefix,stream.value)==before,"new request resets all histories and reproduces greedy completion");
    check(prefix.device_bytes()+io.device_bytes()+generation.device_bytes()==bytes,"generation workspace stays fixed across decode and reset");

    // Make EOS and ID2 equal positive maxima using the real final hidden state;
    // this tests the existing GPU sampler and early exit without a sampler mock.
    core::DeviceArena buffers(2*(2*residual_size+8)*sizeof(float)+2048);
    auto alloc=[&](size_t n){return static_cast<float*>(buffers.alloc(n*sizeof(float)));};
    auto *x=alloc(2*residual_size),*pre=alloc(8),*y=alloc(2*residual_size),*next=alloc(8);
    prefix.reset(stream.value);io.embed(prompt,x,pre,stream.value);
    prefix.forward(prompt,x,pre,y,next,0,stream.value);
    const auto normalized=collapse_norm(download(y,2*residual_size,stream.value),download(next,8,stream.value));
    for(int d=0;d<hidden;++d){const auto value=kernels::bf16_from_f32(normalized[size_t(hidden+d)]<0.f?-1.f:1.f);head[size_t(hidden+d)]=value;head[size_t(2*hidden+d)]=value;}
    const auto expected=text_io_reference::head(boundary,download(y,2*residual_size,stream.value),download(next,8,stream.value));
    check(expected[1]>0.f && expected[1]==expected[2] && std::max_element(expected.begin(),expected.end())-expected.begin()==1,"analytical EOS fixture has tied positive maxima at IDs1 and2");
    CudaTextIo eos_io(boundary,2);CudaGreedyGeneration eos_generation(prefix,eos_io);
    const auto eos=eos_generation.generate(prompt,4,stream.value);
    check(eos.tokens==std::vector<int32_t>{1} && eos.stopped_on_eos && prefix.next_position()==2,"EOS is returned once and is not evaluated as another input");
    source.fail_layer=3;
    rejects("generation propagates missing original weight failure",[&]{generation.generate(prompt,2,stream.value);},true);
    rejects("failed generation cannot return a partial or reused result",[&]{generation.generate(prompt,1,stream.value);},true);
}

void generation_original(const fs::path& dir,const fs::path& extra,const char* boundary_path,
                         const fs::path& engram_dir,std::span<const int32_t> prompt,int budget,
                         bool prepare,bool validate,const char* capture_dir,
                         const GenerationMemoryLimits* limits=nullptr,int slots=2,bool plan_only=false,uint64_t cpu_cache_budget=0,const char* expert_gguf=nullptr) {
    if(prompt.empty() || prompt.size()>5)throw std::invalid_argument("Guarded generation supports 1..5 prompt IDs");
    if(budget<0 || budget>5 || (budget && prompt.size()+size_t(budget-1)>5))throw std::invalid_argument("Generation budget exceeds five-input-token profile");
    for(int32_t id:prompt)if(id<0 || id>=129280)throw std::invalid_argument("Generation token outside original vocabulary");
    GenerationMemoryPlan plan;core::DeviceInfo before;uint64_t observed_device=0,cap_bytes=0,held_bytes=0;
    std::unique_ptr<core::DeviceArena> test_reservation;
    if(limits){
        plan=short_generation_memory(slots,int(prompt.size()),cpu_cache_budget);
        // Test runner only: hold the difference from this larger physical card.
        // The native allocator then has at most the declared card size left.
        if(const char* raw=std::getenv("STRATA_DEEPSEEK41_TEST_GPU_CAP_MIB")){
            size_t end=0;const int mib=std::stoi(raw,&end);if(mib<=0 || end!=std::strlen(raw))throw std::invalid_argument("Bad test GPU cap");
            cap_bytes=uint64_t(mib)<<20;const auto initial=core::device_info();
            if(cap_bytes>initial.total_bytes)throw std::invalid_argument("Test cap exceeds physical GPU size");
            held_bytes=initial.total_bytes-cap_bytes;
            if(held_bytes)test_reservation=std::make_unique<core::DeviceArena>(held_bytes);
            std::printf("TEST_DEVICE_CAP {\"cap_bytes\":%llu,\"reserved_bytes\":%llu}\n",(unsigned long long)cap_bytes,(unsigned long long)held_bytes);
        }
        before=core::device_info();const auto host=core::sample_host_memory();
        const MemoryAvailable available{before.free_bytes,host?std::optional<uint64_t>(host->available):std::nullopt};
        const auto reason=memory_rejection(plan,*limits,available);
        std::printf("MEMORY_PLAN {\"profile\":\"1017-full40-context5\",\"expert_slots_per_layer\":%d,\"gpu_budget_bytes\":%llu,\"host_budget_bytes\":%llu,\"gpu_headroom_bytes\":%llu,\"host_headroom_bytes\":%llu,\"dense_device_bytes\":%llu,\"expert_device_bytes\":%llu,\"workspace_device_upper_bytes\":%llu,\"device_upper_bytes\":%llu,\"host_weights_upper_bytes\":%llu,\"host_source_upper_bytes\":%llu,\"host_workspace_upper_bytes\":%llu,\"host_upper_bytes\":%llu,\"cpu_expert_cache_budget_bytes\":%llu,\"host_expert_cache_bytes\":%llu,\"gpu_free_bytes\":%llu,\"host_available_bytes\":%llu,\"host_probe_known\":%s,\"admitted\":%s,\"reason\":\"%s\"}\n",slots,
            (unsigned long long)limits->device_bytes,(unsigned long long)limits->host_bytes,(unsigned long long)limits->device_headroom,(unsigned long long)limits->host_headroom,
            (unsigned long long)plan.dense_device,(unsigned long long)plan.expert_device,(unsigned long long)plan.workspace_device,(unsigned long long)plan.device_bytes(),
            (unsigned long long)plan.weights_host,(unsigned long long)plan.source_host,(unsigned long long)plan.workspace_host,(unsigned long long)plan.host_bytes(),(unsigned long long)cpu_cache_budget,(unsigned long long)plan.expert_cache_host,
            (unsigned long long)before.free_bytes,(unsigned long long)available.host.value_or(0),host?"true":"false",reason.empty()?"true":"false",reason.c_str());
        std::fflush(stdout);require_memory(plan,*limits,available);if(plan_only)return;
    }
    const fs::path capture=capture_dir?fs::path(capture_dir):fs::path();
    if(capture_dir && !fs::create_directory(capture))throw std::runtime_error("Capture directory must be new");
    prefix_fixture::RealWeights f(dir,40);
    f.table=(engram_dir/"engram1-table.gguf").string();f.table14=(engram_dir/"engram14-table.gguf").string();
    const auto model=GgufModel::open(boundary_path);const auto boundary=read_text_io_weights(model);
    Stream stream;EngramIoOptions options;options.allow_partial=true;
    std::unique_ptr<prefix_fixture::Source> source;
    std::unique_ptr<IndexedExpertSource> indexed;
    std::unique_ptr<CachedExpertSource> host_cache;
    uint64_t mapped_expert_bound=0;
    if(limits){
        auto index=expert_gguf?ExpertFileIndex::open_checkpoint(f.g,expert_gguf):prefix_fixture::expert_file_index(f,extra);
        indexed=std::make_unique<IndexedExpertSource>(std::move(index));
        for(const auto& coverage:f.populated)mapped_expert_bound+=uint64_t(std::count(coverage.begin(),coverage.end(),true))*indexed->index().role_bytes()*3;
        if(indexed->host_upper_bytes()+mapped_expert_bound>plan.source_host)
            throw std::runtime_error("Indexed source exceeds the admitted source/staging bound");
        host_cache=std::make_unique<CachedExpertSource>(f.g,*indexed,cpu_cache_budget);
    }else source=std::make_unique<prefix_fixture::Source>(f,extra,prepare);
    core::ExpertSource& active_source=host_cache?static_cast<core::ExpertSource&>(*host_cache):*source;
    CudaTextPrefix prefix(f.g,f.weights,f.map,f.table,active_source,slots,int(prompt.size()),2,options,0,f.map14,f.table14);
    CudaTextIo io(boundary,int(prompt.size()));CudaGreedyGeneration generation(prefix,io);
    const size_t bytes=prefix.device_bytes()+io.device_bytes()+generation.device_bytes();
    auto verify_budget=[&]{
        if(!limits)return;
        if(host_cache->allocated_bytes()!=plan.expert_cache_host)throw std::runtime_error("Actual host cache allocation differs from the startup plan");
        if(bytes>plan.device_bytes())throw std::runtime_error("Actual device arenas exceed the startup plan");
        const auto now=core::device_info();
        const uint64_t delta=before.free_bytes>now.free_bytes?before.free_bytes-now.free_bytes:0;
        observed_device=std::max(observed_device,delta);
        if(observed_device>limits->device_bytes || observed_device>plan.device_bytes()+limits->device_headroom)
            throw std::runtime_error("Observed GPU use exceeds the admitted memory plan");
        if(cap_bytes && now.total_bytes-now.free_bytes-held_bytes>cap_bytes)
            throw std::runtime_error("Observed GPU use exceeds the simulated card capacity");
        const auto host=core::sample_host_memory();
        if(!core::conversation_memory_admit(host?std::optional<uint64_t>(host->available):std::nullopt,0,limits->host_headroom))
            throw std::runtime_error("Live host/cgroup memory no longer retains admission headroom");
    };
    verify_budget();
    std::vector<int32_t> expected;std::vector<std::vector<float>> history;
    std::array<std::set<int>,40> selected{};
    if(validate){
        // Explicit composition + CPU argmax is the control for the production
        // GPU-sampler loop. Captures permit an independent pinned CPU head check.
        core::DeviceArena buffers(prompt.size()*(2*residual_size+8)*sizeof(float)+size_t(boundary.vocab_size)*sizeof(float)+2048);
        auto alloc=[&](size_t n){return static_cast<float*>(buffers.alloc(n*sizeof(float)));};
        auto *x=alloc(prompt.size()*residual_size),*pre=alloc(prompt.size()*4),*y=alloc(prompt.size()*residual_size),*next=alloc(prompt.size()*4),*logits=alloc(size_t(boundary.vocab_size));
        auto save=[&](const std::string& name,const auto& values){if(!capture_dir)return;std::ofstream out(capture/name,std::ios::binary);out.exceptions(std::ios::badbit|std::ios::failbit);out.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*sizeof(values[0])));};
        for(int step=0;step<budget;++step){
            const int32_t last=step?expected.back():0;const auto input=step?std::span(&last,1):prompt;
            io.embed(input,x,pre,stream.value);prefix.forward(input,x,pre,y,next,prefix.next_position(),stream.value);
            io.head(y,next,int(input.size()),logits,stream.value);const auto values=download(logits,size_t(boundary.vocab_size),stream.value);
            check(std::all_of(values.begin(),values.end(),[](float v){return std::isfinite(v);}),"original greedy control has finite full-vocabulary logits step="+std::to_string(step));
            const int32_t id=int32_t(std::max_element(values.begin(),values.end())-values.begin());expected.push_back(id);
            const auto name="case"+std::to_string(step)+".";
            save(name+"tokens.i32",std::vector<int32_t>(input.begin(),input.end()));
            save(name+"embedding.f32",download(x,input.size()*residual_size,stream.value));save(name+"initial.f32",download(pre,input.size()*4,stream.value));
            save(name+"residual.f32",download(y,input.size()*residual_size,stream.value));save(name+"coefficients.f32",download(next,input.size()*4,stream.value));save(name+"logits.f32",values);
            if(id==1)break;
        }
        history=snapshot(prefix,stream.value);selected=source->selected;
    }
    const auto result=generation.generate(prompt,budget,stream.value);
    verify_budget();
    if(validate){
        check(result.tokens==expected,"all40 GPU greedy output IDs match independent CPU argmax feedback loop");
        check(result.stopped_on_eos==(!expected.empty() && expected.back()==1),"original generation reports the correct finish reason");
        check(snapshot(prefix,stream.value)==history && source->selected==selected,"generated-token feedback preserves all40 windows and all shared KV/index histories");
        const uint64_t evaluated_count=prompt.size()+expected.size()-1;
        for(size_t l=0;l<40;++l){const auto st=prefix.stats()[l];
            check(st.hits+st.misses==2*evaluated_count*6 && st.misses==source->loads[l] && st.expert_h2d_bytes==st.misses*nvfp4_expert_bytes(f.g) && st.router_d2h_bytes==2*evaluated_count*384*4,
                  "original generation two-slot cache accounting layer="+std::to_string(l));}
        check(prefix.next_position()==int(prompt.size()+result.tokens.size()-1) && prefix.device_bytes()+io.device_bytes()+generation.device_bytes()==bytes,"original generation position and fixed allocations agree");
    }
    std::printf("GENERATION_RESULT {\"prompt_ids\":[");for(size_t i=0;i<prompt.size();++i)std::printf("%s%d",i?",":"",prompt[i]);
    std::printf("],\"generated_ids\":[");for(size_t i=0;i<result.tokens.size();++i)std::printf("%s%d",i?",":"",result.tokens[i]);
    std::printf("],\"stopped_on_eos\":%s,\"evaluated_tokens\":%d,\"context_capacity\":%d,\"prefix_device_bytes\":%zu,\"io_device_bytes\":%zu,\"generation_device_bytes\":%zu",result.stopped_on_eos?"true":"false",prefix.next_position(),generation.context_capacity(),prefix.device_bytes(),io.device_bytes(),generation.device_bytes());
    if(limits)std::printf(",\"memory\":{\"gpu_budget_bytes\":%llu,\"host_budget_bytes\":%llu,\"device_upper_bytes\":%llu,\"host_upper_bytes\":%llu,\"observed_device_delta_bytes\":%llu,\"expert_slots_per_layer\":%d,\"simulated_card_bytes\":%llu}",(unsigned long long)limits->device_bytes,(unsigned long long)limits->host_bytes,(unsigned long long)plan.device_bytes(),(unsigned long long)plan.host_bytes(),(unsigned long long)observed_device,slots,(unsigned long long)cap_bytes);
    if(host_cache){
        const auto cs=host_cache->stats();const auto io_stats=indexed->io_stats();
        std::printf(",\"host_expert_cache\":{\"allocated_bytes\":%zu,\"resident_payload_bytes\":%zu,\"hits\":%llu,\"misses\":%llu,\"evictions\":%llu,\"upstream_reads\":%lld},\"expert_file_io\":{\"staging_bytes\":%zu,\"reads\":%llu,\"requested_bytes\":%llu,\"completed_bytes\":%llu,\"converted\":%llu,\"failures\":%llu}",
            host_cache->allocated_bytes(),host_cache->resident_bytes(),(unsigned long long)cs.hits,(unsigned long long)cs.misses,(unsigned long long)cs.evictions,(long long)indexed->reads(),
            indexed->reader_staging_bytes(),(unsigned long long)io_stats.reads,(unsigned long long)io_stats.requested_bytes,(unsigned long long)io_stats.completed_bytes,(unsigned long long)io_stats.converted,(unsigned long long)io_stats.failures);
        std::printf(",\"indexed_expert_source\":{\"entries\":%zu,\"files\":%zu,\"metadata_upper_bytes\":%zu,\"staging_bytes\":%zu,\"retained_mapped_expert_upper_bytes\":%llu,\"host_upper_bytes\":%llu,\"logical_payload_bytes\":%llu,\"complete_expert_coverage\":%s}",
            indexed->index().entries(),indexed->index().files(),indexed->index().metadata_upper_bytes(),indexed->staging_bytes(),(unsigned long long)mapped_expert_bound,
            (unsigned long long)(indexed->host_upper_bytes()+mapped_expert_bound),(unsigned long long)io_stats.payload_bytes,expert_gguf?"true":"false");
    }
    std::printf("}\n");
    if(source){std::printf("COMPLETE_ORIGINAL_ROUTES [");for(size_t l=0;l<40;++l){std::printf("%s[",l?",":"");bool first=true;for(int id:source->selected[l]){std::printf("%s%d",first?"":",",id);first=false;}std::printf("]");}std::printf("]\n");}std::fflush(stdout);
}
