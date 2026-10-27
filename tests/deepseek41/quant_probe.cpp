// Bounded real GSQ-RCO sample: upstream Strata CUDA kernels, independent ggml
// dequantization + double dot products, and Strata O_DIRECT reads. No new kernel.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/platform/direct_file.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "ds4_iq.cuh"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace iq = strata::ds4::gguf::iq;
namespace io = strata::platform;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p,n)); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
};
struct Stream {
    cudaStream_t s{};
    Stream() { ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(s); }
};
struct Events {
    cudaEvent_t a{},b{};
    Events() { ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b)); }
    ~Events() { cudaEventDestroy(a); cudaEventDestroy(b); }
};
double median(std::vector<double> values) {
    std::sort(values.begin(),values.end()); return values[values.size()/2];
}
struct Span { uint64_t offset; size_t bytes; const uint8_t* data; };

std::string matrix(const strata::GgufFile& file,const strata::TensorInfo& t,cudaStream_t s,bool timing) {
    require(t.shape.size()==2 && iq::supported(int(t.type)),"Unsupported sample matrix");
    const int k=int(t.shape[0]),n=int(t.shape[1]);
    require((k==5120 && n==2304)||(k==2304 && n==5120),"Wrong original expert shape");
    require(t.type==16 || t.type==19 || t.type==22 || t.type==29,"Unexpected sample quantization");
    const size_t row=iq::row_bytes(int(t.type),k),bytes=row*size_t(n);
    require(t.offset<=file.file_size()-file.data_start() && bytes<=file.file_size()-file.data_start()-t.offset,"Sample tensor exceeds file");
    const auto* raw=file.tensor_data(t);
    const auto* traits=ggml_get_type_traits(static_cast<ggml_type>(t.type));
    require(traits && traits->to_float,"No independent ggml dequantizer");
    constexpr int cols=3;
    std::mt19937 rng(1017u+unsigned(t.type)+unsigned(k));
    std::uniform_int_distribution<int> codes(-127,127);
    std::vector<int8_t> q(size_t(k)*cols);
    std::vector<float> d(size_t(k/32)*cols);
    for(int c=0;c<cols;++c) {
        for(int j=0;j<k;++j) q[size_t(c)*k+j]=c==2?0:int8_t(codes(rng));
        for(int j=0;j<k/32;++j) d[size_t(c)*(k/32)+j]=c==2?0.f:float(1+j%13)/512.f;
    }
    std::vector<double> ref(size_t(n)*cols);
    std::vector<float> decoded(k);
    for(int r=0;r<n;++r) {
        traits->to_float(raw+size_t(r)*row,decoded.data(),k);
        for(int c=0;c<cols;++c) {
            double sum=0;
            for(int j=0;j<k;++j) {
                require(std::isfinite(decoded[j]),"Nonfinite decoded weight");
                sum+=double(decoded[j])*double(q[size_t(c)*k+j])*double(d[size_t(c)*(k/32)+j/32]);
            }
            ref[size_t(c)*n+r]=sum;
        }
    }
    Device w(bytes),xq(q.size()),xd(d.size()*4),y(ref.size()*4);
    ck(cudaMemcpyAsync(w.p,raw,bytes,cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(xq.p,q.data(),q.size(),cudaMemcpyHostToDevice,s));
    ck(cudaMemcpyAsync(xd.p,d.data(),d.size()*4,cudaMemcpyHostToDevice,s));
    iq::MvJob job;job.w=w.as<uint8_t>();job.ld=row;job.type=int(t.type);job.k=k;job.n=n;job.ncol=cols;
    job.xq=xq.as<int8_t>();job.xd=xd.as<float>();job.y=y.as<float>();job.ycs=n;
    require(iq::mv(&job,1,s),"Upstream CUDA matrix launch refused");
    std::vector<float> got(ref.size());
    ck(cudaMemcpyAsync(got.data(),y.p,got.size()*4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));
    double maxerr=0,maxref=0,err2=0,ref2=0;
    for(size_t i=0;i<ref.size();++i) {
        require(std::isfinite(got[i]),"Nonfinite GPU result");
        const double e=double(got[i])-ref[i];maxerr=std::max(maxerr,std::abs(e));
        maxref=std::max(maxref,std::abs(ref[i]));err2+=e*e;ref2+=ref[i]*ref[i];
    }
    const double relative=std::sqrt(err2/std::max(ref2,1e-30));
    require(relative<=5e-5 && maxerr<=2e-4*std::max(1.,maxref),"Quantized GPU dot differs from independent CPU reference");
    require(std::all_of(got.begin()+2*n,got.end(),[](float v){return v==0.f;}),"Zero activation not preserved");
    // Single-column projection timing, resident weights, excluding transfer and activation quantization.
    job.ncol=1;Events events;
    for(int i=0;i<(timing?5:0);++i) require(iq::mv(&job,1,s),"Warm-up failed");
    std::vector<double> times;
    for(int trial=0;trial<(timing?7:0);++trial) {
        ck(cudaEventRecord(events.a,s));
        for(int i=0;i<30;++i) require(iq::mv(&job,1,s),"Timing launch failed");
        ck(cudaEventRecord(events.b,s));ck(cudaEventSynchronize(events.b));
        float ms=0;ck(cudaEventElapsedTime(&ms,events.a,events.b));times.push_back(double(ms)*1000./30.);
    }
    std::ostringstream out;out<<std::setprecision(10)<<"{\"tensor\":\""<<t.name<<"\",\"type\":\""<<iq::name(int(t.type))
        <<"\",\"bytes\":"<<bytes<<",\"relative_l2\":"<<relative<<",\"max_abs_error\":"<<maxerr
        <<",\"projection_us_median\":"<<(times.empty()?0.:median(times))<<",\"projection_us_trials\":[";
    for(size_t i=0;i<times.size();++i) out<<(i?",":"")<<times[i];
    out<<"]}";return out.str();
}

size_t engram(const strata::GgufFile& f,const strata::TensorInfo& t,cudaStream_t s) {
    require(t.type==20 && t.shape==std::vector<uint64_t>{256,5},"Wrong Engram sample layout");
    const size_t bytes=5*iq::row_bytes(20,256);
    require(t.offset<=f.file_size()-f.data_start() && bytes<=f.file_size()-f.data_start()-t.offset,"Engram sample exceeds file");
    Device w(bytes),out(5*256*4);
    ck(cudaMemcpyAsync(w.p,f.tensor_data(t),bytes,cudaMemcpyHostToDevice,s));
    strata::kernels::iq_dequant_f32(20,w.p,5*256,out.as<float>(),s);
    std::vector<float> got(5*256),want(5*256);
    ck(cudaMemcpyAsync(got.data(),out.p,got.size()*4,cudaMemcpyDeviceToHost,s));ck(cudaStreamSynchronize(s));
    ggml_get_type_traits(GGML_TYPE_IQ4_NL)->to_float(f.tensor_data(t),want.data(),int(want.size()));
    for(size_t i=0;i<got.size();++i)
        require(std::isfinite(want[i]) && std::memcmp(&got[i],&want[i],sizeof(float))==0,"Engram F32 output differs from ggml");
    return got.size();
}

std::string reads(const std::string& path,const std::vector<Span>& spans) {
    io::DirectFile file;std::string error;
    if(!file.open(path,error)) throw std::runtime_error(error);
    size_t cap=0;for(const auto& v:spans) cap=std::max(cap,((v.offset%4096+v.bytes+4095)/4096)*4096);
    std::unique_ptr<void,decltype(&io::DirectFile::free_aligned)> buffer(io::DirectFile::alloc_aligned(cap),io::DirectFile::free_aligned);
    require(bool(buffer),"DirectFile buffer allocation failed");
    std::vector<double> elapsed;uint64_t logical=0,physical=0;
    // Five complete passes, one request in flight, byte equality checked after timing each read.
    for(int pass=0;pass<5;++pass) {
        double us=0;uint64_t pass_bytes=0;
        for(size_t i=0;i<spans.size();++i) {
            const auto& v=spans[i];const uint64_t offset=v.offset/4096*4096;
            const auto size=uint32_t((v.offset%4096+v.bytes+4095)/4096*4096);
            auto start=std::chrono::steady_clock::now();
            if(!file.submit(offset,buffer.get(),size,i,error)) throw std::runtime_error(error);
            io::Completion c;require(file.wait(&c,1,30000)==1,"DirectFile read timed out");
            us+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
            require(c.ok && c.tag==i && c.bytes>=v.offset%4096+v.bytes,"DirectFile read failed/short");
            require(std::memcmp(static_cast<uint8_t*>(buffer.get())+v.offset%4096,v.data,v.bytes)==0,"Direct read payload mismatch");
            pass_bytes+=c.bytes;if(pass==0)logical+=v.bytes;
        }
        elapsed.push_back(us);if(pass==0)physical=pass_bytes;
    }
    std::ostringstream out;out<<std::setprecision(10)<<"{\"mode\":\"O_DIRECT, one inflight, repeated sample file\",\"reads_per_pass\":"<<spans.size()
        <<",\"logical_bytes_per_pass\":"<<logical<<",\"physical_bytes_per_pass\":"<<physical
        <<",\"median_us\":"<<median(elapsed)<<",\"median_MB_per_s\":"<<double(physical)/median(elapsed)<<",\"pass_us\":[";
    for(size_t i=0;i<elapsed.size();++i)out<<(i?",":"")<<elapsed[i];
    out<<"]}";return out.str();
}

int main(int argc,char** argv) {
    try {
        require(argc==2 || (argc==3 && std::string(argv[2])=="--check-only"),"Usage: strata-deepseek41-quant-probe quant-sample.gguf [--check-only]");
        const bool timing=argc==2;
        strata::GgufFile f(argv[1]);
        auto text=[&](const char* key,const char* value){const auto* p=f.get(key);return p && p->type==strata::MetaType::STRING && p->s==value;};
        require(text("general.architecture","deepseek41-quant-sample") && text("strata.deepseek41.quant_revision","cb02422c900df489cea5833e8a92171224059b3c"),"Wrong sample profile/revision");
        const auto* count=f.get("strata.deepseek41.sample_count");require(count && count->type==strata::MetaType::U32 && count->u==7,"Expected seven bounded experts");
        require(f.tensors().size()==22,"Expected 21 expert matrices and five Engram rows");
        cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));Stream stream;
        std::vector<std::string> results;std::vector<Span> spans;
        for(int i=0;i<7;++i)for(const char* role:{"gate","up","down"}) {
            const auto* t=f.find("sample."+std::to_string(i)+"."+role);require(t,"Missing sample projection");
            results.push_back(matrix(f,*t,stream.s,timing));
            spans.push_back({f.data_start()+t->offset,iq::row_bytes(int(t->type),int64_t(t->shape[0]))*t->shape[1],f.tensor_data(*t)});
        }
        const auto* e=f.find("engram.rows");require(e,"Missing Engram rows");const auto words=engram(f,*e,stream.s);
        const auto io_result=reads(argv[1],spans);
        std::cout<<"QUANT_SAMPLE_RESULT {\"gpu\":\""<<prop.name<<"\",\"sm\":"<<prop.major*10+prop.minor
                 <<",\"matrix_checks\":21,\"columns_per_check\":3,\"engram_f32_values_exact\":"<<words<<",\"matrices\":[";
        for(size_t i=0;i<results.size();++i)std::cout<<(i?",":"")<<results[i];
        std::cout<<"],\"expert_direct_io\":"<<io_result<<"}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
