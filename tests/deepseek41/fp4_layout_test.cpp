#include "strata/deepseek41/fp4_layout.hpp"
#include "ggml.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

using strata::deepseek41::mxfp4_to_nvfp4;
int main() {
    try {
        std::array<uint8_t, 34> source{};
        for (size_t j = 0; j < 16; ++j) {
            source[1 + j] = uint8_t(j | ((15 - j) << 4));
            source[18 + j] = uint8_t((15 - j) | (j << 4));
        }
        // Exhaustive scale bytes: exact conversion for every supported power of two,
        // explicit rejection of every other one. All sixteen E2M1 codes are included.
        int accepted = 0, rejected = 0;
        for (int scale = 0; scale < 256; ++scale) {
            source[0] = uint8_t(scale);
            source[17] = uint8_t(scale >= 118 && scale <= 135 ? 253 - scale : 127);
            std::vector<uint8_t> converted;
            try {
                converted = mxfp4_to_nvfp4(source);
            } catch (const std::invalid_argument&) {
                if (scale >= 118 && scale <= 135)
                    throw;
                ++rejected;
                continue;
            }
            if (scale < 118 || scale > 135 || converted.size() != 36)
                throw std::runtime_error("unsupported scale accepted or wrong size");
            for (size_t h = 0; h < 2; ++h)
                for (size_t j = 0; j < 16; ++j) {
                    const auto a = converted[4 + h * 16 + j % 8];
                    const auto b = converted[12 + h * 16 + j % 8];
                    const uint8_t restored =
                        j < 8 ? uint8_t((a & 15) | ((b & 15) << 4)) : uint8_t((a >> 4) | (b & 240));
                    if (restored != source[h * 17 + 1 + j])
                        throw std::runtime_error("E2M1 code changed");
                }
            std::array<float, 64> before{}, after{};
            ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(source.data(), before.data(), 64);
            ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(converted.data(), after.data(), 64);
            if (before != after)
                throw std::runtime_error("ggml decoders disagree on converted values");
            ++accepted;
        }
        // Scale117 is accepted only when every code can be halved exactly.
        // Exercise every signed code at every position in either 32-value half;
        // the other half retains all original codes at its ordinary scale.
        constexpr std::array<uint8_t,16> half{0,255,1,255,2,3,4,5,8,255,9,255,10,11,12,13};
        int tiny_exact=0,tiny_refused=0;
        for(int code=0;code<16;++code)for(int pos=0;pos<64;++pos){
            std::array<uint8_t,34> small{};small[0]=small[17]=127;
            for(size_t h=0;h<2;++h)for(size_t j=0;j<16;++j)small[h*17+1+j]=uint8_t(j|((15-j)<<4));
            const size_t owner=size_t(pos/32),local=size_t(pos%32);small[owner*17]=117;
            for(size_t j=0;j<16;++j)small[owner*17+1+j]=0x88;
            auto& byte=small[owner*17+1+local%16];
            byte=local<16?uint8_t((byte&240)|code):uint8_t((byte&15)|(code<<4));
            std::vector<uint8_t> converted;
            try{converted=mxfp4_to_nvfp4(small);}
            catch(const std::invalid_argument&){if(half[size_t(code)]!=255)throw; ++tiny_refused;continue;}
            if(half[size_t(code)]==255)throw std::runtime_error("inexact tiny scale was accepted");
            if(converted[2*owner]!=1 || converted[2*owner+1]!=1)throw std::runtime_error("tiny scale did not normalize to2^-9");
            for(size_t h=0;h<2;++h)for(size_t q=0;q<32;++q){
                const auto original=uint8_t((small[h*17+1+q%16]>>(q<16?0:4))&15);
                const auto restored=uint8_t((converted[4+h*16+(q/16)*8+q%8]>>(q%16<8?0:4))&15);
                if(restored!=(h==owner?half[original]:original))throw std::runtime_error("FP4 remap or signed zero changed");
            }
            std::array<float,64> before{},after{};
            ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(small.data(),before.data(),64);
            ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(converted.data(),after.data(),64);
            if(before!=after)throw std::runtime_error("tiny scale changed decoded weight values");
            ++tiny_exact;
        }
        if(tiny_exact!=768 || tiny_refused!=256)throw std::runtime_error("tiny scale coverage incomplete");
        std::printf("Tiny scale117: %d exact code/position cases, %d inexact cases refused; signed codes preserved numerically\n",tiny_exact,tiny_refused);
        // Complete expert blobs use Strata's standard per-projection scale tail.
        // A scale117 code0.5 cannot use the unscaled adapter, but an exponent
        // shift and global0.5 preserve every value without enlarging any matrix.
        strata::deepseek41::Geometry g;g.hidden=g.expert_width=64;
        std::array<std::vector<uint8_t>,3> storage;
        strata::deepseek41::ExpertView weights;
        const size_t original_part=size_t(g.hidden)*g.expert_width/32*17;
        const size_t native_part=size_t(g.hidden)*g.expert_width/64*36;
        for(int shifted_role=0;shifted_role<3;++shifted_role){
            for(size_t role=0;role<3;++role){
                auto& bytes=storage[role];bytes.resize(original_part);
                for(size_t b=0;b<bytes.size();b+=17){
                    bytes[b]=127;
                    for(size_t j=0;j<16;++j)bytes[b+1+j]=uint8_t(j|((15-j)<<4));
                }
                bytes[0]=role==size_t(shifted_role)?117:135;
                if(role==size_t(shifted_role))bytes[17]=134;
                weights.parts[role].type=39;weights.parts[role].data=bytes;
            }
            std::vector<uint8_t> packed(strata::deepseek41::nvfp4_expert_bytes(g));
            strata::deepseek41::pack_nvfp4_expert(g,weights,packed);
            if(packed.size()!=3*native_part+16)throw std::runtime_error("missing standard NVFP4 scale tail");
            std::array<float,4> tail{};std::memcpy(tail.data(),packed.data()+3*native_part,16);
            if(tail[3]!=0.f)throw std::runtime_error("nonzero scale-tail reserved field");
            for(size_t role=0;role<3;++role){
                if(tail[role]!=(role==size_t(shifted_role)?.5f:1.f))throw std::runtime_error("wrong per-projection scale");
                std::vector<float> a(4096),z(4096);
                ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(storage[role].data(),a.data(),4096);
                ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(packed.data()+role*native_part,z.data(),4096);
                for(float& value:z)value*=tail[role];
                if(a!=z)throw std::runtime_error("scaled cache changed original weight values");
                if(role!=size_t(shifted_role)){
                    const auto old=mxfp4_to_nvfp4(storage[role]);
                    if(std::memcmp(old.data(),packed.data()+role*native_part,old.size()))throw std::runtime_error("ordinary payload changed");
                }
            }
        }
        for(int invalid:{135,116,255}){
            storage[2][0]=117;storage[2][17]=uint8_t(invalid);
            std::vector<uint8_t> packed(strata::deepseek41::nvfp4_expert_bytes(g));bool refused=false;
            try{strata::deepseek41::pack_nvfp4_expert(g,weights,packed);}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused)throw std::runtime_error("inexact shifted range accepted");
        }
        std::printf("Scaled expert cache: all3 projection positions exact, old payloads retained, incompatible ranges refused\n");
        std::vector<uint8_t> metadata(strata::deepseek41::nvfp4_expert_bytes(g),0);
        auto write_tail=[&](const std::array<float,4>& tail){std::memcpy(metadata.data()+metadata.size()-16,tail.data(),16);};
        write_tail({1.f,.5f,1.f,0.f});
        if(strata::deepseek41::nvfp4_expert_scales(metadata)!=std::array<float,3>{1.f,.5f,1.f})throw std::runtime_error("scale tail parsed incorrectly");
        for(float invalid:{0.f,-.5f,2.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
            write_tail({1.f,invalid,1.f,0.f});bool refused=false;
            try{(void)strata::deepseek41::nvfp4_expert_scales(metadata);}
            catch(const std::invalid_argument&){refused=true;}
            if(!refused)throw std::runtime_error("invalid scale tail accepted");
        }
        write_tail({1.f,.5f,1.f,1.f});bool reserved_refused=false,empty_refused=false;
        try{(void)strata::deepseek41::nvfp4_expert_scales(metadata);}catch(const std::invalid_argument&){reserved_refused=true;}
        try{(void)strata::deepseek41::nvfp4_expert_scales({});}catch(const std::invalid_argument&){empty_refused=true;}
        if(!reserved_refused || !empty_refused)throw std::runtime_error("malformed scale tail accepted");
        std::printf("Scale metadata: valid roles retained; zero/negative/unsupported/nonfinite/reserved/empty tails refused\n");
        for (size_t n : {size_t(0), size_t(17), size_t(33)}) {
            bool refused = false;
            try {
                (void)mxfp4_to_nvfp4(std::span(source).first(n));
            } catch (const std::invalid_argument&) {
                refused = true;
            }
            if (!refused)
                throw std::runtime_error("partial or empty pair accepted");
        }
        std::printf(
            "FP4: %d exact scales, %d refused scales; all codes preserved; ggml decoders agree; bad sizes refused\n",
            accepted, rejected);
        return accepted == 18 && rejected == 238 ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
