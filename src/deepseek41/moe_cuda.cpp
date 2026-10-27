#include "strata/deepseek41/moe_cuda.hpp"
#include "strata/kernels/elementwise.hpp"
#include "expert_ops.hpp"

#include <cuda_runtime.h>

namespace strata::deepseek41 {
CudaMoE::CudaMoE(const Geometry& g, int layer, std::span<const uint16_t> router, std::span<const float> bias,
                 core::ExpertSource& source, int slots, const SharedExpertWeights& shared, int device)
    : hidden_(g.hidden), routed_(g, layer, router, bias, source, slots, device), shared_(g, shared, device),
      arena_(size_t(g.hidden) * sizeof(float), device),
      shared_output_(static_cast<float*>(arena_.alloc(size_t(g.hidden) * sizeof(float)))) {}

Route CudaMoE::forward(const float* input, float* output, void* stream) {
    if (failed_)
        throw std::logic_error("DeepSeek MoE must be reconstructed after an execution failure");
    try {
        const Route route = routed_.forward(input, output, stream);
        shared_.forward(input, shared_output_, stream);
        kernels::add_inplace(output, shared_output_, hidden_, stream);
        detail::round_bf16_values(output, output, hidden_, stream);
        const auto status = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        if (status != cudaSuccess)
            throw core::CudaError(std::string("wait for DeepSeek MoE: ") + cudaGetErrorString(status), int(status));
        return route;
    } catch (...) {
        failed_ = true;
        if (stream) (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        throw;
    }
}
} // namespace strata::deepseek41
