// ==============================================================================
// zeros_op.cpp - Multi-backend Triton JIT Zeros Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "zeros_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor zeros_like(const at::Tensor& input) {
    // Allocate output
    at::Tensor out = triton_jit::ops::backend_empty(input.sizes(), input.scalar_type(), input.device());

    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("zeros.py"), "zeros_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    c10::DeviceGuard guard(out.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

    f(stream, num_blocks, 1, 1, num_warps, num_stages, out, n, tile_size);

    return out;
}

at::Tensor zeros(at::IntArrayRef size, at::ScalarType dtype, const at::Device& device) {
    // Allocate output
    at::Tensor out = triton_jit::ops::backend_empty(size, dtype, device);

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("zeros.py"), "zeros_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    c10::DeviceGuard guard(out.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(out);

    f(stream, num_blocks, 1, 1, num_warps, num_stages, out, n, tile_size);

    return out;
}

TORCH_LIBRARY(zeros_ops, m) {
    m.def("zeros_like(Tensor self) -> Tensor");
}

REGISTER_TRITON_OP(zeros_ops, "zeros_like", zeros_like)

}  // namespace my_ops
