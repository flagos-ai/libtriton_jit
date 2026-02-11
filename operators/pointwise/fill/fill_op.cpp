// ==============================================================================
// fill_op.cpp - Multi-backend Triton JIT Fill Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "fill_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

// ==============================================================================
//                         KERNEL IMPLEMENTATION
// ==============================================================================

namespace my_ops {
using namespace triton_jit;

at::Tensor fill_tensor(const at::Tensor& input, double value) {
    // Output allocation
    at::Tensor out = triton_jit::ops::backend_empty(input.sizes(), input.scalar_type(), input.device());

    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fill.py"), "fill_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    // Kernel launch
    c10::DeviceGuard guard(out.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

    // Convert value to appropriate type
    float float_value = static_cast<float>(value);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, out, float_value, n, tile_size);

    return out;
}

at::Tensor& fill_tensor_(at::Tensor& input, double value) {
    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fill.py"), "fill_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = input.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    // Kernel launch
    c10::DeviceGuard guard(input.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

    float float_value = static_cast<float>(value);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, input, float_value, n, tile_size);

    return input;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(fill_ops, m) {
    m.def("fill_tensor(Tensor self, float value) -> Tensor");
    m.def("fill_tensor_(Tensor(a!) self, float value) -> Tensor(a!)");
}

REGISTER_TRITON_OP(fill_ops, "fill_tensor", fill_tensor)
REGISTER_TRITON_OP(fill_ops, "fill_tensor_", fill_tensor_)

}  // namespace my_ops
