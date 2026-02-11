// ==============================================================================
// rms_norm_op.cpp - Multi-backend Triton JIT RMS Normalization
// ==============================================================================

#include "rms_norm_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps) {
    TORCH_CHECK(input.size(-1) == weight.size(0), "Hidden dim must match weight size");

    auto orig_shape = input.sizes().vec();
    int64_t hidden_size = input.size(-1);
    int64_t n_rows = input.numel() / hidden_size;

    at::Tensor x_flat = input.view({n_rows, hidden_size}).contiguous();

    at::Tensor output = triton_jit::ops::backend_empty(
        {n_rows, hidden_size}, input.scalar_type(), input.device());

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("rms_norm.py"), "rms_norm_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < hidden_size) BLOCK_SIZE *= 2;

    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(input.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

    f(stream, n_rows, 1, 1, num_warps, num_stages,
      x_flat, weight, output,
      x_flat.stride(0), output.stride(0),
      hidden_size, static_cast<float>(eps),
      BLOCK_SIZE);

    return output.view(orig_shape);
}

TORCH_LIBRARY(rms_norm_ops, m) {
    m.def("rms_norm(Tensor input, Tensor weight, float eps) -> Tensor");
}

REGISTER_TRITON_OP(rms_norm_ops, "rms_norm", rms_norm)

}  // namespace my_ops
