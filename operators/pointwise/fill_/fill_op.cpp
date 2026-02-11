// ==============================================================================
// fill_op.cpp - Multi-backend Triton JIT In-place Fill Operation
// ==============================================================================

#include "fill_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor& fill_(at::Tensor& tensor, const at::Scalar& value) {
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous for in-place fill");

    int64_t n_elements = tensor.numel();
    float fill_value = value.toFloat();

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fill_.py"), "fill_kernel");

    constexpr int64_t BLOCK_SIZE = 1024;
    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    int64_t num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    c10::DeviceGuard guard(tensor.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(tensor);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      tensor, fill_value, n_elements, BLOCK_SIZE);

    return tensor;
}

TORCH_LIBRARY(fill_inplace_ops, m) {
    m.def("fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)");
}

REGISTER_TRITON_OP(fill_inplace_ops, "fill_", fill_)

}  // namespace my_ops
