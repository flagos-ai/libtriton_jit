// ==============================================================================
// fill_op.cpp - Multi-backend Triton JIT In-place Fill Operation
// ==============================================================================

#include "fill_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor& fill_(at::Tensor& tensor, const at::Scalar& value) {
  TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous for in-place fill");

  int64_t n_elements = tensor.numel();
  float fill_value = value.toFloat();

  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("fill_.py"), "fill_kernel");

  constexpr auto cfg = triton_jit::ops::default_pointwise_config();

  int64_t num_blocks = (n_elements + cfg.tile_size - 1) / cfg.tile_size;

  c10::DeviceGuard guard(tensor.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(tensor);

  f(stream, num_blocks, 1, 1, cfg.num_warps, cfg.num_stages, tensor, fill_value, n_elements, cfg.tile_size);

  return tensor;
}

TORCH_LIBRARY(fill_inplace_ops, m) {
  m.def("fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)");
}

REGISTER_TRITON_OP(fill_inplace_ops, "fill_", fill_)

}  // namespace my_ops
