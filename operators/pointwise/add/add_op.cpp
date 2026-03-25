// ==============================================================================
// add_op.cpp - Multi-backend Triton JIT Add Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "add_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

// ==============================================================================
//                         KERNEL IMPLEMENTATION
// ==============================================================================

namespace my_ops {
using namespace triton_jit;

at::Tensor add_tensor(const at::Tensor& a_, const at::Tensor& b_) {
  // ------------------------- Input Preparation -----------------------------
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor& a = res[0];
  const at::Tensor& b = res[1];

  // ------------------------- Output Allocation -----------------------------
  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());

  at::Tensor out = triton_jit::ops::backend_empty(a.sizes(), out_dtype, a.device());

  // ------------------------- Kernel Parameters -----------------------------
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string("add.py"), "binary_pointwise_kernel");

  constexpr auto cfg = triton_jit::ops::default_pointwise_config();

  const int64_t n = out.numel();
  const unsigned int num_blocks = (n + cfg.tile_size - 1) / cfg.tile_size;

  // ------------------------- Kernel Launch ---------------------------------
  c10::DeviceGuard guard(out.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(a);

  // Launch kernel
  f(stream, num_blocks, 1, 1, cfg.num_warps, cfg.num_stages, a, b, out, n, cfg.tile_size);

  return out;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

// Backend-specific dispatch key registration
REGISTER_TRITON_OP(my_ops, "add_tensor", add_tensor)

}  // namespace my_ops
