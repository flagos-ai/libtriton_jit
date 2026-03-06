// ==============================================================================
// exponential_op.cpp - Multi-backend Triton JIT Exponential_ Operation
// ==============================================================================

#include "exponential_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor& exponential_(at::Tensor& input, double lambd) {
  // Ensure tensor is float type
  TORCH_CHECK(input.is_floating_point(), "exponential_ requires floating point tensor");

  // First fill with uniform random
#if defined(BACKEND_MUSA)
  // For MUSA, generate uniform on CPU and copy
  auto cpu_uniform = at::rand(input.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(at::kCPU));
  musaMemcpy(input.data_ptr(),
             cpu_uniform.data_ptr(),
             input.numel() * at::elementSize(input.scalar_type()),
             musaMemcpyHostToDevice);
#else
  input.uniform_(0, 1);
#endif

  // Kernel setup
  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string("exponential_.py"), "exponential_kernel");

  constexpr auto cfg = triton_jit::ops::default_pointwise_config();

  const int64_t n = input.numel();
  const unsigned int num_blocks = (n + cfg.tile_size - 1) / cfg.tile_size;

  c10::DeviceGuard guard(input.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

  float float_lambd = static_cast<float>(lambd);
  f(stream, num_blocks, 1, 1, cfg.num_warps, cfg.num_stages, input, input, float_lambd, n, cfg.tile_size);

  return input;
}

TORCH_LIBRARY(exponential_ops, m) {
  m.def("exponential_(Tensor(a!) self, float lambd=1.0) -> Tensor(a!)");
}

REGISTER_TRITON_OP(exponential_ops, "exponential_", exponential_)

}  // namespace my_ops
