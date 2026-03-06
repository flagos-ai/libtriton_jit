// ==============================================================================
// contiguous_op.cpp - Multi-backend Triton JIT Contiguous Operation
// ==============================================================================

#include "contiguous_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor contiguous(const at::Tensor& input) {
  // If already contiguous, return as-is
  if (input.is_contiguous()) {
    return input;
  }

  // Allocate contiguous output
  at::Tensor out = triton_jit::ops::backend_empty(input.sizes(), input.scalar_type(), input.device());

  c10::DeviceGuard guard(input.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

  if (input.dim() == 1 || input.numel() == 0) {
    // 1D case: use simple copy kernel
    const TritonJITFunction& f =
        TritonJITFunction::get_instance(std::string("contiguous.py"), "copy_1d_kernel");

    constexpr auto cfg = triton_jit::ops::default_pointwise_config();

    const int64_t n = input.numel();
    const unsigned int num_blocks = (n + cfg.tile_size - 1) / cfg.tile_size;

    f(stream,
      num_blocks,
      1,
      1,
      cfg.num_warps,
      cfg.num_stages,
      input.flatten(),
      out.flatten(),
      n,
      cfg.tile_size);
  } else {
    // Multi-dimensional case: use 2D strided kernel
    const TritonJITFunction& f =
        TritonJITFunction::get_instance(std::string("contiguous.py"), "copy_strided_2d_kernel");

    // Flatten to 2D
    auto input_2d = input.view({-1, input.size(-1)});
    auto out_2d = out.view({-1, out.size(-1)});

    const int64_t n_rows = input_2d.size(0);
    const int64_t n_cols = input_2d.size(1);

    constexpr int64_t BLOCK_M = 32;
    constexpr int64_t BLOCK_N = 32;
    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    const unsigned int grid_m = (n_rows + BLOCK_M - 1) / BLOCK_M;
    const unsigned int grid_n = (n_cols + BLOCK_N - 1) / BLOCK_N;

    f(stream,
      grid_m,
      grid_n,
      1,
      num_warps,
      num_stages,
      input_2d,
      out_2d,
      n_rows,
      n_cols,
      input_2d.stride(0),
      input_2d.stride(1),
      out_2d.stride(0),
      out_2d.stride(1),
      BLOCK_M,
      BLOCK_N);
  }

  return out;
}

TORCH_LIBRARY(contiguous_ops, m) {
  m.def("contiguous(Tensor self) -> Tensor");
}

REGISTER_TRITON_OP(contiguous_ops, "contiguous", contiguous)

}  // namespace my_ops
