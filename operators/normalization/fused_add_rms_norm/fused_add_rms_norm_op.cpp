// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ==============================================================================
// fused_add_rms_norm_op.cpp - Multi-backend Fused Add + RMS Normalization
// ==============================================================================

#include "fused_add_rms_norm_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm(const at::Tensor& input,
                                                      const at::Tensor& residual,
                                                      const at::Tensor& weight,
                                                      double eps) {
  TORCH_CHECK(input.sizes() == residual.sizes(), "Input and residual must have same shape");
  TORCH_CHECK(input.size(-1) == weight.size(0), "Hidden dim must match weight size");

  auto orig_shape = input.sizes().vec();
  int64_t hidden_size = input.size(-1);
  int64_t n_rows = input.numel() / hidden_size;

  // The kernel operates in-place, so we need mutable copies
  at::Tensor x_flat = input.view({n_rows, hidden_size}).contiguous().clone();
  at::Tensor res_flat = residual.view({n_rows, hidden_size}).contiguous().clone();

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string("fused_add_rms_norm.py"), "fused_add_rms_norm_kernel");

  int64_t BLOCK_SIZE = 1;
  while (BLOCK_SIZE < hidden_size) BLOCK_SIZE *= 2;

  constexpr auto cfg = triton_jit::ops::default_norm_config();
  if constexpr (cfg.max_block_size > 0) {
    BLOCK_SIZE = std::min(BLOCK_SIZE, cfg.max_block_size);
  }

  c10::DeviceGuard guard(input.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

  // Kernel signature:
  // fused_add_rms_norm_kernel(X, R, W, x_stride_r, x_stride_c, r_stride_r, r_stride_c, N, eps, BLOCK_SIZE)
  // The kernel modifies X and R in-place
  f(stream,
    n_rows,
    1,
    1,
    cfg.num_warps,
    cfg.num_stages,
    x_flat,                   // X: input (modified in-place)
    res_flat,                 // R: residual (modified in-place)
    weight,                   // W: weight
    x_flat.stride(0),         // x_stride_r
    x_flat.stride(1),         // x_stride_c
    res_flat.stride(0),       // r_stride_r
    res_flat.stride(1),       // r_stride_c
    hidden_size,              // N
    static_cast<float>(eps),  // eps
    BLOCK_SIZE);              // BLOCK_SIZE (constexpr)

  return std::make_tuple(x_flat.view(orig_shape), res_flat.view(orig_shape));
}

TORCH_LIBRARY(fused_add_rms_norm_ops, m) {
  m.def("fused_add_rms_norm(Tensor input, Tensor residual, Tensor weight, float eps) -> (Tensor, Tensor)");
}

REGISTER_TRITON_OP(fused_add_rms_norm_ops, "fused_add_rms_norm", fused_add_rms_norm)

}  // namespace my_ops
