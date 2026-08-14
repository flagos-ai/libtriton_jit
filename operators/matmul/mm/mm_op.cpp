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
// mm_op.cpp - Multi-backend Triton JIT Matrix Multiplication
// ==============================================================================

#include "mm_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor mm(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2, "Expected 2D tensor for a");
  TORCH_CHECK(b.dim() == 2, "Expected 2D tensor for b");
  TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions must match");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.size(1);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();

  at::Tensor c = triton_jit::ops::backend_empty({M, N}, a.scalar_type(), a.device());

  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("mm.py"), "matmul_kernel");

  constexpr auto cfg = triton_jit::ops::default_matmul_config();

  int64_t grid_m = (M + cfg.BLOCK_M - 1) / cfg.BLOCK_M;
  int64_t grid_n = (N + cfg.BLOCK_N - 1) / cfg.BLOCK_N;
  unsigned int num_blocks = grid_m * grid_n;

  c10::DeviceGuard guard(a.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(a);

  f(stream,
    num_blocks,
    1,
    1,
    cfg.num_warps,
    cfg.num_stages,
    a_contig,
    b_contig,
    c,
    M,
    N,
    K,
    a_contig.stride(0),
    a_contig.stride(1),
    b_contig.stride(0),
    b_contig.stride(1),
    c.stride(0),
    c.stride(1),
    cfg.BLOCK_M,
    cfg.BLOCK_N,
    cfg.BLOCK_K,
    cfg.GROUP_M);

  return c;
}

TORCH_LIBRARY(mm_ops, m) {
  m.def("mm(Tensor self, Tensor other) -> Tensor");
}

REGISTER_TRITON_OP(mm_ops, "mm", mm)

}  // namespace my_ops
