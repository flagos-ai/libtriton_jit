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
// rwkv_mm_sparsity_op.cpp - Multi-backend RWKV Sparse Matrix Multiply
// ==============================================================================

#include "rwkv_mm_sparsity_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor rwkv_mm_sparsity(const at::Tensor& a, const at::Tensor& b, const at::Tensor& mask) {
  TORCH_CHECK(a.dim() == 2, "A must be 2D");
  TORCH_CHECK(b.dim() == 2, "B must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.size(1);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor mask_contig = mask.contiguous();

  at::Tensor output = triton_jit::ops::backend_empty({M, N}, a.scalar_type(), a.device());

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string("rwkv_mm_sparsity.py"), "rwkv_mm_sparsity_kernel");

  constexpr int64_t BLOCK_M = 64;
  constexpr int64_t BLOCK_N = 64;
  constexpr int64_t BLOCK_K = 32;
  constexpr int num_warps = 4;
  constexpr int num_stages = 2;

  int64_t num_m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
  int64_t num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
  unsigned int num_blocks = num_m_tiles * num_n_tiles;

  c10::DeviceGuard guard(a.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(a);

  f(stream,
    num_blocks,
    1,
    1,
    num_warps,
    num_stages,
    a_contig,
    b_contig,
    mask_contig,
    output,
    M,
    N,
    K,
    a_contig.stride(0),
    a_contig.stride(1),
    b_contig.stride(0),
    b_contig.stride(1),
    int64_t(1),
    output.stride(0),
    output.stride(1),
    BLOCK_M,
    BLOCK_N,
    BLOCK_K);

  return output;
}

TORCH_LIBRARY(rwkv_mm_sparsity_ops, m) {
  m.def("rwkv_mm_sparsity(Tensor a, Tensor b, Tensor mask) -> Tensor");
}

REGISTER_TRITON_OP(rwkv_mm_sparsity_ops, "rwkv_mm_sparsity", rwkv_mm_sparsity)

}  // namespace my_ops
