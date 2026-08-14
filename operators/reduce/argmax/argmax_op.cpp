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
// argmax_op.cpp - Multi-backend Triton JIT Argmax Operation
// ==============================================================================

#include "argmax_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include "ATen/WrapDimUtils.h"

#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor argmax(const at::Tensor& self, int64_t dim, bool keepdim) {
  dim = at::maybe_wrap_dim(dim, self.dim());

  std::vector<int64_t> perm;
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (i != dim) perm.push_back(i);
  }
  perm.push_back(dim);

  at::Tensor permuted = self.permute(perm).contiguous();
  int64_t M = permuted.numel() / permuted.size(-1);
  int64_t N = permuted.size(-1);

  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < self.dim(); ++i) {
    if (i != dim) out_shape.push_back(self.size(i));
  }

  at::Tensor out = triton_jit::ops::backend_empty({M}, at::kLong, self.device());

  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("argmax.py"), "argmax_dim_kernel");

  constexpr int64_t BLOCK_M = 4;
  constexpr int64_t BLOCK_N = 512;
  constexpr int num_warps = 8;
  constexpr int num_stages = 1;
  constexpr int64_t K = 1;  // After permute, reduce dim is last, so K=1
  const unsigned int num_blocks = (M + BLOCK_M - 1) / BLOCK_M;

  c10::DeviceGuard guard(self.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(permuted);

  f(stream, num_blocks, 1, 1, num_warps, num_stages, permuted.view({M, N}), out, M, N, K, BLOCK_M, BLOCK_N);

  // Reshape output - out_shape is already in correct order
  if (!out_shape.empty()) {
    out = out.view(out_shape);
  }

  if (keepdim) {
    out = out.unsqueeze(dim);
  }

  return out;
}

TORCH_LIBRARY(argmax_ops, m) {
  m.def("argmax(Tensor self, int dim, bool keepdim=False) -> Tensor");
}

REGISTER_TRITON_OP(argmax_ops, "argmax", argmax)

}  // namespace my_ops
