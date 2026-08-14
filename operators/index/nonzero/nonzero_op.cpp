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
// nonzero_op.cpp - Multi-backend Triton JIT Nonzero (Placeholder)
// Note: Full implementation requires two-pass algorithm with prefix sum
// ==============================================================================

#include "nonzero_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor nonzero(const at::Tensor& input) {
  // Placeholder: delegate to PyTorch's implementation
  // Full Triton implementation requires two-pass algorithm:
  // 1. Count nonzero elements per block
  // 2. Prefix sum to get output indices
  // 3. Write indices to output

  TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

  // For now, use fallback
  return at::nonzero(input);
}

TORCH_LIBRARY(nonzero_ops, m) {
  m.def("nonzero(Tensor input) -> Tensor");
}

REGISTER_TRITON_OP(nonzero_ops, "nonzero", nonzero)

}  // namespace my_ops
