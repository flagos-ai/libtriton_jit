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
