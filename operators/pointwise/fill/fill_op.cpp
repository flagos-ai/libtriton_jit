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
// fill_op.cpp - Multi-backend Triton JIT Fill Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "fill_op.h"
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

at::Tensor fill_tensor(const at::Tensor& input, double value) {
  // Output allocation
  at::Tensor out = triton_jit::ops::backend_empty(input.sizes(), input.scalar_type(), input.device());

  // Kernel setup
  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("fill.py"), "fill_kernel");

  constexpr auto cfg = triton_jit::ops::default_pointwise_config();

  const int64_t n = out.numel();
  const unsigned int num_blocks = (n + cfg.tile_size - 1) / cfg.tile_size;

  // Kernel launch
  c10::DeviceGuard guard(out.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

  // Convert value to appropriate type
  float float_value = static_cast<float>(value);
  f(stream, num_blocks, 1, 1, cfg.num_warps, cfg.num_stages, out, float_value, n, cfg.tile_size);

  return out;
}

at::Tensor& fill_tensor_(at::Tensor& input, double value) {
  // Kernel setup
  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("fill.py"), "fill_kernel");

  constexpr auto cfg = triton_jit::ops::default_pointwise_config();

  const int64_t n = input.numel();
  const unsigned int num_blocks = (n + cfg.tile_size - 1) / cfg.tile_size;

  // Kernel launch
  c10::DeviceGuard guard(input.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(input);

  float float_value = static_cast<float>(value);
  f(stream, num_blocks, 1, 1, cfg.num_warps, cfg.num_stages, input, float_value, n, cfg.tile_size);

  return input;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(fill_ops, m) {
  m.def("fill_tensor(Tensor self, float value) -> Tensor");
  m.def("fill_tensor_(Tensor(a!) self, float value) -> Tensor(a!)");
}

REGISTER_TRITON_OP(fill_ops, "fill_tensor", fill_tensor)
REGISTER_TRITON_OP(fill_ops, "fill_tensor_", fill_tensor_)

}  // namespace my_ops
