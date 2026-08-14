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
// rwkv_ka_fusion_op.cpp - Multi-backend RWKV Key-Attention Fusion
// ==============================================================================

#include "rwkv_ka_fusion_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor rwkv_ka_fusion(const at::Tensor& k, const at::Tensor& a) {
  TORCH_CHECK(k.sizes() == a.sizes(), "K and A must have same shape");
  TORCH_CHECK(k.dim() == 3, "Expected 3D tensors [batch, seq, hidden]");

  int64_t batch_size = k.size(0);
  int64_t seq_len = k.size(1);
  int64_t hidden_dim = k.size(2);

  at::Tensor k_contig = k.contiguous();
  at::Tensor a_contig = a.contiguous();

  at::Tensor output =
      triton_jit::ops::backend_empty({batch_size, seq_len, hidden_dim}, k.scalar_type(), k.device());

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string("rwkv_ka_fusion.py"), "rwkv_ka_fusion_kernel");

  int64_t BLOCK_SIZE = 1;
  while (BLOCK_SIZE < hidden_dim) BLOCK_SIZE *= 2;

  constexpr int num_warps = 4;
  constexpr int num_stages = 1;

  c10::DeviceGuard guard(k.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(k);

  // GCU: grid.y limit is 255, swap grid dims so seq_len uses grid.x (limit 65535)
  // Kernel uses program_id(0) for batch, program_id(1) for seq
  // We launch with swapped grid and use a swapped-axis kernel variant
  unsigned int grid_x = static_cast<unsigned int>(batch_size);
  unsigned int grid_y = static_cast<unsigned int>(seq_len);
#if defined(BACKEND_GCU)
  if (grid_y > 255) {
    grid_x = static_cast<unsigned int>(seq_len);
    grid_y = static_cast<unsigned int>(batch_size);
    const TritonJITFunction& f_swap =
        TritonJITFunction::get_instance(std::string("rwkv_ka_fusion.py"), "rwkv_ka_fusion_kernel_swapped");
    f_swap(stream,
      grid_x, grid_y, 1,
      num_warps, num_stages,
      k_contig, a_contig, output,
      batch_size, seq_len, hidden_dim,
      k_contig.stride(0), k_contig.stride(1),
      a_contig.stride(0), a_contig.stride(1),
      output.stride(0), output.stride(1),
      BLOCK_SIZE);
    return output;
  }
#endif

  f(stream,
    grid_x, grid_y, 1,
    num_warps,
    num_stages,
    k_contig,
    a_contig,
    output,
    batch_size,
    seq_len,
    hidden_dim,
    k_contig.stride(0),
    k_contig.stride(1),
    a_contig.stride(0),
    a_contig.stride(1),
    output.stride(0),
    output.stride(1),
    BLOCK_SIZE);

  return output;
}

TORCH_LIBRARY(rwkv_ka_fusion_ops, m) {
  m.def("rwkv_ka_fusion(Tensor k, Tensor a) -> Tensor");
}

REGISTER_TRITON_OP(rwkv_ka_fusion_ops, "rwkv_ka_fusion", rwkv_ka_fusion)

}  // namespace my_ops
