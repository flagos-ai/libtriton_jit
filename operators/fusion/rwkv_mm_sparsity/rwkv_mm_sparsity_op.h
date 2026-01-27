#ifndef TRITON_JIT_RWKV_MM_SPARSITY_OP_H
#define TRITON_JIT_RWKV_MM_SPARSITY_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor rwkv_mm_sparsity(const at::Tensor& a, const at::Tensor& b, const at::Tensor& mask);

}  // namespace my_ops

#endif  // TRITON_JIT_RWKV_MM_SPARSITY_OP_H
