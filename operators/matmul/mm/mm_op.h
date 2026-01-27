#ifndef TRITON_JIT_MM_OP_H
#define TRITON_JIT_MM_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor mm(const at::Tensor& a, const at::Tensor& b);

}  // namespace my_ops

#endif  // TRITON_JIT_MM_OP_H
