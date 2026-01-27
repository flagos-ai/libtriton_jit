#ifndef TRITON_JIT_EMBEDDING_OP_H
#define TRITON_JIT_EMBEDDING_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor embedding(const at::Tensor& indices, const at::Tensor& weight);

}  // namespace my_ops

#endif  // TRITON_JIT_EMBEDDING_OP_H
