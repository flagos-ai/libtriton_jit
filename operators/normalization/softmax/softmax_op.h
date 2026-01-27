#ifndef TRITON_JIT_SOFTMAX_OP_H
#define TRITON_JIT_SOFTMAX_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor softmax(const at::Tensor& input, int64_t dim);

}  // namespace my_ops

#endif  // TRITON_JIT_SOFTMAX_OP_H
