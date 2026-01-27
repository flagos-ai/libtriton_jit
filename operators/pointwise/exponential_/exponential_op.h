#ifndef TRITON_JIT_EXPONENTIAL_OP_H
#define TRITON_JIT_EXPONENTIAL_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Fill tensor with exponentially distributed random numbers (in-place)
 * @param input Tensor to fill
 * @param lambd Rate parameter (lambda)
 * @return Reference to modified tensor
 */
at::Tensor& exponential_(at::Tensor& input, double lambd = 1.0);

}  // namespace my_ops

#endif  // TRITON_JIT_EXPONENTIAL_OP_H
