#ifndef TRITON_JIT_ARGMAX_OP_H
#define TRITON_JIT_ARGMAX_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Compute argmax along a dimension
 */
at::Tensor argmax(const at::Tensor& self, int64_t dim, bool keepdim = false);

}  // namespace my_ops

#endif  // TRITON_JIT_ARGMAX_OP_H
