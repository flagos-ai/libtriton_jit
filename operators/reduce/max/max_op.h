#ifndef TRITON_JIT_MAX_OP_H
#define TRITON_JIT_MAX_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Compute max along a dimension
 * @return Tuple of (values, indices)
 */
std::tuple<at::Tensor, at::Tensor> max_dim(const at::Tensor& self, int64_t dim, bool keepdim = false);

/**
 * @brief Compute max of all elements
 */
at::Tensor max(const at::Tensor& self);

}  // namespace my_ops

#endif  // TRITON_JIT_MAX_OP_H
