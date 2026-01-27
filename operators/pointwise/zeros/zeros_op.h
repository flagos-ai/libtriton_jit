#ifndef TRITON_JIT_ZEROS_OP_H
#define TRITON_JIT_ZEROS_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Create zero tensor with same shape and dtype as input
 */
at::Tensor zeros_like(const at::Tensor& input);

/**
 * @brief Create zero tensor with specified shape
 */
at::Tensor zeros(at::IntArrayRef size, at::ScalarType dtype, const at::Device& device);

}  // namespace my_ops

#endif  // TRITON_JIT_ZEROS_OP_H
