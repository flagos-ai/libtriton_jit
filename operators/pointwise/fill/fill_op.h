#ifndef TRITON_JIT_FILL_OP_H
#define TRITON_JIT_FILL_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Fill tensor with a constant value (out-of-place)
 * @param input Input tensor (shape is preserved)
 * @param value Fill value
 * @return New tensor filled with value
 */
at::Tensor fill_tensor(const at::Tensor& input, double value);

/**
 * @brief Fill tensor with a constant value (in-place)
 * @param input Input tensor to modify
 * @param value Fill value
 * @return Reference to modified input
 */
at::Tensor& fill_tensor_(at::Tensor& input, double value);

}  // namespace my_ops

#endif  // TRITON_JIT_FILL_OP_H
