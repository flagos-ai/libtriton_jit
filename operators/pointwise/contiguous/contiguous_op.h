#ifndef TRITON_JIT_CONTIGUOUS_OP_H
#define TRITON_JIT_CONTIGUOUS_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Make tensor contiguous
 * @param input Non-contiguous tensor
 * @return Contiguous tensor copy
 */
at::Tensor contiguous(const at::Tensor& input);

}  // namespace my_ops

#endif  // TRITON_JIT_CONTIGUOUS_OP_H
