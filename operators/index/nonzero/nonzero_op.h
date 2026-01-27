#ifndef TRITON_JIT_NONZERO_OP_H
#define TRITON_JIT_NONZERO_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor nonzero(const at::Tensor& input);

}  // namespace my_ops

#endif  // TRITON_JIT_NONZERO_OP_H
