#ifndef TRITON_JIT_FILL_INPLACE_OP_H
#define TRITON_JIT_FILL_INPLACE_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor& fill_(at::Tensor& tensor, const at::Scalar& value);

}  // namespace my_ops

#endif  // TRITON_JIT_FILL_INPLACE_OP_H
