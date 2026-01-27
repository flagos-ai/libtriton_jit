#ifndef TRITON_JIT_CAT_OP_H
#define TRITON_JIT_CAT_OP_H

#include <torch/torch.h>
#include <vector>

namespace my_ops {

at::Tensor cat(const std::vector<at::Tensor>& tensors, int64_t dim);

}  // namespace my_ops

#endif  // TRITON_JIT_CAT_OP_H
