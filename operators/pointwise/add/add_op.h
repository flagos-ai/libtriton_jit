

#include <iostream>

#include "torch/torch.h"

namespace my_ops {

at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_);
}  // namespace my_ops
