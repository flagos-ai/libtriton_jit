#ifndef TRITON_JIT_RMS_NORM_OP_H
#define TRITON_JIT_RMS_NORM_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double eps);

}  // namespace my_ops

#endif  // TRITON_JIT_RMS_NORM_OP_H
