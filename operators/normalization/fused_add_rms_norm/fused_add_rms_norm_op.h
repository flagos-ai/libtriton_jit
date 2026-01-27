#ifndef TRITON_JIT_FUSED_ADD_RMS_NORM_OP_H
#define TRITON_JIT_FUSED_ADD_RMS_NORM_OP_H

#include <torch/torch.h>

namespace my_ops {

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, 
    const at::Tensor& weight, double eps);

}  // namespace my_ops

#endif  // TRITON_JIT_FUSED_ADD_RMS_NORM_OP_H
