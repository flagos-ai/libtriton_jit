#ifndef TRITON_JIT_ADDMM_OP_H
#define TRITON_JIT_ADDMM_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor addmm(const at::Tensor& input, const at::Tensor& a, const at::Tensor& b,
                 const at::Scalar& beta, const at::Scalar& alpha);

}  // namespace my_ops

#endif  // TRITON_JIT_ADDMM_OP_H
