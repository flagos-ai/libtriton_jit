#ifndef TRITON_JIT_RWKV_KA_FUSION_OP_H
#define TRITON_JIT_RWKV_KA_FUSION_OP_H

#include <torch/torch.h>

namespace my_ops {

at::Tensor rwkv_ka_fusion(const at::Tensor& k, const at::Tensor& a);

}  // namespace my_ops

#endif  // TRITON_JIT_RWKV_KA_FUSION_OP_H
