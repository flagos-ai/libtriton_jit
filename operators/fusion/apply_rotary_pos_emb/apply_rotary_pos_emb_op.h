#ifndef TRITON_JIT_APPLY_ROTARY_POS_EMB_OP_H
#define TRITON_JIT_APPLY_ROTARY_POS_EMB_OP_H

#include <torch/torch.h>

namespace my_ops {

std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rotary_dim);

}  // namespace my_ops

#endif  // TRITON_JIT_APPLY_ROTARY_POS_EMB_OP_H
