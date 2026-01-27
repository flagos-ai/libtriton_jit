#ifndef TRITON_JIT_TOPK_OP_H
#define TRITON_JIT_TOPK_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Compute top-k values and indices along a dimension
 */
std::tuple<at::Tensor, at::Tensor> topk(
    const at::Tensor& self,
    int64_t k,
    int64_t dim = -1,
    bool largest = true,
    bool sorted = true);

}  // namespace my_ops

#endif  // TRITON_JIT_TOPK_OP_H
