#ifndef TRITON_JIT_RESHAPE_AND_CACHE_FLASH_OP_H
#define TRITON_JIT_RESHAPE_AND_CACHE_FLASH_OP_H

#include <torch/torch.h>

namespace my_ops {

void reshape_and_cache_flash(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping);

}  // namespace my_ops

#endif  // TRITON_JIT_RESHAPE_AND_CACHE_FLASH_OP_H
