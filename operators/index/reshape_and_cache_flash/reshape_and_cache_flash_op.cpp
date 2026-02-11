// ==============================================================================
// reshape_and_cache_flash_op.cpp - Multi-backend Reshape and Cache for Flash Attention
// ==============================================================================

#include "reshape_and_cache_flash_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

void reshape_and_cache_flash(
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    const at::Tensor& slot_mapping) {
    
    TORCH_CHECK(key.dim() == 3, "Key must be 3D [num_tokens, num_heads, head_dim]");
    TORCH_CHECK(value.dim() == 3, "Value must be 3D");
    TORCH_CHECK(key_cache.dim() == 4, "Key cache must be 4D [num_blocks, num_heads, block_size, head_dim]");
    TORCH_CHECK(slot_mapping.dim() == 1, "Slot mapping must be 1D");

    int64_t num_tokens = key.size(0);
    int64_t num_heads = key.size(1);
    int64_t head_dim = key.size(2);
    int64_t block_size = key_cache.size(2);

    at::Tensor key_contig = key.contiguous();
    at::Tensor value_contig = value.contiguous();
    at::Tensor slot_mapping_contig = slot_mapping.contiguous();

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("reshape_and_cache_flash.py"), "reshape_and_cache_flash_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < head_dim) BLOCK_SIZE *= 2;

    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(key.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(key);

    f(stream, num_tokens, num_heads, 1, num_warps, num_stages,
      key_contig, value_contig, key_cache, value_cache, slot_mapping_contig,
      num_tokens, num_heads, head_dim, block_size,
      key_contig.stride(0), key_contig.stride(1),
      value_contig.stride(0), value_contig.stride(1),
      key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
      BLOCK_SIZE);
}

TORCH_LIBRARY(reshape_and_cache_flash_ops, m) {
    m.def("reshape_and_cache_flash(Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, Tensor slot_mapping) -> ()");
}

REGISTER_TRITON_OP(reshape_and_cache_flash_ops, "reshape_and_cache_flash", reshape_and_cache_flash)

}  // namespace my_ops
