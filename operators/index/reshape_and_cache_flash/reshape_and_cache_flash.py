# ==============================================================================
# reshape_and_cache_flash.py - Reshape and Cache for Flash Attention
# Used in vLLM/TRT-LLM style paged attention
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def reshape_and_cache_flash_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    num_tokens,
    num_heads,
    head_dim,
    block_size,
    key_stride_token,
    key_stride_head,
    value_stride_token,
    value_stride_head,
    cache_stride_block,
    cache_stride_head,
    cache_stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """Reshape key/value and cache them for paged attention."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    if token_idx >= num_tokens:
        return
    
    # Get slot mapping
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    
    block_idx = slot // block_size
    block_offset = slot % block_size
    
    # Load key/value for this token and head
    head_offsets = tl.arange(0, BLOCK_SIZE)
    mask = head_offsets < head_dim
    
    key_ptr_base = key_ptr + token_idx * key_stride_token + head_idx * key_stride_head
    value_ptr_base = value_ptr + token_idx * value_stride_token + head_idx * value_stride_head
    
    k = tl.load(key_ptr_base + head_offsets, mask=mask, other=0.0)
    v = tl.load(value_ptr_base + head_offsets, mask=mask, other=0.0)
    
    # Store to cache
    cache_ptr_base = block_idx * cache_stride_block + head_idx * cache_stride_head + block_offset * cache_stride_seq
    
    key_cache_base = key_cache_ptr + cache_ptr_base
    value_cache_base = value_cache_ptr + cache_ptr_base
    
    tl.store(key_cache_base + head_offsets, k, mask=mask)
    tl.store(value_cache_base + head_offsets, v, mask=mask)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Reshape key/value and cache them for paged attention.
    
    Args:
        key: [num_tokens, num_heads, head_dim]
        value: [num_tokens, num_heads, head_dim]
        key_cache: [num_blocks, num_heads, block_size, head_dim]
        value_cache: [num_blocks, num_heads, block_size, head_dim]
        slot_mapping: [num_tokens] mapping tokens to cache slots
    """
    num_tokens, num_heads, head_dim = key.shape
    block_size = key_cache.size(2)
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    grid = (num_tokens, num_heads)
    with torch.cuda.device(key.device):
        reshape_and_cache_flash_kernel[grid](
            key, value, key_cache, value_cache, slot_mapping,
            num_tokens, num_heads, head_dim, block_size,
            key.stride(0), key.stride(1),
            value.stride(0), value.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )


if __name__ == "__main__":
    num_tokens = 32
    num_heads = 8
    head_dim = 64
    num_blocks = 16
    block_size = 16
    
    key = torch.randn(num_tokens, num_heads, head_dim, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_dim, device="cuda")
    key_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device="cuda")
    value_cache = torch.zeros(num_blocks, num_heads, block_size, head_dim, device="cuda")
    slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    
    reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping)
    
    torch.cuda.synchronize()
    print("reshape_and_cache_flash completed successfully")
