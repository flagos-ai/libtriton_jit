# ==============================================================================
# apply_rotary_pos_emb.py - Rotary Position Embedding Triton Kernel
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def apply_rotary_pos_emb_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    seq_len,
    num_heads,
    head_dim,
    rotary_dim,
    q_stride_seq,
    q_stride_head,
    k_stride_seq,
    k_stride_head,
    cos_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary position embeddings to query and key tensors."""
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    if seq_idx >= seq_len:
        return
    
    half_rotary = rotary_dim // 2
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process first half and second half together
    mask_rotary = offsets < half_rotary
    mask_full = offsets < head_dim
    
    # Load cos/sin for this position
    cos_ptr_base = cos_ptr + seq_idx * cos_stride
    sin_ptr_base = sin_ptr + seq_idx * cos_stride
    
    cos_val = tl.load(cos_ptr_base + offsets, mask=mask_rotary, other=1.0)
    sin_val = tl.load(sin_ptr_base + offsets, mask=mask_rotary, other=0.0)
    
    # Load query
    q_base = q_ptr + seq_idx * q_stride_seq + head_idx * q_stride_head
    q_first = tl.load(q_base + offsets, mask=mask_rotary, other=0.0)
    q_second = tl.load(q_base + half_rotary + offsets, mask=mask_rotary, other=0.0)
    
    # Apply rotation to query
    q_rot_first = q_first * cos_val - q_second * sin_val
    q_rot_second = q_first * sin_val + q_second * cos_val
    
    # Store rotated query
    q_out_base = q_out_ptr + seq_idx * q_stride_seq + head_idx * q_stride_head
    tl.store(q_out_base + offsets, q_rot_first, mask=mask_rotary)
    tl.store(q_out_base + half_rotary + offsets, q_rot_second, mask=mask_rotary)
    
    # Load key
    k_base = k_ptr + seq_idx * k_stride_seq + head_idx * k_stride_head
    k_first = tl.load(k_base + offsets, mask=mask_rotary, other=0.0)
    k_second = tl.load(k_base + half_rotary + offsets, mask=mask_rotary, other=0.0)

    # Apply rotation to key
    k_rot_first = k_first * cos_val - k_second * sin_val
    k_rot_second = k_first * sin_val + k_second * cos_val

    # Store rotated key
    k_out_base = k_out_ptr + seq_idx * k_stride_seq + head_idx * k_stride_head
    tl.store(k_out_base + offsets, k_rot_first, mask=mask_rotary)
    tl.store(k_out_base + half_rotary + offsets, k_rot_second, mask=mask_rotary)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int = None,
) -> tuple:
    """Apply rotary position embeddings to query and key.
    
    Args:
        q: Query tensor [seq_len, num_heads, head_dim]
        k: Key tensor [seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, rotary_dim//2]
        sin: Sine values [seq_len, rotary_dim//2]
        rotary_dim: Dimension to apply rotation (default: head_dim)
    
    Returns:
        Tuple of rotated (query, key)
    """
    seq_len, num_heads, head_dim = q.shape
    if rotary_dim is None:
        rotary_dim = head_dim
    
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    grid = (seq_len, num_heads)
    with torch.cuda.device(q.device):
        apply_rotary_pos_emb_kernel[grid](
            q, k, cos, sin, q_out, k_out,
            seq_len, num_heads, head_dim, rotary_dim,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            cos.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    
    return q_out, k_out


if __name__ == "__main__":
    seq_len, num_heads, head_dim = 128, 8, 64
    
    q = torch.randn(seq_len, num_heads, head_dim, device="cuda")
    k = torch.randn(seq_len, num_heads, head_dim, device="cuda")
    cos = torch.randn(seq_len, head_dim // 2, device="cuda")
    sin = torch.randn(seq_len, head_dim // 2, device="cuda")
    
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    
    torch.cuda.synchronize()
    print(f"Q output shape: {q_out.shape}")
    print(f"K output shape: {k_out.shape}")
    print("apply_rotary_pos_emb completed successfully")
