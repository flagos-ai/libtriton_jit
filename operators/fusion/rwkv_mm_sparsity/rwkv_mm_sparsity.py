# ==============================================================================
# rwkv_mm_sparsity.py - RWKV Matrix Multiply with Sparsity Triton Kernel
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def rwkv_mm_sparsity_kernel(
    a_ptr,
    b_ptr,
    mask_ptr,
    output_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mask,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """RWKV sparse matrix multiplication kernel.
    
    Computes C = A @ B with sparsity mask applied.
    """
    pid = tl.program_id(0)
    
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    
    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles
    
    offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Check sparsity mask
    mask_val = tl.load(mask_ptr + tile_m * stride_mask, mask=tile_m < num_m_tiles)
    
    # If masked out, skip computation
    if mask_val == 0:
        # Store zeros
        c_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=c_mask)
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    # Store result
    c_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def rwkv_mm_sparsity(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RWKV sparse matrix multiplication.
    
    Args:
        a: Matrix A [M, K]
        b: Matrix B [K, N]
        mask: Sparsity mask [M // BLOCK_M]
    
    Returns:
        Result matrix [M, N] with sparsity applied
    """
    assert a.dim() == 2 and b.dim() == 2, "A and B must be 2D"
    assert a.size(1) == b.size(0), "Inner dimensions must match"
    
    M, K = a.shape
    _, N = b.shape
    
    a = a.contiguous()
    b = b.contiguous()
    mask = mask.contiguous()
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    num_m_tiles = triton.cdiv(M, BLOCK_M)
    num_n_tiles = triton.cdiv(N, BLOCK_N)
    grid = (num_m_tiles * num_n_tiles,)
    
    with torch.cuda.device(a.device):
        rwkv_mm_sparsity_kernel[grid](
            a, b, mask, output,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            1,
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
    
    return output


if __name__ == "__main__":
    M, K, N = 256, 128, 256
    BLOCK_M = 64
    
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    mask = torch.ones(M // BLOCK_M, device="cuda", dtype=torch.int32)
    
    result = rwkv_mm_sparsity(a, b, mask)
    
    torch.cuda.synchronize()
    print(f"Output shape: {result.shape}")
    print("rwkv_mm_sparsity completed successfully")
