# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================
# rwkv_ka_fusion.py - RWKV Key-Attention Fusion Triton Kernel (NPU-compatible)
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def rwkv_ka_fusion_kernel(
    k_ptr,
    a_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    k_stride_batch,
    k_stride_seq,
    a_stride_batch,
    a_stride_seq,
    out_stride_batch,
    out_stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """RWKV key-attention fusion kernel.

    Fuses key transformation with attention computation for RWKV model.
    Simple element-wise multiply - should be NPU-compatible.
    """
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    offsets = tl.arange(0, BLOCK_SIZE)

    # Load key and attention
    k_base = k_ptr + batch_idx * k_stride_batch + seq_idx * k_stride_seq
    a_base = a_ptr + batch_idx * a_stride_batch + seq_idx * a_stride_seq

    # Process in chunks
    for off in range(0, hidden_dim, BLOCK_SIZE):
        idx = off + offsets
        mask = idx < hidden_dim

        k = tl.load(k_base + idx, mask=mask, other=0.0)
        a = tl.load(a_base + idx, mask=mask, other=0.0)

        # Fused operation: element-wise multiply
        output = k * a

        # Store
        out_base = output_ptr + batch_idx * out_stride_batch + seq_idx * out_stride_seq
        tl.store(out_base + idx, output, mask=mask)


@triton.jit
def rwkv_ka_fusion_kernel_swapped(
    k_ptr,
    a_ptr,
    output_ptr,
    batch_size,
    seq_len,
    hidden_dim,
    k_stride_batch,
    k_stride_seq,
    a_stride_batch,
    a_stride_seq,
    out_stride_batch,
    out_stride_seq,
    BLOCK_SIZE: tl.constexpr,
):
    """Swapped grid version: program_id(0)=seq, program_id(1)=batch for GCU grid.y limit."""
    seq_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    offsets = tl.arange(0, BLOCK_SIZE)

    k_base = k_ptr + batch_idx * k_stride_batch + seq_idx * k_stride_seq
    a_base = a_ptr + batch_idx * a_stride_batch + seq_idx * a_stride_seq

    for off in range(0, hidden_dim, BLOCK_SIZE):
        idx = off + offsets
        mask = idx < hidden_dim

        k = tl.load(k_base + idx, mask=mask, other=0.0)
        a = tl.load(a_base + idx, mask=mask, other=0.0)

        output = k * a

        out_base = output_ptr + batch_idx * out_stride_batch + seq_idx * out_stride_seq
        tl.store(out_base + idx, output, mask=mask)


def rwkv_ka_fusion(k: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """RWKV key-attention fusion (NPU-compatible).

    Args:
        k: Key tensor [batch_size, seq_len, hidden_dim]
        a: Attention tensor [batch_size, seq_len, hidden_dim]

    Returns:
        Fused output tensor
    """
    assert k.shape == a.shape, "K and A must have same shape"

    batch_size, seq_len, hidden_dim = k.shape

    k = k.contiguous()
    a = a.contiguous()
    output = torch.empty_like(k)

    # Use smaller block size for NPU compatibility
    BLOCK_SIZE = min(triton.next_power_of_2(hidden_dim), 1024)

    grid = (batch_size, seq_len)

    try:
        from torch.cuda import device as cuda_device
        with cuda_device(k.device):
            rwkv_ka_fusion_kernel[grid](
                k, a, output,
                batch_size, seq_len, hidden_dim,
                k.stride(0), k.stride(1),
                a.stride(0), a.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
            )
    except:
        rwkv_ka_fusion_kernel[grid](
            k, a, output,
            batch_size, seq_len, hidden_dim,
            k.stride(0), k.stride(1),
            a.stride(0), a.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

    return output


if __name__ == "__main__":
    batch_size, seq_len, hidden_dim = 8, 128, 256

    k = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")
    a = torch.randn(batch_size, seq_len, hidden_dim, device="cuda")

    result = rwkv_ka_fusion(k, a)

    # Verify against element-wise multiply
    expected = k * a

    torch.cuda.synchronize()
    print(f"Output shape: {result.shape}")
    print(f"Match: {torch.allclose(result, expected)}")
    print("rwkv_ka_fusion completed successfully")
