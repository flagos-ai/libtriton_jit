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
# apply_rotary_pos_emb.py - Rotary Position Embedding Triton Kernel
# NPU-compatible version based on FlagGems Ascend patterns
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def rotary_embedding_rw_kernel(
    state_out,
    state,
    cos,
    sin,
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    num_tokens,
    num_heads,
    token_range,
    head_range,
    dim_range_x,
    dim_range_y,
):
    """Apply rotary embedding to a single set of dimensions."""
    state_x_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_x[None, None, :] * stride_state_d
    )
    state_y_offset = (
        token_range[:, None, None] * stride_state_n
        + head_range[None, :, None] * stride_state_h
        + dim_range_y[None, None, :] * stride_state_d
    )

    cos_sim_offset = (
        token_range[:, None, None] * stride_cos_n
        + dim_range_x[None, None, :] * stride_cos_d
    )
    sin_sim_offset = cos_sim_offset

    state_x = tl.load(
        state + state_x_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )
    state_y = tl.load(
        state + state_y_offset,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
        other=0.0,
    )

    cos_loaded = tl.load(
        cos + cos_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    ).to(tl.float32)
    sin_loaded = tl.load(
        sin + sin_sim_offset,
        mask=token_range[:, None, None] < num_tokens,
        other=0.0,
    ).to(tl.float32)

    out_x = state_x * cos_loaded - state_y * sin_loaded
    out_y = state_x * sin_loaded + state_y * cos_loaded

    tl.store(
        state_out + state_x_offset,
        out_x,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )
    tl.store(
        state_out + state_y_offset,
        out_y,
        mask=(token_range[:, None, None] < num_tokens)
        & (head_range[None, :, None] < num_heads),
    )


@triton.jit
def rotary_embedding_kernel(
    state_out,
    state,
    cos,
    sin,
    stride_state_n,
    stride_state_h,
    stride_state_d,
    stride_cos_n,
    stride_cos_d,
    num_tokens,
    num_heads,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Rotary position embedding kernel (NPU-compatible)."""
    token_index = tl.program_id(0)
    token_range = token_index * BLOCK_N + tl.arange(0, BLOCK_N)
    head_index = tl.program_id(1)
    head_range = head_index * BLOCK_H + tl.arange(0, BLOCK_H)

    # Standard (non-interleaved) rotary embedding
    dim_range_x = tl.arange(0, BLOCK_D // 2)
    dim_range_y = tl.arange(BLOCK_D // 2, BLOCK_D)

    rotary_embedding_rw_kernel(
        state_out,
        state,
        cos,
        sin,
        stride_state_n,
        stride_state_h,
        stride_state_d,
        stride_cos_n,
        stride_cos_d,
        num_tokens,
        num_heads,
        token_range,
        head_range,
        dim_range_x,
        dim_range_y,
    )


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int = None,
) -> tuple:
    """Apply rotary position embeddings to query and key (NPU-compatible).

    Args:
        q: Query tensor [seq_len, num_heads, head_dim] or [batch, seq_len, num_heads, head_dim]
        k: Key tensor [seq_len, num_heads, head_dim] or [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, rotary_dim//2]
        sin: Sine values [seq_len, rotary_dim//2]
        rotary_dim: Dimension to apply rotation (default: head_dim)

    Returns:
        Tuple of rotated (query, key)
    """
    assert k.shape[-1] == q.shape[-1], \
        f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert cos.shape[-1] == sin.shape[-1], \
        f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert cos.shape[-1] * 2 == q.shape[-1], \
        f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"

    q_shape = q.shape
    k_shape = k.shape

    # Reshape to 3D: [num_tokens, num_heads, head_dim]
    q = q.view(-1, q.shape[-2], q.shape[-1]).contiguous()
    k = k.view(-1, k.shape[-2], k.shape[-1]).contiguous()

    num_tokens = q.shape[0]
    num_heads_q = q.shape[1]
    num_heads_k = k.shape[1]
    head_dim = q.shape[-1]

    if rotary_dim is None:
        rotary_dim = head_dim

    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    # Expand cos/sin to match token dimension
    if cos.shape[0] < num_tokens:
        # Tile to match num_tokens
        repeats = (num_tokens + cos.shape[0] - 1) // cos.shape[0]
        cos = cos.repeat(repeats, 1)[:num_tokens]
        sin = sin.repeat(repeats, 1)[:num_tokens]
    elif cos.shape[0] > num_tokens:
        cos = cos[:num_tokens]
        sin = sin[:num_tokens]

    # Add a middle dimension for broadcasting: [num_tokens, 1, rotary_dim//2]
    cos = cos.unsqueeze(1).contiguous()
    sin = sin.unsqueeze(1).contiguous()

    BLOCK_N = 8
    BLOCK_H = 4

    def launch_kernel(state_out, state, num_heads):
        grid = (
            triton.cdiv(num_tokens, BLOCK_N),
            triton.cdiv(num_heads, BLOCK_H),
        )
        rotary_embedding_kernel[grid](
            state_out,
            state,
            cos,
            sin,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            cos.stride(0),
            cos.stride(2),
            num_tokens,
            num_heads,
            BLOCK_N=BLOCK_N,
            BLOCK_H=BLOCK_H,
            BLOCK_D=head_dim,
        )

    try:
        from torch.cuda import device as cuda_device
        with cuda_device(q.device):
            launch_kernel(q_embed, q, num_heads_q)
            launch_kernel(k_embed, k, num_heads_k)
    except:
        launch_kernel(q_embed, q, num_heads_q)
        launch_kernel(k_embed, k, num_heads_k)

    q_embed = q_embed.view(q_shape)
    k_embed = k_embed.view(k_shape)

    return q_embed, k_embed


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
