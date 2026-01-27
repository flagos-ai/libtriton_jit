# ==============================================================================
# fused_add_rms_norm.py - Fused Add + RMS Normalization Triton Kernel
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def fused_add_rms_norm_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_out_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + RMS normalization kernel.
    
    output = rms_norm(input + residual)
    residual_out = input + residual
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input and residual
    input_row_ptr = input_ptr + row_idx * input_row_stride
    residual_row_ptr = residual_ptr + row_idx * input_row_stride
    
    x = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)
    res = tl.load(residual_row_ptr + col_offsets, mask=mask, other=0.0)
    
    # Fused add
    x_add = x + res
    
    # Store residual output
    residual_out_row_ptr = residual_out_ptr + row_idx * output_row_stride
    tl.store(residual_out_row_ptr + col_offsets, x_add, mask=mask)
    
    # Compute RMS
    x_sq = x_add * x_add
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rms = tl.sqrt(mean_sq + eps)
    
    # Normalize
    x_normed = x_add / rms
    
    # Load weight and apply
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = x_normed * w
    
    # Store normalized output
    output_row_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_ptr + col_offsets, output, mask=mask)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, 
                       weight: torch.Tensor, eps: float = 1e-6):
    """Fused Add + RMS Normalization.
    
    Returns:
        output: rms_norm(x + residual)
        residual_out: x + residual
    """
    assert x.shape == residual.shape, "Input and residual must have same shape"
    assert x.size(-1) == weight.size(0), "Hidden dim must match weight size"
    
    orig_shape = x.shape
    x_flat = x.view(-1, x.size(-1)).contiguous()
    res_flat = residual.view(-1, residual.size(-1)).contiguous()
    n_rows, n_cols = x_flat.shape
    
    output = torch.empty_like(x_flat)
    residual_out = torch.empty_like(x_flat)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    grid = (n_rows,)
    with torch.cuda.device(x.device):
        fused_add_rms_norm_kernel[grid](
            x_flat, res_flat, weight, output, residual_out,
            x_flat.stride(0), output.stride(0),
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    
    return output.view(orig_shape), residual_out.view(orig_shape)


if __name__ == "__main__":
    x = torch.randn(16, 1024, device="cuda")
    residual = torch.randn(16, 1024, device="cuda")
    weight = torch.ones(1024, device="cuda")
    
    output, res_out = fused_add_rms_norm(x, residual, weight)
    
    # Reference implementation
    x_add = x + residual
    rms = torch.sqrt(x_add.pow(2).mean(-1, keepdim=True) + 1e-6)
    expected = (x_add / rms) * weight
    
    torch.cuda.synchronize()
    print(f"Output max diff: {(output - expected).abs().max().item()}")
    print(f"Residual max diff: {(res_out - x_add).abs().max().item()}")
    print(f"Match: {torch.allclose(output, expected, rtol=1e-4, atol=1e-4)}")
