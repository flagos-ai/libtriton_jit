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
# fill_.py - In-place Fill Triton Kernel (Placeholder)
# ==============================================================================

import torch
import triton
from triton import language as tl


@triton.jit
def fill_kernel(
    output_ptr,
    value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place fill kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    tl.store(output_ptr + offsets, value, mask=mask)


def fill_(tensor: torch.Tensor, value: float) -> torch.Tensor:
    """Fill tensor in-place with a scalar value.
    
    Args:
        tensor: Tensor to fill (modified in-place)
        value: Scalar value to fill with
    
    Returns:
        The input tensor (modified in-place)
    """
    n_elements = tensor.numel()
    
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    grid = (num_blocks,)
    with torch.cuda.device(tensor.device):
        fill_kernel[grid](
            tensor,
            value,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    
    return tensor


if __name__ == "__main__":
    x = torch.empty(1024, device="cuda")
    
    fill_(x, 3.14)
    
    torch.cuda.synchronize()
    print(f"Filled values (sample): {x[:5]}")
    print(f"All equal to 3.14: {torch.allclose(x, torch.full_like(x, 3.14))}")
