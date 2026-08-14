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

import torch
import triton
from triton import language as tl


@triton.jit
def zeros_kernel(Out, n, BLOCK_N: tl.constexpr):
    """Fill tensor with zeros."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    tl.store(Out + offsets, tl.zeros([BLOCK_N], dtype=Out.dtype.element_ty), mask=mask)


def zeros_like(x: torch.Tensor) -> torch.Tensor:
    """Create zero tensor with same shape and dtype as input."""
    out = torch.empty_like(x)
    n = out.numel()
    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N), 1, 1)
    with torch.cuda.device(x.device):
        zeros_kernel[grid](out, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1)
    return out


if __name__ == "__main__":
    N = 128 * 1024
    x = torch.randn(N, device="cuda")

    result = zeros_like(x)
    expected = torch.zeros_like(x)

    torch.cuda.synchronize()
    print(f"Result[0:5]: {result[0:5]}")
    print(f"Expected[0:5]: {expected[0:5]}")
    print(f"Match: {torch.allclose(result, expected)}")
