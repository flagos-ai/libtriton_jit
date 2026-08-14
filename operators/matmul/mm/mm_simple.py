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
def matmul_simple_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple matmul without tl.dot - use element-wise operations instead.
    This is slow but helps debug if tl.dot is the issue.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Each program computes one element of C
    if pid_m < M and pid_n < N:
        # Compute C[pid_m, pid_n] = sum(A[pid_m, :] * B[:, pid_n])
        acc = tl.zeros((1,), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE):
            k_idx = k + tl.arange(0, BLOCK_SIZE)
            mask = k_idx < K

            a_val = tl.load(A + pid_m * stride_am + k_idx * stride_ak, mask=mask, other=0.0)
            b_val = tl.load(B + k_idx * stride_bk + pid_n * stride_bn, mask=mask, other=0.0)

            acc += tl.sum(a_val * b_val)

        tl.store(C + pid_m * stride_cm + pid_n * stride_cn, acc)
