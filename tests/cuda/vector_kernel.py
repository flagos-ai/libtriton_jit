import triton
import triton.language as tl


@triton.jit
def vector_kernel(x, y, n, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + index, tl.load(x + index, index < n, other=0) * 2, index < n)


@triton.jit
def layout_kernel(
    x, y, M: tl.constexpr, N: tl.constexpr, sx0, sx1, sy0, sy1, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // N
    col = index % N
    value = tl.load(x + row * sx0 + col * sx1, row < M, other=0)
    tl.store(y + row * sy0 + col * sy1, value * 2, row < M)
