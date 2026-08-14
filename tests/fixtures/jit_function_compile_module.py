import triton
import triton.language as tl


@triton.jit
def mul_func(x, y):
    return x * y


@triton.jit
def apply_jitfunction_kernel(
    output,
    operation: tl.constexpr,
    n_elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    values = operation(offsets.to(tl.float32), 2.0)
    tl.store(output + offsets, values, mask=offsets < n_elements)
