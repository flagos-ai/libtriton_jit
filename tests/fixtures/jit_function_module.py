import triton
import triton.language as tl


@triton.jit
def mul_func(x, y):
    return x * y


not_a_jit_function = 7
