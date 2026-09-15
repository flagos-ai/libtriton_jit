"""Plain @triton.autotune kernels with short trial benchmarks."""

import inspect

import triton
import triton.language as tl
import triton.testing


def event_bench(fn, quantiles):
    return triton.testing.do_bench(fn, warmup=5, rep=12, quantiles=quantiles)


def bench_options():
    if "do_bench" in inspect.signature(triton.autotune).parameters:
        return {"do_bench": event_bench}
    return {"warmup": 5, "rep": 12}


def configs():
    return [
        triton.Config({"BLOCK": block}, num_warps=warps, num_stages=1)
        for block, warps in [(128, 4), (256, 4), (512, 8)]
    ]


@triton.autotune(configs=configs(), key=["n"], **bench_options())
@triton.jit
def native_vector_kernel(x, y, n, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + index, tl.load(x + index, index < n, other=0) * 2, index < n)


@triton.autotune(configs=configs(), key=["n"], reset_to_zero=["y"], **bench_options())
@triton.jit
def native_accumulate_kernel(x, y, n, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.atomic_add(y + index, tl.load(x + index, index < n, other=0), index < n)


@triton.autotune(configs=configs(), key=["n"], **bench_options())
@triton.jit
def native_cast_kernel(x, y, n, DTYPE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < n, other=0) * 2
    tl.store(y + index, value.to(DTYPE), index < n)


@triton.autotune(configs=configs(), key=["SCALE"], **bench_options())
@triton.jit
def native_float_key_kernel(x, y, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y + index, tl.load(x + index, index < n, other=0) * SCALE, index < n)
