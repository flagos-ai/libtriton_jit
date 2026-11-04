import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
# Initialize Torch and the Ascend extension before Triton discovers plugins.
import torch  # noqa: F401, E402
import torch_npu  # noqa: F401, E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def gemv_kernel(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    matrix = tl.load(
        A + rows[:, None] * N + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
        other=0,
    )
    vector = tl.load(X + cols, mask=cols < N, other=0)
    result = tl.sum(matrix.to(tl.float32) * vector[None, :].to(tl.float32), 1)
    tl.store(Y + rows, result, mask=rows < M)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    ],
    key=["M", "N"],
)
@triton.jit
def native_gemv(
    A,
    X,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    gemv_kernel(A, X, Y, M, N, BLOCK_M, BLOCK_N)


@triton.jit
def counter_kernel(Counts, N: tl.constexpr):
    pid = tl.program_id(0)
    tl.atomic_add(Counts + pid, 1, mask=pid < N, sem="relaxed")
