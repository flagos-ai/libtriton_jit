import torch
import triton
from triton import language as tl


@triton.jit
def sum_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    # Map the program id to the row of inp it should compute.
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_ids < M

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=cdtype)
    for off in tl.range(0, N, BLOCK_N, STAGE):
        col_ids = off + tl.arange(0, BLOCK_N)
        col_mask = col_ids < N
        mask = row_mask[:, None] & col_mask[None, :]

        a = tl.load(in_ptr + row_ids[:, None] * N + col_ids, mask, other=0).to(cdtype)
        acc += a
    out = tl.sum(acc, axis=1)
    tl.store(out_ptr + row_ids, out, row_mask)


# ------ wrapper ------
_integer_dtypes = {
    torch.bool,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def dim_compress(inp, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    stride = inp.stride()
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: stride[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return inp.permute(order).contiguous()


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    if dtype is None:
        dtype = inp.dtype
        if dtype in _integer_dtypes:
            dtype = torch.int64

    if not dim:  # [] or
        raise ValueError("Cannot sum over an empty list of dimensions")

    shape = list(inp.shape)
    dim = [d % inp.ndim for d in dim]
    inp = dim_compress(inp, dim)  # move reduction dim to the end and make it contiguous
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = inp.numel() // N

    out = torch.empty(shape, dtype=dtype, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    with torch.cuda.device(inp.device):
        sum_kernel[grid](inp, out, M, N, BLOCK_M=4, BLOCK_N=512, STAGE=2, num_warps=8)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


if __name__ == "__main__":
    torch_ops_my_ops_sum_dim = torch.library.custom_op(
        "my_ops::sum.dim_IntList",
        mutates_args=(),
        device_types="cuda",
        # the scheme should not include op name
        schema="(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
    )(sum_dim)
    x = torch.randn(16, 4 * 1024, device="cuda")
    result1 = sum_dim(x, [1])
    result2 = torch.sum(x, [1])
    result3 = torch.ops.my_ops.sum_dim_IntList(x, [1])

    torch.cuda.synchronize()
    for _ in range(10):
        torch.sum(x, [1])
    torch.cuda.synchronize()
    for _ in range(10):
        sum_dim(x, [1])
    torch.cuda.synchronize()
    for _ in range(10):
        torch.ops.my_ops.sum_dim_IntList(x, [1])
    torch.cuda.synchronize()
