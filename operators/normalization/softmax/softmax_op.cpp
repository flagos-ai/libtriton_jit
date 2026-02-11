// ==============================================================================
// softmax_op.cpp - Multi-backend Triton JIT Softmax
// ==============================================================================

#include "softmax_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include "ATen/WrapDimUtils.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "operators/common/kernel_config.h"

namespace {

int64_t next_power_of_2(int64_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

}  // anonymous namespace

namespace my_ops {
using namespace triton_jit;

at::Tensor softmax(const at::Tensor& input, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, input.dim());

    // Permute to put softmax dim last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < input.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    at::Tensor x_permuted = input.permute(perm).contiguous();

    int64_t n_rows = x_permuted.numel() / x_permuted.size(-1);
    int64_t n_cols = x_permuted.size(-1);
    at::Tensor x_flat = x_permuted.view({n_rows, n_cols});

    at::Tensor output = triton_jit::ops::backend_empty(
        {n_rows, n_cols}, input.scalar_type(), input.device());

    // softmax_kernel_inner(output_ptr, input_ptr, M, N, input_row_stride, output_row_stride, TILE_N, ONE_TILE_PER_CTA)
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("softmax.py"), "softmax_kernel_inner");

    constexpr auto cfg = triton_jit::ops::default_softmax_config();
    int64_t TILE_N = std::min(next_power_of_2(n_cols), cfg.max_tile_n);
    int64_t ONE_TILE_PER_CTA = (TILE_N >= n_cols) ? 1 : 0;

    c10::DeviceGuard guard(input.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(x_flat);

    f(stream, n_rows, 1, 1, cfg.num_warps, cfg.num_stages,
      output, x_flat,  // output first, then input
      n_rows, n_cols,
      x_flat.stride(0), output.stride(0),
      TILE_N, ONE_TILE_PER_CTA);

    // Reshape and inverse permute
    output = output.view(x_permuted.sizes());

    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    output = output.permute(inv_perm).contiguous();

    return output;
}

TORCH_LIBRARY(softmax_ops, m) {
    m.def("softmax(Tensor self, int dim) -> Tensor");
}

REGISTER_TRITON_OP(softmax_ops, "softmax", softmax)

}  // namespace my_ops
