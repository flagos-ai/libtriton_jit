// ==============================================================================
// max_op.cpp - Multi-backend Triton JIT Max Reduction Operation
// ==============================================================================

#include "max_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include "ATen/WrapDimUtils.h"

#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> max_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    // Wrap dimension
    dim = at::maybe_wrap_dim(dim, self.dim());

    // Permute to put reduction dim last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    at::Tensor permuted = self.permute(perm).contiguous();
    int64_t M = permuted.numel() / permuted.size(-1);
    int64_t N = permuted.size(-1);

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i != dim) out_shape.push_back(self.size(i));
    }

    at::Tensor out_vals = triton_jit::ops::backend_empty({M}, self.scalar_type(), self.device());
    at::Tensor out_idx = triton_jit::ops::backend_empty({M}, at::kLong, self.device());

    // Launch kernel
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("max.py"), "max_with_indices_kernel");

    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 512;
    constexpr int num_warps = 8;
    constexpr int num_stages = 1;
    const unsigned int num_blocks = (M + BLOCK_M - 1) / BLOCK_M;

    c10::DeviceGuard guard(self.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(permuted);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      permuted.view({M, N}), out_vals, out_idx,
      M, N, BLOCK_M, BLOCK_N);

    // Reshape output - out_shape is already in correct order (original dims except reduced)
    if (!out_shape.empty()) {
        out_vals = out_vals.view(out_shape);
        out_idx = out_idx.view(out_shape);
    }

    if (keepdim) {
        out_vals = out_vals.unsqueeze(dim);
        out_idx = out_idx.unsqueeze(dim);
    }

    return {out_vals, out_idx};
}

at::Tensor max(const at::Tensor& self) {
    auto [vals, _] = max_dim(self.flatten(), 0, false);
    return vals;
}

TORCH_LIBRARY(max_ops, m) {
    m.def("max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)");
    m.def("max(Tensor self) -> Tensor");
}

REGISTER_TRITON_OP(max_ops, "max.dim", max_dim)
REGISTER_TRITON_OP(max_ops, "max", max)

}  // namespace my_ops
