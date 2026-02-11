// ==============================================================================
// addmm_op.cpp - Multi-backend Triton JIT Additive Matrix Multiplication
// C = beta * input + alpha * (A @ B)
// ==============================================================================

#include "addmm_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "operators/common/kernel_config.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor addmm(const at::Tensor& input, const at::Tensor& a, const at::Tensor& b,
                 const at::Scalar& beta, const at::Scalar& alpha) {
    TORCH_CHECK(a.dim() == 2, "Expected 2D tensor for a");
    TORCH_CHECK(b.dim() == 2, "Expected 2D tensor for b");
    TORCH_CHECK(a.size(1) == b.size(0), "Matrix dimensions must match");

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    at::Tensor input_expanded = input.expand({M, N}).contiguous();
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();

    float beta_val = beta.toFloat();
    float alpha_val = alpha.toFloat();

    at::Tensor c = triton_jit::ops::backend_empty({M, N}, a.scalar_type(), a.device());

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("addmm.py"), "addmm_kernel");

    constexpr auto cfg = triton_jit::ops::default_matmul_config();

    int64_t num_m_tiles = (M + cfg.BLOCK_M - 1) / cfg.BLOCK_M;
    int64_t num_n_tiles = (N + cfg.BLOCK_N - 1) / cfg.BLOCK_N;
    unsigned int num_blocks = num_m_tiles * num_n_tiles;

    c10::DeviceGuard guard(a.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(a);

    f(stream, num_blocks, 1, 1, cfg.num_warps, cfg.num_stages,
      input_expanded, a_contig, b_contig, c,
      M, N, K,
      alpha_val, beta_val,
      input_expanded.stride(0), input_expanded.stride(1),
      a_contig.stride(0), a_contig.stride(1),
      b_contig.stride(0), b_contig.stride(1),
      c.stride(0), c.stride(1),
      cfg.BLOCK_M, cfg.BLOCK_N, cfg.BLOCK_K);

    return c;
}

TORCH_LIBRARY(addmm_ops, m) {
    m.def("addmm(Tensor input, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor");
}

REGISTER_TRITON_OP(addmm_ops, "addmm", addmm)

}  // namespace my_ops
