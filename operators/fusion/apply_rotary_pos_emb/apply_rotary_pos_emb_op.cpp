// ==============================================================================
// apply_rotary_pos_emb_op.cpp - Multi-backend Rotary Position Embedding
// ==============================================================================

#include "apply_rotary_pos_emb_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "operators/common/kernel_config.h"

namespace my_ops {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t rotary_dim) {

    TORCH_CHECK(q.dim() == 3, "Query must be 3D [seq_len, num_heads, head_dim]");
    TORCH_CHECK(k.dim() == 3, "Key must be 3D");

    int64_t num_tokens = q.size(0);
    int64_t num_heads_q = q.size(1);
    int64_t num_heads_k = k.size(1);
    int64_t head_dim = q.size(2);

    if (rotary_dim <= 0) {
        rotary_dim = head_dim;
    }

    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();

    // cos/sin shape: [seq_len, rotary_dim//2] -> expand to [seq_len, 1, rotary_dim//2]
    at::Tensor cos_expanded = cos.unsqueeze(1).contiguous();
    at::Tensor sin_expanded = sin.unsqueeze(1).contiguous();

    at::Tensor q_out = triton_jit::ops::backend_empty(
        {num_tokens, num_heads_q, head_dim}, q.scalar_type(), q.device());
    at::Tensor k_out = triton_jit::ops::backend_empty(
        {num_tokens, num_heads_k, head_dim}, k.scalar_type(), k.device());

    // rotary_embedding_kernel(state_out, state, cos, sin,
    //     stride_state_n, stride_state_h, stride_state_d,
    //     stride_cos_n, stride_cos_d,
    //     num_tokens, num_heads,
    //     BLOCK_N, BLOCK_H, BLOCK_D)
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("apply_rotary_pos_emb.py"), "rotary_embedding_kernel");

    constexpr auto cfg = triton_jit::ops::default_rotary_config();

    c10::DeviceGuard guard(q.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(q);

    // Grid: (cdiv(num_tokens, BLOCK_N), cdiv(num_heads, BLOCK_H))
    auto grid_q_x = (num_tokens + cfg.BLOCK_N - 1) / cfg.BLOCK_N;
    auto grid_q_y = (num_heads_q + cfg.BLOCK_H - 1) / cfg.BLOCK_H;
    auto grid_k_y = (num_heads_k + cfg.BLOCK_H - 1) / cfg.BLOCK_H;

    // Launch for Q
    f(stream, grid_q_x, grid_q_y, 1, cfg.num_warps, cfg.num_stages,
      q_out, q_contig, cos_expanded, sin_expanded,
      q_contig.stride(0), q_contig.stride(1), q_contig.stride(2),
      cos_expanded.stride(0), cos_expanded.stride(2),
      num_tokens, num_heads_q,
      cfg.BLOCK_N, cfg.BLOCK_H, head_dim);

    // Launch for K
    f(stream, grid_q_x, grid_k_y, 1, cfg.num_warps, cfg.num_stages,
      k_out, k_contig, cos_expanded, sin_expanded,
      k_contig.stride(0), k_contig.stride(1), k_contig.stride(2),
      cos_expanded.stride(0), cos_expanded.stride(2),
      num_tokens, num_heads_k,
      cfg.BLOCK_N, cfg.BLOCK_H, head_dim);

    return std::make_tuple(q_out, k_out);
}

TORCH_LIBRARY(apply_rotary_pos_emb_ops, m) {
    m.def("apply_rotary_pos_emb(Tensor q, Tensor k, Tensor cos, Tensor sin, int rotary_dim) -> (Tensor, Tensor)");
}

REGISTER_TRITON_OP(apply_rotary_pos_emb_ops, "apply_rotary_pos_emb", apply_rotary_pos_emb)

}  // namespace my_ops
