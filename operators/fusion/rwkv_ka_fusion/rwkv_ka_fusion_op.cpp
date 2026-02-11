// ==============================================================================
// rwkv_ka_fusion_op.cpp - Multi-backend RWKV Key-Attention Fusion
// ==============================================================================

#include "rwkv_ka_fusion_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"
#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor rwkv_ka_fusion(const at::Tensor& k, const at::Tensor& a) {
    TORCH_CHECK(k.sizes() == a.sizes(), "K and A must have same shape");
    TORCH_CHECK(k.dim() == 3, "Expected 3D tensors [batch, seq, hidden]");

    int64_t batch_size = k.size(0);
    int64_t seq_len = k.size(1);
    int64_t hidden_dim = k.size(2);

    at::Tensor k_contig = k.contiguous();
    at::Tensor a_contig = a.contiguous();

    at::Tensor output = triton_jit::ops::backend_empty(
        {batch_size, seq_len, hidden_dim}, k.scalar_type(), k.device());

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("rwkv_ka_fusion.py"), "rwkv_ka_fusion_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < hidden_dim) BLOCK_SIZE *= 2;

    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(k.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(k);

    f(stream, batch_size, seq_len, 1, num_warps, num_stages,
      k_contig, a_contig, output,
      batch_size, seq_len, hidden_dim,
      k_contig.stride(0), k_contig.stride(1),
      a_contig.stride(0), a_contig.stride(1),
      output.stride(0), output.stride(1),
      BLOCK_SIZE);

    return output;
}

TORCH_LIBRARY(rwkv_ka_fusion_ops, m) {
    m.def("rwkv_ka_fusion(Tensor k, Tensor a) -> Tensor");
}

REGISTER_TRITON_OP(rwkv_ka_fusion_ops, "rwkv_ka_fusion", rwkv_ka_fusion)

}  // namespace my_ops
