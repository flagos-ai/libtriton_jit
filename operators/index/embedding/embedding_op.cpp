// ==============================================================================
// embedding_op.cpp - Multi-backend Triton JIT Embedding Lookup
// ==============================================================================

#include "embedding_op.h"
#include "operators/common/backend_ops.h"
#include "operators/common/kernel_config.h"
#include "operators/common/op_registration.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor embedding(const at::Tensor& indices, const at::Tensor& weight) {
  TORCH_CHECK(weight.dim() == 2, "Weight must be 2D [num_embeddings, embedding_dim]");

  int64_t num_embeddings = weight.size(0);
  int64_t embedding_dim = weight.size(1);

  at::Tensor indices_flat = indices.view(-1).contiguous();
  int64_t num_indices = indices_flat.numel();

  auto orig_shape = indices.sizes().vec();
  orig_shape.push_back(embedding_dim);

  at::Tensor output =
      triton_jit::ops::backend_empty({num_indices, embedding_dim}, weight.scalar_type(), weight.device());

  const TritonJITFunction& f =
      TritonJITFunction::get_instance(std::string("embedding.py"), "embedding_kernel");

  int64_t BLOCK_SIZE = 1;
  while (BLOCK_SIZE < embedding_dim) BLOCK_SIZE *= 2;

  constexpr auto cfg = triton_jit::ops::default_basic_config();

  c10::DeviceGuard guard(weight.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(weight);

  f(stream,
    num_indices,
    1,
    1,
    cfg.num_warps,
    cfg.num_stages,
    indices_flat,
    weight,
    output,
    num_embeddings,
    embedding_dim,
    int64_t(1),
    weight.stride(0),
    output.stride(0),
    BLOCK_SIZE);

  return output.view(orig_shape);
}

TORCH_LIBRARY(embedding_ops, m) {
  m.def("embedding(Tensor indices, Tensor weight) -> Tensor");
}

REGISTER_TRITON_OP(embedding_ops, "embedding", embedding)

}  // namespace my_ops
