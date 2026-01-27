// ==============================================================================
// apply_rotary_pos_emb_op.cpp - Multi-backend Rotary Position Embedding
// ==============================================================================

#include "apply_rotary_pos_emb_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#if defined(BACKEND_NPU)
    #if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
        #include "torch_npu/csrc/core/npu/NPUStream.h"
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif
#elif defined(BACKEND_MUSA)
    #include <musa_runtime.h>
#else
    #include "c10/cuda/CUDAStream.h"
#endif

namespace {

#if defined(BACKEND_NPU)
    using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
    using RawStream = musaStream_t;
#else
    using RawStream = CUstream;
#endif

inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& tensor) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif
#elif defined(BACKEND_MUSA)
    return nullptr;
#else
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

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

    int64_t seq_len = q.size(0);
    int64_t num_heads = q.size(1);
    int64_t head_dim = q.size(2);

    if (rotary_dim <= 0) {
        rotary_dim = head_dim;
    }

    at::Tensor q_contig = q.contiguous();
    at::Tensor k_contig = k.contiguous();
    at::Tensor cos_contig = cos.contiguous();
    at::Tensor sin_contig = sin.contiguous();

#if defined(BACKEND_MUSA)
    void* q_out_ptr = nullptr;
    void* k_out_ptr = nullptr;
    size_t bytes = seq_len * num_heads * head_dim * at::elementSize(q.scalar_type());
    musaMalloc(&q_out_ptr, bytes);
    musaMalloc(&k_out_ptr, bytes);
    auto opts = at::TensorOptions().dtype(q.scalar_type()).device(q.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor q_out = at::from_blob(q_out_ptr, {seq_len, num_heads, head_dim}, deleter, opts);
    at::Tensor k_out = at::from_blob(k_out_ptr, {seq_len, num_heads, head_dim}, deleter, opts);
#else
    at::Tensor q_out = at::empty_like(q);
    at::Tensor k_out = at::empty_like(k);
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("apply_rotary_pos_emb.py"), "apply_rotary_pos_emb_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < head_dim) BLOCK_SIZE *= 2;

    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(q.device());
    RawStream stream = get_device_stream(q);

    f(stream, seq_len, num_heads, 1, num_warps, num_stages,
      q_contig, k_contig, cos_contig, sin_contig, q_out, k_out,
      seq_len, num_heads, head_dim, rotary_dim,
      q_contig.stride(0), q_contig.stride(1),
      k_contig.stride(0), k_contig.stride(1),
      cos_contig.stride(0),
      BLOCK_SIZE);

    return std::make_tuple(q_out, k_out);
}

TORCH_LIBRARY(apply_rotary_pos_emb_ops, m) {
    m.def("apply_rotary_pos_emb(Tensor q, Tensor k, Tensor cos, Tensor sin, int rotary_dim) -> (Tensor, Tensor)");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(apply_rotary_pos_emb_ops, PrivateUse1, m) {
        m.impl("apply_rotary_pos_emb", TORCH_FN(apply_rotary_pos_emb));
    }
#else
    TORCH_LIBRARY_IMPL(apply_rotary_pos_emb_ops, CUDA, m) {
        m.impl("apply_rotary_pos_emb", TORCH_FN(apply_rotary_pos_emb));
    }
#endif

}  // namespace my_ops
