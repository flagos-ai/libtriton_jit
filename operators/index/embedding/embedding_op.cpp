// ==============================================================================
// embedding_op.cpp - Multi-backend Triton JIT Embedding Lookup
// ==============================================================================

#include "embedding_op.h"
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

at::Tensor embedding(const at::Tensor& indices, const at::Tensor& weight) {
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D [num_embeddings, embedding_dim]");

    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);

    at::Tensor indices_flat = indices.view(-1).contiguous();
    int64_t num_indices = indices_flat.numel();

    auto orig_shape = indices.sizes().vec();
    orig_shape.push_back(embedding_dim);

#if defined(BACKEND_MUSA)
    void* out_ptr = nullptr;
    size_t out_bytes = num_indices * embedding_dim * at::elementSize(weight.scalar_type());
    musaMalloc(&out_ptr, out_bytes);
    auto opts = at::TensorOptions().dtype(weight.scalar_type()).device(weight.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor output = at::from_blob(out_ptr, {num_indices, embedding_dim}, deleter, opts);
#else
    at::Tensor output = at::empty({num_indices, embedding_dim}, weight.options());
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("embedding.py"), "embedding_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < embedding_dim) BLOCK_SIZE *= 2;

    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(weight.device());
    RawStream stream = get_device_stream(weight);

    f(stream, num_indices, 1, 1, num_warps, num_stages,
      indices_flat, weight, output,
      num_embeddings, embedding_dim,
      int64_t(1), weight.stride(0), output.stride(0),
      BLOCK_SIZE);

    return output.view(orig_shape);
}

TORCH_LIBRARY(embedding_ops, m) {
    m.def("embedding(Tensor indices, Tensor weight) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(embedding_ops, PrivateUse1, m) {
        m.impl("embedding", TORCH_FN(embedding));
    }
#else
    TORCH_LIBRARY_IMPL(embedding_ops, CUDA, m) {
        m.impl("embedding", TORCH_FN(embedding));
    }
#endif

}  // namespace my_ops
