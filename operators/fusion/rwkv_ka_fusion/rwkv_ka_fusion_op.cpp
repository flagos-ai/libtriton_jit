// ==============================================================================
// rwkv_ka_fusion_op.cpp - Multi-backend RWKV Key-Attention Fusion
// ==============================================================================

#include "rwkv_ka_fusion_op.h"
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

at::Tensor rwkv_ka_fusion(const at::Tensor& k, const at::Tensor& a) {
    TORCH_CHECK(k.sizes() == a.sizes(), "K and A must have same shape");
    TORCH_CHECK(k.dim() == 3, "Expected 3D tensors [batch, seq, hidden]");

    int64_t batch_size = k.size(0);
    int64_t seq_len = k.size(1);
    int64_t hidden_dim = k.size(2);

    at::Tensor k_contig = k.contiguous();
    at::Tensor a_contig = a.contiguous();

#if defined(BACKEND_MUSA)
    void* out_ptr = nullptr;
    size_t bytes = batch_size * seq_len * hidden_dim * at::elementSize(k.scalar_type());
    musaMalloc(&out_ptr, bytes);
    auto opts = at::TensorOptions().dtype(k.scalar_type()).device(k.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor output = at::from_blob(out_ptr, {batch_size, seq_len, hidden_dim}, deleter, opts);
#else
    at::Tensor output = at::empty_like(k);
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("rwkv_ka_fusion.py"), "rwkv_ka_fusion_kernel");

    int64_t BLOCK_SIZE = 1;
    while (BLOCK_SIZE < hidden_dim) BLOCK_SIZE *= 2;

    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(k.device());
    RawStream stream = get_device_stream(k);

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

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(rwkv_ka_fusion_ops, PrivateUse1, m) {
        m.impl("rwkv_ka_fusion", TORCH_FN(rwkv_ka_fusion));
    }
#else
    TORCH_LIBRARY_IMPL(rwkv_ka_fusion_ops, CUDA, m) {
        m.impl("rwkv_ka_fusion", TORCH_FN(rwkv_ka_fusion));
    }
#endif

}  // namespace my_ops
