// ==============================================================================
// rwkv_mm_sparsity_op.cpp - Multi-backend RWKV Sparse Matrix Multiply
// ==============================================================================

#include "rwkv_mm_sparsity_op.h"
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

at::Tensor rwkv_mm_sparsity(const at::Tensor& a, const at::Tensor& b, const at::Tensor& mask) {
    TORCH_CHECK(a.dim() == 2, "A must be 2D");
    TORCH_CHECK(b.dim() == 2, "B must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "Inner dimensions must match");

    int64_t M = a.size(0);
    int64_t K = a.size(1);
    int64_t N = b.size(1);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor mask_contig = mask.contiguous();

#if defined(BACKEND_MUSA)
    void* out_ptr = nullptr;
    size_t bytes = M * N * at::elementSize(a.scalar_type());
    musaMalloc(&out_ptr, bytes);
    auto opts = at::TensorOptions().dtype(a.scalar_type()).device(a.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor output = at::from_blob(out_ptr, {M, N}, deleter, opts);
#else
    at::Tensor output = at::empty({M, N}, a.options());
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("rwkv_mm_sparsity.py"), "rwkv_mm_sparsity_kernel");

    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 4;
    constexpr int num_stages = 2;

    int64_t num_m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    int64_t num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    unsigned int num_blocks = num_m_tiles * num_n_tiles;

    c10::DeviceGuard guard(a.device());
    RawStream stream = get_device_stream(a);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      a_contig, b_contig, mask_contig, output,
      M, N, K,
      a_contig.stride(0), a_contig.stride(1),
      b_contig.stride(0), b_contig.stride(1),
      int64_t(1),
      output.stride(0), output.stride(1),
      BLOCK_M, BLOCK_N, BLOCK_K);

    return output;
}

TORCH_LIBRARY(rwkv_mm_sparsity_ops, m) {
    m.def("rwkv_mm_sparsity(Tensor a, Tensor b, Tensor mask) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(rwkv_mm_sparsity_ops, PrivateUse1, m) {
        m.impl("rwkv_mm_sparsity", TORCH_FN(rwkv_mm_sparsity));
    }
#else
    TORCH_LIBRARY_IMPL(rwkv_mm_sparsity_ops, CUDA, m) {
        m.impl("rwkv_mm_sparsity", TORCH_FN(rwkv_mm_sparsity));
    }
#endif

}  // namespace my_ops
