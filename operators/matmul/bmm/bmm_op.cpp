// ==============================================================================
// bmm_op.cpp - Multi-backend Triton JIT Batched Matrix Multiplication
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "bmm_op.h"
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

at::Tensor bmm(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 3, "Expected 3D tensor for a");
    TORCH_CHECK(b.dim() == 3, "Expected 3D tensor for b");
    TORCH_CHECK(a.size(0) == b.size(0), "Batch sizes must match");
    TORCH_CHECK(a.size(2) == b.size(1), "Inner dimensions must match");

    int64_t B = a.size(0);
    int64_t M = a.size(1);
    int64_t K = a.size(2);
    int64_t N = b.size(2);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();

#if defined(BACKEND_MUSA)
    void* c_ptr = nullptr;
    size_t c_bytes = B * M * N * at::elementSize(a.scalar_type());
    musaMalloc(&c_ptr, c_bytes);
    auto opts = at::TensorOptions().dtype(a.scalar_type()).device(a.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor c = at::from_blob(c_ptr, {B, M, N}, deleter, opts);
#else
    at::Tensor c = at::empty({B, M, N}, a.options());
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("bmm.py"), "bmm_kernel");

    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 4;
    constexpr int num_stages = 2;

    int64_t num_m_tiles = (M + BLOCK_M - 1) / BLOCK_M;
    int64_t num_n_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    unsigned int grid_x = num_m_tiles * num_n_tiles;
    unsigned int grid_y = B;

    c10::DeviceGuard guard(a.device());
    RawStream stream = get_device_stream(a);

    f(stream, grid_x, grid_y, 1, num_warps, num_stages,
      a_contig, b_contig, c,
      B, M, N, K,
      a_contig.stride(0), a_contig.stride(1), a_contig.stride(2),
      b_contig.stride(0), b_contig.stride(1), b_contig.stride(2),
      c.stride(0), c.stride(1), c.stride(2),
      BLOCK_M, BLOCK_N, BLOCK_K);

    return c;
}

TORCH_LIBRARY(bmm_ops, m) {
    m.def("bmm(Tensor self, Tensor other) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(bmm_ops, PrivateUse1, m) {
        m.impl("bmm", TORCH_FN(bmm));
    }
#else
    TORCH_LIBRARY_IMPL(bmm_ops, CUDA, m) {
        m.impl("bmm", TORCH_FN(bmm));
    }
#endif

}  // namespace my_ops
