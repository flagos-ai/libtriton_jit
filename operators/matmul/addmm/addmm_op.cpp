// ==============================================================================
// addmm_op.cpp - Multi-backend Triton JIT Additive Matrix Multiplication
// C = beta * input + alpha * (A @ B)
// ==============================================================================

#include "addmm_op.h"
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

#if defined(BACKEND_MUSA)
    void* c_ptr = nullptr;
    size_t c_bytes = M * N * at::elementSize(a.scalar_type());
    musaMalloc(&c_ptr, c_bytes);
    auto opts = at::TensorOptions().dtype(a.scalar_type()).device(a.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor c = at::from_blob(c_ptr, {M, N}, deleter, opts);
#else
    at::Tensor c = at::empty({M, N}, a.options());
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("addmm.py"), "addmm_kernel");

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
      input_expanded, a_contig, b_contig, c,
      M, N, K,
      alpha_val, beta_val,
      input_expanded.stride(0), input_expanded.stride(1),
      a_contig.stride(0), a_contig.stride(1),
      b_contig.stride(0), b_contig.stride(1),
      c.stride(0), c.stride(1),
      BLOCK_M, BLOCK_N, BLOCK_K);

    return c;
}

TORCH_LIBRARY(addmm_ops, m) {
    m.def("addmm(Tensor input, Tensor mat1, Tensor mat2, Scalar beta, Scalar alpha) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(addmm_ops, PrivateUse1, m) {
        m.impl("addmm", TORCH_FN(addmm));
    }
#else
    TORCH_LIBRARY_IMPL(addmm_ops, CUDA, m) {
        m.impl("addmm", TORCH_FN(addmm));
    }
#endif

}  // namespace my_ops
