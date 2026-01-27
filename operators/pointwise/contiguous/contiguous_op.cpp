// ==============================================================================
// contiguous_op.cpp - Multi-backend Triton JIT Contiguous Operation
// ==============================================================================

#include "contiguous_op.h"
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

at::Tensor contiguous(const at::Tensor& input) {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    // Allocate contiguous output
#if defined(BACKEND_MUSA)
    void* d_ptr = nullptr;
    size_t num_bytes = input.numel() * at::elementSize(input.scalar_type());
    musaError_t err = musaMalloc(&d_ptr, num_bytes);
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }

    auto options = at::TensorOptions().dtype(input.scalar_type()).device(input.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor out = at::from_blob(d_ptr, input.sizes().vec(), deleter, options);
#else
    at::Tensor out = at::empty(input.sizes(),
        at::TensorOptions().dtype(input.scalar_type()).device(input.device())
        .memory_format(c10::MemoryFormat::Contiguous));
#endif

    c10::DeviceGuard guard(input.device());
    RawStream stream = get_device_stream(input);

    if (input.dim() == 1 || input.numel() == 0) {
        // 1D case: use simple copy kernel
        const TritonJITFunction& f = TritonJITFunction::get_instance(
            std::string("contiguous.py"), "copy_1d_kernel");

        constexpr int64_t tile_size = 1024;
        constexpr int num_warps = 8;
        constexpr int num_stages = 1;
        const int64_t n = input.numel();
        const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

        f(stream, num_blocks, 1, 1, num_warps, num_stages,
          input.flatten(), out.flatten(), n, tile_size);
    } else {
        // Multi-dimensional case: use 2D strided kernel
        const TritonJITFunction& f = TritonJITFunction::get_instance(
            std::string("contiguous.py"), "copy_strided_2d_kernel");

        // Flatten to 2D
        auto input_2d = input.view({-1, input.size(-1)});
        auto out_2d = out.view({-1, out.size(-1)});

        const int64_t n_rows = input_2d.size(0);
        const int64_t n_cols = input_2d.size(1);

        constexpr int64_t BLOCK_M = 32;
        constexpr int64_t BLOCK_N = 32;
        constexpr int num_warps = 4;
        constexpr int num_stages = 1;

        const unsigned int grid_m = (n_rows + BLOCK_M - 1) / BLOCK_M;
        const unsigned int grid_n = (n_cols + BLOCK_N - 1) / BLOCK_N;

        f(stream, grid_m, grid_n, 1, num_warps, num_stages,
          input_2d, out_2d,
          n_rows, n_cols,
          input_2d.stride(0), input_2d.stride(1),
          out_2d.stride(0), out_2d.stride(1),
          BLOCK_M, BLOCK_N);
    }

    return out;
}

TORCH_LIBRARY(contiguous_ops, m) {
    m.def("contiguous(Tensor self) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(contiguous_ops, PrivateUse1, m) {
        m.impl("contiguous", TORCH_FN(contiguous));
    }
#else
    TORCH_LIBRARY_IMPL(contiguous_ops, CUDA, m) {
        m.impl("contiguous", TORCH_FN(contiguous));
    }
#endif

}  // namespace my_ops
