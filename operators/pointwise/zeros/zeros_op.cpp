// ==============================================================================
// zeros_op.cpp - Multi-backend Triton JIT Zeros Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "zeros_op.h"
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

inline RawStream get_device_stream([[maybe_unused]] const at::Device& device) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(device.index()).stream();
    #else
        return nullptr;
    #endif
#elif defined(BACKEND_MUSA)
    return nullptr;
#else
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(device.index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

namespace my_ops {
using namespace triton_jit;

at::Tensor zeros_like(const at::Tensor& input) {
    // Allocate output
#if defined(BACKEND_MUSA)
    void* d_ptr = nullptr;
    size_t num_bytes = input.numel() * at::elementSize(input.scalar_type());
    musaError_t err = musaMalloc(&d_ptr, num_bytes);
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }

    auto options = at::TensorOptions().dtype(input.scalar_type()).device(input.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor out = at::from_blob(d_ptr, input.sizes(), deleter, options);
#else
    at::Tensor out = at::empty(input.sizes(),
        at::TensorOptions().dtype(input.scalar_type()).device(input.device()));
#endif

    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("zeros.py"), "zeros_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    c10::DeviceGuard guard(out.device());
    RawStream stream = get_device_stream(input.device());

    f(stream, num_blocks, 1, 1, num_warps, num_stages, out, n, tile_size);

    return out;
}

at::Tensor zeros(at::IntArrayRef size, at::ScalarType dtype, const at::Device& device) {
    // Allocate output
#if defined(BACKEND_MUSA)
    int64_t numel = 1;
    for (auto s : size) numel *= s;

    void* d_ptr = nullptr;
    size_t num_bytes = numel * at::elementSize(dtype);
    musaError_t err = musaMalloc(&d_ptr, num_bytes);
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }

    auto options = at::TensorOptions().dtype(dtype).device(device);
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor out = at::from_blob(d_ptr, size.vec(), deleter, options);
#else
    at::Tensor out = at::empty(size, at::TensorOptions().dtype(dtype).device(device));
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("zeros.py"), "zeros_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    c10::DeviceGuard guard(out.device());
    RawStream stream = get_device_stream(device);

    f(stream, num_blocks, 1, 1, num_warps, num_stages, out, n, tile_size);

    return out;
}

TORCH_LIBRARY(zeros_ops, m) {
    m.def("zeros_like(Tensor self) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(zeros_ops, PrivateUse1, m) {
        m.impl("zeros_like", TORCH_FN(zeros_like));
    }
#else
    TORCH_LIBRARY_IMPL(zeros_ops, CUDA, m) {
        m.impl("zeros_like", TORCH_FN(zeros_like));
    }
#endif

}  // namespace my_ops
