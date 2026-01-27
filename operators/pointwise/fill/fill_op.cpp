// ==============================================================================
// fill_op.cpp - Multi-backend Triton JIT Fill Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "fill_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

// ==============================================================================
//                         BACKEND DETECTION & HEADERS
// ==============================================================================

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

// ==============================================================================
//                         BACKEND-SPECIFIC TYPES & UTILITIES
// ==============================================================================

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

// ==============================================================================
//                         KERNEL IMPLEMENTATION
// ==============================================================================

namespace my_ops {
using namespace triton_jit;

at::Tensor fill_tensor(const at::Tensor& input, double value) {
    // Output allocation
#if defined(BACKEND_MUSA)
    void* d_ptr = nullptr;
    size_t num_bytes = input.numel() * at::elementSize(input.scalar_type());
    musaError_t err = musaMalloc(&d_ptr, num_bytes);
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }

    auto options = at::TensorOptions()
        .dtype(input.scalar_type())
        .device(input.device());

    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor out = at::from_blob(d_ptr, input.sizes(), deleter, options);
#else
    at::Tensor out = at::empty(input.sizes(),
        at::TensorOptions().dtype(input.scalar_type()).device(input.device()));
#endif

    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fill.py"), "fill_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = out.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    // Kernel launch
    c10::DeviceGuard guard(out.device());
    RawStream stream = get_device_stream(input);

    // Convert value to appropriate type
    float float_value = static_cast<float>(value);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, out, float_value, n, tile_size);

    return out;
}

at::Tensor& fill_tensor_(at::Tensor& input, double value) {
    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fill.py"), "fill_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = input.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    // Kernel launch
    c10::DeviceGuard guard(input.device());
    RawStream stream = get_device_stream(input);

    float float_value = static_cast<float>(value);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, input, float_value, n, tile_size);

    return input;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(fill_ops, m) {
    m.def("fill_tensor(Tensor self, float value) -> Tensor");
    m.def("fill_tensor_(Tensor(a!) self, float value) -> Tensor(a!)");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(fill_ops, PrivateUse1, m) {
        m.impl("fill_tensor", TORCH_FN(fill_tensor));
        m.impl("fill_tensor_", TORCH_FN(fill_tensor_));
    }
#else
    TORCH_LIBRARY_IMPL(fill_ops, CUDA, m) {
        m.impl("fill_tensor", TORCH_FN(fill_tensor));
        m.impl("fill_tensor_", TORCH_FN(fill_tensor_));
    }
#endif

}  // namespace my_ops
