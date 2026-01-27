// ==============================================================================
// exponential_op.cpp - Multi-backend Triton JIT Exponential_ Operation
// ==============================================================================

#include "exponential_op.h"
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

at::Tensor& exponential_(at::Tensor& input, double lambd) {
    // Ensure tensor is float type
    TORCH_CHECK(input.is_floating_point(), "exponential_ requires floating point tensor");

    // First fill with uniform random
#if defined(BACKEND_MUSA)
    // For MUSA, generate uniform on CPU and copy
    auto cpu_uniform = at::rand(input.sizes(), at::TensorOptions().dtype(input.scalar_type()).device(at::kCPU));
    musaMemcpy(input.data_ptr(), cpu_uniform.data_ptr(),
               input.numel() * at::elementSize(input.scalar_type()),
               musaMemcpyHostToDevice);
#else
    input.uniform_(0, 1);
#endif

    // Kernel setup
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("exponential_.py"), "exponential_kernel");

    constexpr int64_t tile_size  = 1024;
    constexpr int     num_warps  = 8;
    constexpr int     num_stages = 1;
    const int64_t n = input.numel();
    const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

    c10::DeviceGuard guard(input.device());
    RawStream stream = get_device_stream(input);

    float float_lambd = static_cast<float>(lambd);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, input, input, float_lambd, n, tile_size);

    return input;
}

TORCH_LIBRARY(exponential_ops, m) {
    m.def("exponential_(Tensor(a!) self, float lambd=1.0) -> Tensor(a!)");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(exponential_ops, PrivateUse1, m) {
        m.impl("exponential_", TORCH_FN(exponential_));
    }
#else
    TORCH_LIBRARY_IMPL(exponential_ops, CUDA, m) {
        m.impl("exponential_", TORCH_FN(exponential_));
    }
#endif

}  // namespace my_ops
