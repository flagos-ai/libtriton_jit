// ==============================================================================
// fill_op.cpp - Multi-backend Triton JIT In-place Fill Operation
// ==============================================================================

#include "fill_op.h"
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

at::Tensor& fill_(at::Tensor& tensor, const at::Scalar& value) {
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous for in-place fill");

    int64_t n_elements = tensor.numel();
    float fill_value = value.toFloat();

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("fill_.py"), "fill_kernel");

    constexpr int64_t BLOCK_SIZE = 1024;
    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    int64_t num_blocks = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    c10::DeviceGuard guard(tensor.device());
    RawStream stream = get_device_stream(tensor);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      tensor, fill_value, n_elements, BLOCK_SIZE);

    return tensor;
}

TORCH_LIBRARY(fill_inplace_ops, m) {
    m.def("fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(fill_inplace_ops, PrivateUse1, m) {
        m.impl("fill_", TORCH_FN(fill_));
    }
#else
    TORCH_LIBRARY_IMPL(fill_inplace_ops, CUDA, m) {
        m.impl("fill_", TORCH_FN(fill_));
    }
#endif

}  // namespace my_ops
