// ==============================================================================
// test_rms_norm.cpp - Multi-backend Triton JIT RMS Norm Test
// ==============================================================================

#include "rms_norm_op.h"
#include "torch/torch.h"
#include <iostream>

#if defined(BACKEND_NPU)
    #include "acl/acl.h"
#elif defined(BACKEND_MUSA)
    #include "musa_runtime.h"
    #include "pybind11/embed.h"
#else
    #include "c10/cuda/CUDAFunctions.h"
#endif

namespace {

inline void device_synchronize() {
#if defined(BACKEND_NPU)
    aclrtSynchronizeDevice();
#elif defined(BACKEND_MUSA)
    musaDeviceSynchronize();
#else
    c10::cuda::device_synchronize();
#endif
}

}  // anonymous namespace

int main() {
    constexpr int64_t BATCH = 16;
    constexpr int64_t HIDDEN = 1024;

#if defined(BACKEND_MUSA)
    namespace py = pybind11;
    py::scoped_interpreter guard{};
    py::module_::import("torch_musa");
    at::Device device(at::DeviceType::PrivateUse1, 0);
#elif defined(BACKEND_NPU)
    at::Device device(at::kPrivateUse1);
#else
    at::Device device(at::kCUDA);
#endif

    at::Tensor input = at::randn({BATCH, HIDDEN}, at::TensorOptions().device(device));
    at::Tensor weight = at::ones({HIDDEN}, at::TensorOptions().device(device));

    std::cout << "Testing rms_norm operator..." << std::endl;
    std::cout << "Input shape: [" << BATCH << ", " << HIDDEN << "]" << std::endl;

    at::Tensor result = my_ops::rms_norm(input, weight, 1e-6);
    device_synchronize();

    std::cout << "Output shape: [" << result.size(0) << ", " << result.size(1) << "]" << std::endl;
    std::cout << "RMS norm test completed!" << std::endl;
    return 0;
}
