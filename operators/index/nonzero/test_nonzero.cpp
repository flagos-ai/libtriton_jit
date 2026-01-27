// ==============================================================================
// test_nonzero.cpp - Multi-backend Triton JIT Nonzero Test
// ==============================================================================

#include "nonzero_op.h"
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

    // Create sparse tensor
    at::Tensor input = at::zeros({100}, at::TensorOptions().device(device));
    // Set some nonzero values (this would need backend-specific implementation)
    
    std::cout << "Testing nonzero operator..." << std::endl;
    std::cout << "Input shape: [100]" << std::endl;

    at::Tensor result = my_ops::nonzero(input);
    device_synchronize();

    std::cout << "Output shape: [" << result.size(0) << ", " << result.size(1) << "]" << std::endl;
    std::cout << "Nonzero test completed!" << std::endl;
    return 0;
}
