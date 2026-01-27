// ==============================================================================
// test_fill_.cpp - Multi-backend Triton JIT In-place Fill Test
// ==============================================================================

#include "fill_op.h"
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
    constexpr int64_t SIZE = 1024;
    constexpr float FILL_VALUE = 3.14f;

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

    at::Tensor tensor = at::empty({SIZE}, at::TensorOptions().device(device));

    std::cout << "Testing fill_ operator..." << std::endl;
    std::cout << "Tensor size: " << SIZE << std::endl;
    std::cout << "Fill value: " << FILL_VALUE << std::endl;

    my_ops::fill_(tensor, FILL_VALUE);
    device_synchronize();

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA)
    at::Tensor expected = at::full({SIZE}, FILL_VALUE, at::TensorOptions().device(device));
    bool match = at::allclose(tensor, expected);
    std::cout << "Results match reference: " << (match ? "YES" : "NO") << std::endl;
#endif

    std::cout << "fill_ test completed!" << std::endl;
    return 0;
}
