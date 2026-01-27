// ==============================================================================
// test_addmm.cpp - Multi-backend Triton JIT Addmm Test
// ==============================================================================

#include "addmm_op.h"
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
    constexpr int64_t M = 64;
    constexpr int64_t K = 128;
    constexpr int64_t N = 64;

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

    at::Tensor input = at::randn({M, N}, at::TensorOptions().device(device));
    at::Tensor a = at::randn({M, K}, at::TensorOptions().device(device));
    at::Tensor b = at::randn({K, N}, at::TensorOptions().device(device));

    std::cout << "Testing addmm operator..." << std::endl;
    std::cout << "Input: [" << M << ", " << N << "]" << std::endl;
    std::cout << "A: [" << M << ", " << K << "] x B: [" << K << ", " << N << "]" << std::endl;

    at::Tensor result = my_ops::addmm(input, a, b, 0.5, 1.0);
    device_synchronize();

    std::cout << "Output shape: [" << result.size(0) << ", " << result.size(1) << "]" << std::endl;

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA)
    at::Tensor expected = at::addmm(input, a, b, 0.5, 1.0);
    bool match = at::allclose(result, expected, 1e-3, 1e-3);
    std::cout << "Results match reference: " << (match ? "YES" : "NO") << std::endl;
#endif

    std::cout << "Addmm test completed!" << std::endl;
    return 0;
}
