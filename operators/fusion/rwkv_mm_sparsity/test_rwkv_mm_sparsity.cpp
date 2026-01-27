// ==============================================================================
// test_rwkv_mm_sparsity.cpp - Multi-backend RWKV MM Sparsity Test
// ==============================================================================

#include "rwkv_mm_sparsity_op.h"
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
    constexpr int64_t M = 256;
    constexpr int64_t K = 128;
    constexpr int64_t N = 256;
    constexpr int64_t BLOCK_M = 64;

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

    at::Tensor a = at::randn({M, K}, at::TensorOptions().device(device));
    at::Tensor b = at::randn({K, N}, at::TensorOptions().device(device));
    at::Tensor mask = at::ones({M / BLOCK_M}, at::TensorOptions().dtype(at::kInt).device(device));

    std::cout << "Testing rwkv_mm_sparsity operator..." << std::endl;
    std::cout << "A: [" << M << ", " << K << "], B: [" << K << ", " << N << "]" << std::endl;

    at::Tensor result = my_ops::rwkv_mm_sparsity(a, b, mask);
    device_synchronize();

    std::cout << "Output shape: [" << result.size(0) << ", " << result.size(1) << "]" << std::endl;
    std::cout << "rwkv_mm_sparsity test completed!" << std::endl;
    return 0;
}
