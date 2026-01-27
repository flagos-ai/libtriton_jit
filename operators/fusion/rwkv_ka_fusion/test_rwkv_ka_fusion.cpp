// ==============================================================================
// test_rwkv_ka_fusion.cpp - Multi-backend RWKV KA Fusion Test
// ==============================================================================

#include "rwkv_ka_fusion_op.h"
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
    constexpr int64_t BATCH = 8;
    constexpr int64_t SEQ_LEN = 128;
    constexpr int64_t HIDDEN = 256;

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

    at::Tensor k = at::randn({BATCH, SEQ_LEN, HIDDEN}, at::TensorOptions().device(device));
    at::Tensor a = at::randn({BATCH, SEQ_LEN, HIDDEN}, at::TensorOptions().device(device));

    std::cout << "Testing rwkv_ka_fusion operator..." << std::endl;
    std::cout << "Input shape: [" << BATCH << ", " << SEQ_LEN << ", " << HIDDEN << "]" << std::endl;

    at::Tensor result = my_ops::rwkv_ka_fusion(k, a);
    device_synchronize();

    std::cout << "Output shape: [" << result.size(0) << ", " << result.size(1) << ", " << result.size(2) << "]" << std::endl;
    std::cout << "rwkv_ka_fusion test completed!" << std::endl;
    return 0;
}
