// ==============================================================================
// test_fused_add_rms_norm.cpp - Multi-backend Fused Add RMS Norm Test
// ==============================================================================

#include "fused_add_rms_norm_op.h"
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
    at::Tensor residual = at::randn({BATCH, HIDDEN}, at::TensorOptions().device(device));
    at::Tensor weight = at::ones({HIDDEN}, at::TensorOptions().device(device));

    std::cout << "Testing fused_add_rms_norm operator..." << std::endl;
    std::cout << "Input shape: [" << BATCH << ", " << HIDDEN << "]" << std::endl;

    auto [output, res_out] = my_ops::fused_add_rms_norm(input, residual, weight, 1e-6);
    device_synchronize();

    std::cout << "Output shape: [" << output.size(0) << ", " << output.size(1) << "]" << std::endl;
    std::cout << "Residual out shape: [" << res_out.size(0) << ", " << res_out.size(1) << "]" << std::endl;
    std::cout << "Fused add RMS norm test completed!" << std::endl;
    return 0;
}
