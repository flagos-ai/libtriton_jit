// ==============================================================================
// test_reshape_and_cache_flash.cpp - Multi-backend Reshape and Cache Test
// ==============================================================================

#include "reshape_and_cache_flash_op.h"
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
    constexpr int64_t NUM_TOKENS = 32;
    constexpr int64_t NUM_HEADS = 8;
    constexpr int64_t HEAD_DIM = 64;
    constexpr int64_t NUM_BLOCKS = 16;
    constexpr int64_t BLOCK_SIZE = 16;

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

    at::Tensor key = at::randn({NUM_TOKENS, NUM_HEADS, HEAD_DIM}, at::TensorOptions().device(device));
    at::Tensor value = at::randn({NUM_TOKENS, NUM_HEADS, HEAD_DIM}, at::TensorOptions().device(device));
    at::Tensor key_cache = at::zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM}, at::TensorOptions().device(device));
    at::Tensor value_cache = at::zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM}, at::TensorOptions().device(device));
    at::Tensor slot_mapping = at::arange(NUM_TOKENS, at::TensorOptions().dtype(at::kLong).device(device));

    std::cout << "Testing reshape_and_cache_flash operator..." << std::endl;
    std::cout << "Key shape: [" << NUM_TOKENS << ", " << NUM_HEADS << ", " << HEAD_DIM << "]" << std::endl;
    std::cout << "Cache shape: [" << NUM_BLOCKS << ", " << NUM_HEADS << ", " << BLOCK_SIZE << ", " << HEAD_DIM << "]" << std::endl;

    my_ops::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping);
    device_synchronize();

    std::cout << "reshape_and_cache_flash test completed!" << std::endl;
    return 0;
}
