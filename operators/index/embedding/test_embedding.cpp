// ==============================================================================
// test_embedding.cpp - Multi-backend Triton JIT Embedding Test
// ==============================================================================

#include "embedding_op.h"
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
    constexpr int64_t NUM_EMBEDDINGS = 1000;
    constexpr int64_t EMBEDDING_DIM = 256;
    constexpr int64_t BATCH = 16;
    constexpr int64_t SEQ_LEN = 32;

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

    at::Tensor weight = at::randn({NUM_EMBEDDINGS, EMBEDDING_DIM}, at::TensorOptions().device(device));
    at::Tensor indices = at::randint(0, NUM_EMBEDDINGS, {BATCH, SEQ_LEN}, 
                                      at::TensorOptions().dtype(at::kLong).device(device));

    std::cout << "Testing embedding operator..." << std::endl;
    std::cout << "Weight shape: [" << NUM_EMBEDDINGS << ", " << EMBEDDING_DIM << "]" << std::endl;
    std::cout << "Indices shape: [" << BATCH << ", " << SEQ_LEN << "]" << std::endl;

    at::Tensor result = my_ops::embedding(indices, weight);
    device_synchronize();

    std::cout << "Output shape: [" << result.size(0) << ", " << result.size(1) 
              << ", " << result.size(2) << "]" << std::endl;
    std::cout << "Embedding test completed!" << std::endl;
    return 0;
}
