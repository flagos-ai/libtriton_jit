// ==============================================================================
// test_apply_rotary_pos_emb.cpp - Multi-backend Rotary Position Embedding Test
// ==============================================================================

#include "apply_rotary_pos_emb_op.h"
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
    constexpr int64_t SEQ_LEN = 128;
    constexpr int64_t NUM_HEADS = 8;
    constexpr int64_t HEAD_DIM = 64;

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

    at::Tensor q = at::randn({SEQ_LEN, NUM_HEADS, HEAD_DIM}, at::TensorOptions().device(device));
    at::Tensor k = at::randn({SEQ_LEN, NUM_HEADS, HEAD_DIM}, at::TensorOptions().device(device));
    at::Tensor cos = at::randn({SEQ_LEN, HEAD_DIM / 2}, at::TensorOptions().device(device));
    at::Tensor sin = at::randn({SEQ_LEN, HEAD_DIM / 2}, at::TensorOptions().device(device));

    std::cout << "Testing apply_rotary_pos_emb operator..." << std::endl;
    std::cout << "Q/K shape: [" << SEQ_LEN << ", " << NUM_HEADS << ", " << HEAD_DIM << "]" << std::endl;

    auto [q_out, k_out] = my_ops::apply_rotary_pos_emb(q, k, cos, sin, HEAD_DIM);
    device_synchronize();

    std::cout << "Q output shape: [" << q_out.size(0) << ", " << q_out.size(1) << ", " << q_out.size(2) << "]" << std::endl;
    std::cout << "K output shape: [" << k_out.size(0) << ", " << k_out.size(1) << ", " << k_out.size(2) << "]" << std::endl;
    std::cout << "apply_rotary_pos_emb test completed!" << std::endl;
    return 0;
}
