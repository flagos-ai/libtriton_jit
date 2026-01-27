// ==============================================================================
// test_sum.cpp - Multi-backend Triton JIT Sum Reduction Test
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "sum_op.h"
#include "torch/torch.h"
#include <iostream>
#include <cstdlib>

// ==============================================================================
//                         BACKEND DETECTION & HEADERS
// ==============================================================================

#if defined(BACKEND_NPU)
    // ----------------------------- NPU Backend -----------------------------
    #include "acl/acl.h"
    #include "acl/acl_rt.h"
    #if __has_include("torch_npu/torch_npu.h")
        #include <torch_npu/torch_npu.h>
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif

#elif defined(BACKEND_MUSA)
    // ----------------------------- MUSA Backend ----------------------------
    #include "musa.h"
    #include "musa_runtime.h"
    #if __has_include("torch_musa/torch_musa.h")
        #include <torch_musa/torch_musa.h>
        #define HAS_TORCH_MUSA 1
    #else
        #define HAS_TORCH_MUSA 0
    #endif

#else
    // ----------------------- CUDA / IX Backend (Default) -------------------
    #include "c10/cuda/CUDAFunctions.h"

#endif

// ==============================================================================
//                         BACKEND-SPECIFIC UTILITIES
// ==============================================================================

namespace {

// ----------------------------- Device Synchronize ----------------------------

inline void device_synchronize() {
#if defined(BACKEND_NPU)
    aclrtSynchronizeDevice();
#elif defined(BACKEND_MUSA)
    musaDeviceSynchronize();
#else
    c10::cuda::device_synchronize();
#endif
}

// ----------------------------- Device Initialization -------------------------

#if defined(BACKEND_NPU)

int init_npu_device(at::Device& device) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

    int32_t deviceId = 1;
    const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
        std::cout << "Using NPU device from env: " << deviceId << std::endl;
    } else {
        std::cout << "NPU_DEVICE_ID not set, using default: " << deviceId << std::endl;
    }

    aclError ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclrtSetDevice failed: " << ret << std::endl;
        return -1;
    }

    #if HAS_TORCH_NPU
        std::string npu_device_str = "npu:" + std::to_string(deviceId);
        torch_npu::init_npu(npu_device_str);
        device = at::Device(npu_device_str);
        std::cout << "NPU initialized: " << device << std::endl;
        return 0;
    #else
        std::cerr << "torch_npu not available" << std::endl;
        return -1;
    #endif
}

void finalize_npu_device() {
    #if HAS_TORCH_NPU
        int32_t deviceId = 0;
        const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
        if (deviceEnv != nullptr) {
            deviceId = std::atoi(deviceEnv);
        }
        aclrtResetDevice(deviceId);
        aclFinalize();
    #endif
}

#elif defined(BACKEND_MUSA)

int init_musa_device(at::Device& device) {
    int32_t deviceId = 0;
    const char* deviceEnv = std::getenv("MUSA_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
    }

    musaError_t ret = musaSetDevice(deviceId);
    if (ret != MUSA_SUCCESS) {
        std::cerr << "musaSetDevice failed: " << ret << std::endl;
        return -1;
    }

    #if HAS_TORCH_MUSA
        std::string musa_device_str = "musa:" + std::to_string(deviceId);
        device = at::Device(musa_device_str);
        std::cout << "MUSA initialized: " << device << std::endl;
        return 0;
    #else
        std::cerr << "torch_musa not available" << std::endl;
        return -1;
    #endif
}

void finalize_musa_device() {
    musaDeviceReset();
}

#else  // CUDA / IX

int init_cuda_device(at::Device& device) {
    device = at::Device(at::kCUDA);
    std::cout << "CUDA device initialized" << std::endl;
    return 0;
}

void finalize_cuda_device() {
    // CUDA cleanup is automatic
}

#endif

// ----------------------------- Tensor Creation -------------------------------

inline at::Tensor create_random_tensor(
    const std::vector<int64_t>& shape, const at::Device& device) {
#if defined(BACKEND_NPU)
    return at::rand(shape, device);
#elif defined(BACKEND_MUSA)
    return at::rand(shape, device);
#else
    return at::rand(shape, at::kCUDA);
#endif
}

}  // anonymous namespace

// ==============================================================================
//                                   MAIN
// ==============================================================================

int main() {
    constexpr int64_t M = 16;
    constexpr int64_t N = 4 * 1024;
    constexpr int WARMUP_ITERS = 10;
    constexpr int BENCH_ITERS = 10;

    // ======================== Device Initialization ==========================
    at::Device device(at::kCPU);

#if defined(BACKEND_NPU)
    if (init_npu_device(device) != 0) return -1;
#elif defined(BACKEND_MUSA)
    if (init_musa_device(device) != 0) return -1;
#else
    if (init_cuda_device(device) != 0) return -1;
#endif

    // ======================== Create Test Tensors ============================
    at::Tensor tensor = create_random_tensor({M, N}, device);

    std::cout << "\n=== Input Tensor Info ===" << std::endl;
    std::cout << "Shape: [" << M << ", " << N << "]" << std::endl;
    std::cout << "Device: " << tensor.device() << std::endl;

    at::Tensor tensor_cpu = tensor.cpu();
    std::cout << "tensor[0, 0:5]: " << tensor_cpu[0].slice(0, 0, 5) << std::endl;

    // ======================== Warm-up & Compute ==============================
    std::cout << "\n=== Executing Computation ===" << std::endl;
    at::Tensor result1 = my_ops::sum_dim(tensor, {1}, false, c10::nullopt);
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    // NPU/MUSA: Use CPU computation as reference
    at::Tensor result2_cpu = at::sum(tensor_cpu, {1}, false, c10::nullopt);
#else
    at::Tensor result2 = at::sum(tensor, {1}, false, c10::nullopt);
#endif
    device_synchronize();

    // ======================== Result Verification ============================
    std::cout << "\n=== Results ===" << std::endl;
    at::Tensor result1_cpu = result1.cpu();
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    std::cout << "my_ops::sum_dim[0:5]: " << result1_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "CPU reference[0:5]:   " << result2_cpu.slice(0, 0, 5) << std::endl;
    bool is_close = at::allclose(result1_cpu, result2_cpu, 1e-4, 1e-4);
#else
    at::Tensor result2_cpu = result2.cpu();
    std::cout << "my_ops::sum_dim[0:5]: " << result1_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "at::sum[0:5]:         " << result2_cpu.slice(0, 0, 5) << std::endl;
    bool is_close = at::allclose(result1, result2, 1e-4, 1e-4);
#endif
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    if (!is_close) {
        auto diff = at::abs(result1_cpu - result2_cpu);
        std::cout << "Max difference: " << at::max(diff).item<float>() << std::endl;
    }

    // ======================== Performance Benchmark ==========================
    std::cout << "\n=== Performance Benchmark ===" << std::endl;

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA)
    // Warm-up: at::sum (skip on NPU/MUSA)
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        auto tmp = at::sum(tensor, {1}, false, c10::nullopt);
    }
    device_synchronize();
#endif

    // Warm-up: my_ops::sum_dim
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        auto tmp = my_ops::sum_dim(tensor, {1}, false, c10::nullopt);
    }
    device_synchronize();

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA)
    // Benchmark: at::sum (skip on NPU/MUSA)
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto tmp = at::sum(tensor, {1}, false, c10::nullopt);
    }
    device_synchronize();
    std::cout << "at::sum benchmark completed (" << BENCH_ITERS << " iters)" << std::endl;
#endif

    // Benchmark: my_ops::sum_dim
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto tmp = my_ops::sum_dim(tensor, {1}, false, c10::nullopt);
    }
    device_synchronize();
    std::cout << "my_ops::sum_dim benchmark completed (" << BENCH_ITERS << " iters)" << std::endl;

    // ======================== Cleanup ========================================
#if defined(BACKEND_NPU)
    finalize_npu_device();
#elif defined(BACKEND_MUSA)
    finalize_musa_device();
#else
    finalize_cuda_device();
#endif

    std::cout << "\nProgram completed successfully!" << std::endl;
    return 0;
}
