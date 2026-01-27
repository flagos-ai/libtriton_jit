// ==============================================================================
// test_add.cpp - Multi-backend Triton JIT Add Operation Test
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "add_op.h"
#include "torch/torch.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>

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
    // Use MUSA Runtime API only (no torch_musa C++ headers to avoid conflicts)
    #include "musa.h"
    #include "musa_runtime.h"

    // For PrivateUse1 backend registration via Python
    #include "pybind11/embed.h"

#else
    // ----------------------- CUDA / IX Backend (Default) -------------------
    #include "c10/cuda/CUDAFunctions.h"
    #include <cuda_runtime.h>

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

// Note: For MUSA backend, torch_musa will handle all device registration
// including PrivateUse1 hooks, so we don't register them manually here

int init_musa_device(at::Device& device) {
    setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

    int32_t deviceId = 0;
    const char* deviceEnv = std::getenv("MUSA_DEVICE_ID");
    if (deviceEnv != nullptr) {
        deviceId = std::atoi(deviceEnv);
        std::cout << "[DEBUG] Using MUSA device from env: " << deviceId << std::endl;
    } else {
        std::cout << "[DEBUG] MUSA_DEVICE_ID not set, using default: " << deviceId << std::endl;
    }

    // Use MUSA Runtime API for initialization
    std::cout << "[DEBUG] Initializing MUSA runtime..." << std::endl;
    musaError_t ret = musaSetDevice(deviceId);
    if (ret != musaSuccess) {
        std::cerr << "[ERROR] musaSetDevice failed: " << musaGetErrorString(ret)
                  << " (" << ret << ")" << std::endl;
        return -1;
    }
    std::cout << "[DEBUG] musaSetDevice succeeded" << std::endl;

    // Get device properties for verification
    int deviceCount;
    musaGetDeviceCount(&deviceCount);
    std::cout << "[DEBUG] Total MUSA devices: " << deviceCount << std::endl;

    musaDeviceProp prop;
    musaGetDeviceProperties(&prop, deviceId);
    std::cout << "[DEBUG] Device " << deviceId << ": " << prop.name << std::endl;

    // Initialize Python and import torch_musa to register PrivateUse1 hooks
    // This must be done BEFORE creating any tensors with at::from_blob
    std::cout << "[DEBUG] Initializing Python and importing torch_musa..." << std::endl;
    namespace py = pybind11;
    if (!Py_IsInitialized()) {
        py::initialize_interpreter();
    }

    try {
        py::gil_scoped_acquire gil;
        py::module_::import("torch_musa");
        std::cout << "[DEBUG] torch_musa imported successfully" << std::endl;
    } catch (const py::error_already_set& e) {
        std::cerr << "[ERROR] Failed to import torch_musa: " << e.what() << std::endl;
        return -1;
    }

    // Create at::Device using PrivateUse1
    device = at::Device(at::DeviceType::PrivateUse1, deviceId);
    std::cout << "MUSA device initialized: " << device << " (PrivateUse1)" << std::endl;

    return 0;
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

inline at::Tensor create_random_tensor(int64_t size, const at::Device& device) {
#if defined(BACKEND_NPU)
    return at::rand({size}, device);

#elif defined(BACKEND_MUSA)
    // For MUSA: manually allocate memory and wrap with tensor to avoid aten::empty dispatch
    // This works because torch_musa is not loaded in pure C++ (no operator registration)

    // 1. Allocate on CPU
    std::vector<float> h_data(size);
    for (int64_t i = 0; i < size; ++i) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 2. Allocate on MUSA device
    void* d_ptr = nullptr;
    musaError_t err = musaMalloc(&d_ptr, size * sizeof(float));
    if (err != musaSuccess) {
        throw std::runtime_error("musaMalloc failed: " + std::string(musaGetErrorString(err)));
    }

    // 3. Copy data to device
    err = musaMemcpy(d_ptr, h_data.data(), size * sizeof(float), musaMemcpyHostToDevice);
    if (err != musaSuccess) {
        musaFree(d_ptr);
        throw std::runtime_error("musaMemcpy failed: " + std::string(musaGetErrorString(err)));
    }

    // 4. Wrap MUSA pointer in PyTorch tensor
    auto options = at::TensorOptions()
        .dtype(at::kFloat)
        .device(device);

    // Use from_blob with custom deleter to manage MUSA memory
    auto deleter = [](void* ptr) { musaFree(ptr); };
    return at::from_blob(d_ptr, {size}, deleter, options);

#else
    return at::rand({size}, at::kCUDA);
#endif
}

}  // anonymous namespace

// ==============================================================================
//                                   MAIN
// ==============================================================================

int main() {
    constexpr int64_t TENSOR_SIZE = 128 * 1024;
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
    at::Tensor a = create_random_tensor(TENSOR_SIZE, device);
    at::Tensor b = create_random_tensor(TENSOR_SIZE, device);

    std::cout << "\n=== Input Tensor Info ===" << std::endl;
    std::cout << "Size: " << a.size(0) << " elements" << std::endl;
    std::cout << "Device: " << a.device() << std::endl;

#if defined(BACKEND_MUSA)
    // MUSA: Manual memory copy to avoid missing copy operators
    std::vector<float> a_cpu_vec(5);
    std::vector<float> b_cpu_vec(5);
    musaMemcpy(a_cpu_vec.data(), a.data_ptr<float>(), 5 * sizeof(float), musaMemcpyDeviceToHost);
    musaMemcpy(b_cpu_vec.data(), b.data_ptr<float>(), 5 * sizeof(float), musaMemcpyDeviceToHost);
    std::cout << "a[0:5]: [";
    for(int i = 0; i < 5; i++) std::cout << (i?", ":"") << a_cpu_vec[i];
    std::cout << "]" << std::endl;
    std::cout << "b[0:5]: [";
    for(int i = 0; i < 5; i++) std::cout << (i?", ":"") << b_cpu_vec[i];
    std::cout << "]" << std::endl;
#else
    at::Tensor a_cpu = a.cpu();
    at::Tensor b_cpu = b.cpu();
    std::cout << "a[0:5]: " << a_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "b[0:5]: " << b_cpu.slice(0, 0, 5) << std::endl;
#endif

    // ======================== Warm-up & Compute ==============================
    std::cout << "\n=== Executing Computation ===" << std::endl;

#if defined(BACKEND_MUSA)
    // Clear any previous MUSA errors
    musaGetLastError();
#elif defined(BACKEND_CUDA) || defined(BACKEND_IX)
    cudaGetLastError();
#endif

    at::Tensor result1 = my_ops::add_tensor(a, b);

#if defined(BACKEND_MUSA)
    // Check for MUSA errors after kernel execution
    musaError_t exec_err = musaGetLastError();
    if (exec_err != musaSuccess) {
        std::cerr << "[ERROR] MUSA error after kernel execution: " << musaGetErrorString(exec_err) << std::endl;
    }
#elif defined(BACKEND_CUDA) || defined(BACKEND_IX)
    cudaError_t exec_err = cudaGetLastError();
    if (exec_err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA error after kernel execution: " << cudaGetErrorString(exec_err) << std::endl;
    }
#endif

    device_synchronize();

#if defined(BACKEND_MUSA)
    // Check for sync errors
    musaError_t sync_err = musaGetLastError();
    if (sync_err != musaSuccess) {
        std::cerr << "[ERROR] MUSA error after synchronize: " << musaGetErrorString(sync_err) << std::endl;
    }
#elif defined(BACKEND_CUDA) || defined(BACKEND_IX)
    cudaError_t sync_err = cudaGetLastError();
    if (sync_err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA error after synchronize: " << cudaGetErrorString(sync_err) << std::endl;
    }
#endif

    // ======================== Result Verification ============================
    std::cout << "\n=== Results ===" << std::endl;

#if defined(BACKEND_MUSA)
    // MUSA: Manual verification with CPU computation
    std::vector<float> result_vec(TENSOR_SIZE);
    musaError_t err = musaMemcpy(result_vec.data(), result1.data_ptr<float>(), TENSOR_SIZE * sizeof(float), musaMemcpyDeviceToHost);
    if (err != musaSuccess) {
        std::cerr << "[DEBUG] musaMemcpy for result failed: " << musaGetErrorString(err) << std::endl;
    }

    // Compute expected on CPU
    std::vector<float> expected_vec(TENSOR_SIZE);
    std::vector<float> a_full(TENSOR_SIZE), b_full(TENSOR_SIZE);
    musaMemcpy(a_full.data(), a.data_ptr<float>(), TENSOR_SIZE * sizeof(float), musaMemcpyDeviceToHost);
    musaMemcpy(b_full.data(), b.data_ptr<float>(), TENSOR_SIZE * sizeof(float), musaMemcpyDeviceToHost);

    // Debug: print first few values from device
    std::cout << "[DEBUG] a_full[0:5] from device: [";
    for(int i = 0; i < 5; i++) std::cout << (i?", ":"") << a_full[i];
    std::cout << "]" << std::endl;
    std::cout << "[DEBUG] b_full[0:5] from device: [";
    for(int i = 0; i < 5; i++) std::cout << (i?", ":"") << b_full[i];
    std::cout << "]" << std::endl;

    for(int64_t i = 0; i < TENSOR_SIZE; i++) {
        expected_vec[i] = a_full[i] + b_full[i];
    }

    // Show first 5 elements
    std::cout << "my_ops::add_tensor[0:5]: [";
    for(int i = 0; i < 5; i++) std::cout << (i?", ":"") << result_vec[i];
    std::cout << "]" << std::endl;
    std::cout << "CPU reference[0:5]:      [";
    for(int i = 0; i < 5; i++) std::cout << (i?", ":"") << expected_vec[i];
    std::cout << "]" << std::endl;

    // Check if results match
    bool is_close = true;
    float max_diff = 0.0f;
    for(int64_t i = 0; i < TENSOR_SIZE; i++) {
        float diff = std::abs(result_vec[i] - expected_vec[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-5f) {
            is_close = false;
        }
    }
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    if (!is_close) {
        std::cout << "Max difference: " << max_diff << std::endl;
    }

#elif defined(BACKEND_NPU)
    // NPU: Use CPU computation as reference
    at::Tensor result1_cpu = result1.cpu();
    at::Tensor result2_cpu = a_cpu + b_cpu;
    std::cout << "my_ops::add_tensor[0:5]: " << result1_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "CPU reference[0:5]:      " << result2_cpu.slice(0, 0, 5) << std::endl;
    bool is_close = at::allclose(result1_cpu, result2_cpu);
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    if (!is_close) {
        auto diff = at::abs(result1_cpu - result2_cpu);
        std::cout << "Max difference: " << at::max(diff).item<float>() << std::endl;
    }

#else
    // CUDA/IX: Use at::add as reference
    at::Tensor result2 = at::add(a, b);
    at::Tensor result1_cpu = result1.cpu();
    at::Tensor result2_cpu = result2.cpu();
    std::cout << "my_ops::add_tensor[0:5]: " << result1_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "at::add[0:5]:            " << result2_cpu.slice(0, 0, 5) << std::endl;
    bool is_close = at::allclose(result1, result2);
    std::cout << "\nResults match: " << (is_close ? "YES" : "NO") << std::endl;
    if (!is_close) {
        auto diff = at::abs(result1_cpu - result2_cpu);
        std::cout << "Max difference: " << at::max(diff).item<float>() << std::endl;
    }
#endif

    // ======================== Performance Benchmark ==========================
    std::cout << "\n=== Performance Benchmark ===" << std::endl;

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA)
    // Warm-up: at::add (skip on NPU/MUSA where operators aren't registered)
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        auto tmp = at::add(a, b);
    }
    device_synchronize();
#endif

    // Warm-up: my_ops::add_tensor
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        auto tmp = my_ops::add_tensor(a, b);
    }
    device_synchronize();

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA)
    // Benchmark: at::add (skip on NPU/MUSA where operators aren't registered)
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto tmp = at::add(a, b);
    }
    device_synchronize();
    std::cout << "at::add benchmark completed (" << BENCH_ITERS << " iters)" << std::endl;
#endif

    // Benchmark: my_ops::add_tensor
    for (int i = 0; i < BENCH_ITERS; ++i) {
        auto tmp = my_ops::add_tensor(a, b);
    }
    device_synchronize();
    std::cout << "my_ops::add_tensor benchmark completed (" << BENCH_ITERS << " iters)" << std::endl;

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
