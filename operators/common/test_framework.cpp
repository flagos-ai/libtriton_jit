/**
 * @file test_framework.cpp
 * @brief Implementation of unified test framework for Triton JIT operators
 */

#include "test_framework.h"
#include <iomanip>
#include <sstream>

namespace triton_jit {
namespace test {

// ============================================================================
// DeviceManager Implementation
// ============================================================================

DeviceManager::DeviceManager()
    : device_(at::Device(at::kCPU)), device_id_(0), initialized_(false) {
}

DeviceManager::~DeviceManager() {
    cleanup();
}

int DeviceManager::initialize(int device_id) {
    if (initialized_) {
        return 0;
    }

    device_id_ = device_id;

#if defined(BACKEND_NPU)
    // Initialize NPU
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to initialize ACL: " << ret << std::endl;
        return -1;
    }

    ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "Failed to set NPU device: " << ret << std::endl;
        return -1;
    }

    device_ = at::Device(at::DeviceType::PrivateUse1, device_id_);
    std::cout << "NPU device " << device_id_ << " initialized" << std::endl;

#elif defined(BACKEND_MUSA)
    // Initialize MUSA
    musaError_t err = musaSetDevice(device_id_);
    if (err != musaSuccess) {
        std::cerr << "Failed to set MUSA device: " << musaGetErrorString(err) << std::endl;
        return -1;
    }

    // Initialize Python interpreter for torch_musa
    interpreter_ = std::make_unique<py::scoped_interpreter>();

    try {
        py::module_::import("torch_musa");
    } catch (const py::error_already_set& e) {
        std::cerr << "Failed to import torch_musa: " << e.what() << std::endl;
        return -1;
    }

    device_ = at::Device(at::DeviceType::PrivateUse1, device_id_);
    std::cout << "MUSA device " << device_id_ << " initialized" << std::endl;

#elif defined(BACKEND_IX)
    // Initialize IX (uses CUDA API)
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set IX device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    device_ = at::Device(at::DeviceType::CUDA, device_id_);
    std::cout << "IX device " << device_id_ << " initialized" << std::endl;

#else  // CUDA
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    device_ = at::Device(at::DeviceType::CUDA, device_id_);
    std::cout << "CUDA device " << device_id_ << " initialized" << std::endl;
#endif

    initialized_ = true;
    return 0;
}

void DeviceManager::synchronize() {
    if (!initialized_) return;

#if defined(BACKEND_NPU)
    aclrtSynchronizeDevice();
#elif defined(BACKEND_MUSA)
    musaDeviceSynchronize();
#else  // CUDA or IX
    cudaDeviceSynchronize();
#endif
}

std::string DeviceManager::get_backend_name() const {
#if defined(BACKEND_NPU)
    return "NPU";
#elif defined(BACKEND_MUSA)
    return "MUSA";
#elif defined(BACKEND_IX)
    return "IX";
#else
    return "CUDA";
#endif
}

void DeviceManager::cleanup() {
    if (!initialized_) return;

#if defined(BACKEND_NPU)
    aclrtResetDevice(device_id_);
    aclFinalize();
#elif defined(BACKEND_MUSA)
    interpreter_.reset();
    musaDeviceReset();
#else  // CUDA or IX
    cudaDeviceReset();
#endif

    initialized_ = false;
}

// ============================================================================
// TensorFactory Implementation
// ============================================================================

TensorFactory::TensorFactory(const DeviceManager& dm) : device_manager_(dm) {
}

at::Tensor TensorFactory::rand(const std::vector<int64_t>& shape, at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    return allocate_tensor_musa(shape, dtype);
#else
    return at::rand(shape, options);
#endif
}

at::Tensor TensorFactory::rand(const std::vector<int64_t>& shape, double low, double high,
                               at::ScalarType dtype) {
    auto t = rand(shape, dtype);
    // Scale from [0, 1] to [low, high]
    return t * (high - low) + low;
}

at::Tensor TensorFactory::randint(const std::vector<int64_t>& shape, int64_t low, int64_t high,
                                  at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    // Create on CPU and copy to MUSA
    auto cpu_tensor = at::randint(low, high, shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_tensor.to(device_manager_.get_device());
#else
    return at::randint(low, high, shape, options);
#endif
}

at::Tensor TensorFactory::zeros(const std::vector<int64_t>& shape, at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    void* ptr = nullptr;
    size_t bytes = numel * at::elementSize(dtype);
    musaMalloc(&ptr, bytes);
    musaMemset(ptr, 0, bytes);

    return at::from_blob(ptr, shape, [](void* p) { musaFree(p); }, options);
#else
    return at::zeros(shape, options);
#endif
}

at::Tensor TensorFactory::ones(const std::vector<int64_t>& shape, at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    auto cpu_tensor = at::ones(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_tensor.to(device_manager_.get_device());
#else
    return at::ones(shape, options);
#endif
}

at::Tensor TensorFactory::empty(const std::vector<int64_t>& shape, at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    void* ptr = nullptr;
    size_t bytes = numel * at::elementSize(dtype);
    musaMalloc(&ptr, bytes);

    return at::from_blob(ptr, shape, [](void* p) { musaFree(p); }, options);
#else
    return at::empty(shape, options);
#endif
}

at::Tensor TensorFactory::full(const std::vector<int64_t>& shape, double value,
                               at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    auto cpu_tensor = at::full(shape, value, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_tensor.to(device_manager_.get_device());
#else
    return at::full(shape, value, options);
#endif
}

at::Tensor TensorFactory::arange(int64_t start, int64_t end, at::ScalarType dtype) {
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());

#if defined(BACKEND_MUSA)
    auto cpu_tensor = at::arange(start, end, at::TensorOptions().dtype(dtype).device(at::kCPU));
    return cpu_tensor.to(device_manager_.get_device());
#else
    return at::arange(start, end, options);
#endif
}

at::Tensor TensorFactory::allocate_tensor_musa(const std::vector<int64_t>& shape,
                                               at::ScalarType dtype) {
#if defined(BACKEND_MUSA)
    int64_t numel = 1;
    for (auto s : shape) numel *= s;

    void* ptr = nullptr;
    size_t bytes = numel * at::elementSize(dtype);
    musaMalloc(&ptr, bytes);

    // Fill with random values from CPU
    auto cpu_tensor = at::rand(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
    musaMemcpy(ptr, cpu_tensor.data_ptr(), bytes, musaMemcpyHostToDevice);

    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());
    return at::from_blob(ptr, shape, [](void* p) { musaFree(p); }, options);
#else
    // Fallback for other backends
    auto options = at::TensorOptions().dtype(dtype).device(device_manager_.get_device());
    return at::rand(shape, options);
#endif
}

// ============================================================================
// CorrectnessChecker Implementation
// ============================================================================

CorrectnessResult CorrectnessChecker::compare(const at::Tensor& actual,
                                              const at::Tensor& expected,
                                              double atol, double rtol) {
    return detailed_compare(actual, expected, atol, rtol);
}

bool CorrectnessChecker::allclose(const at::Tensor& actual,
                                  const at::Tensor& expected,
                                  double atol, double rtol) {
    if (!shapes_equal(actual, expected)) {
        return false;
    }

    at::Tensor a_cpu = actual.to(at::kCPU).contiguous().to(at::kFloat);
    at::Tensor e_cpu = expected.to(at::kCPU).contiguous().to(at::kFloat);

    return at::allclose(a_cpu, e_cpu, rtol, atol);
}

bool CorrectnessChecker::shapes_equal(const at::Tensor& a, const at::Tensor& b) {
    return a.sizes() == b.sizes();
}

bool CorrectnessChecker::dtypes_equal(const at::Tensor& a, const at::Tensor& b) {
    return a.dtype() == b.dtype();
}

CorrectnessResult CorrectnessChecker::detailed_compare(const at::Tensor& actual,
                                                       const at::Tensor& expected,
                                                       double atol, double rtol) {
    CorrectnessResult result;

    // Check shapes
    if (!shapes_equal(actual, expected)) {
        result.passed = false;
        std::stringstream ss;
        ss << "Shape mismatch: actual " << actual.sizes() << " vs expected " << expected.sizes();
        result.message = ss.str();
        return result;
    }

    // Convert to CPU and float for comparison
    at::Tensor a_cpu = actual.to(at::kCPU).contiguous().to(at::kFloat);
    at::Tensor e_cpu = expected.to(at::kCPU).contiguous().to(at::kFloat);

    result.total_elements = a_cpu.numel();

    // Calculate differences
    at::Tensor diff = at::abs(a_cpu - e_cpu);
    at::Tensor rel_diff = diff / (at::abs(e_cpu) + 1e-10);

    result.max_abs_diff = diff.max().item<double>();
    result.max_rel_diff = rel_diff.max().item<double>();

    // Count mismatches
    at::Tensor tolerance = atol + rtol * at::abs(e_cpu);
    at::Tensor mismatches = diff > tolerance;
    result.num_mismatches = mismatches.sum().item<int64_t>();

    // Determine pass/fail
    result.passed = (result.num_mismatches == 0);

    // Build message
    std::stringstream ss;
    if (result.passed) {
        ss << "PASSED (max_abs_diff=" << std::scientific << std::setprecision(3)
           << result.max_abs_diff << ", max_rel_diff=" << result.max_rel_diff << ")";
    } else {
        ss << "FAILED (" << result.num_mismatches << "/" << result.total_elements
           << " mismatches, max_abs_diff=" << std::scientific << std::setprecision(3)
           << result.max_abs_diff << ", max_rel_diff=" << result.max_rel_diff << ")";
    }
    result.message = ss.str();

    return result;
}

// ============================================================================
// TestRunner Implementation
// ============================================================================

TestRunner::TestRunner(DeviceManager& dm) : device_manager_(dm) {
}

void TestRunner::print_result(const CorrectnessResult& result) {
    std::cout << (result.passed ? "[PASS] " : "[FAIL] ") << result.message << std::endl;
    if (!result.passed) {
        std::cout << "  Max absolute difference: " << std::scientific << std::setprecision(6)
                  << result.max_abs_diff << std::endl;
        std::cout << "  Max relative difference: " << result.max_rel_diff << std::endl;
        std::cout << "  Mismatches: " << result.num_mismatches << "/" << result.total_elements
                  << std::endl;
    }
}

void TestRunner::print_result(const BenchmarkResult& result) {
    std::cout << "\n=== Benchmark: " << result.operator_name << " ===" << std::endl;
    std::cout << "  Warmup iterations:  " << result.warmup_iters << std::endl;
    std::cout << "  Benchmark iterations: " << result.bench_iters << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Mean latency:       " << result.mean_latency_us << " us" << std::endl;
    std::cout << "  Std latency:        " << result.std_latency_us << " us" << std::endl;
    std::cout << "  Min latency:        " << result.min_latency_us << " us" << std::endl;
    std::cout << "  Max latency:        " << result.max_latency_us << " us" << std::endl;
    if (result.throughput_gbps > 0) {
        std::cout << "  Throughput:         " << result.throughput_gbps << " GB/s" << std::endl;
    }
    if (result.tflops > 0) {
        std::cout << "  Performance:        " << result.tflops << " TFLOPS" << std::endl;
    }
}

} // namespace test
} // namespace triton_jit
