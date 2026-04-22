/**
 * @file test_framework.h
 * @brief Unified test framework for Triton JIT operators
 *
 * Provides device management, tensor factory, and correctness verification
 * for CUDA, MUSA, NPU, and IX backends.
 */

#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Backend-specific includes
#if defined(BACKEND_NPU)
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#define DEVICE_TYPE at::DeviceType::PrivateUse1
#elif defined(BACKEND_MUSA)
#include <musa_runtime.h>
#include <pybind11/embed.h>
namespace py = pybind11;
#define DEVICE_TYPE at::DeviceType::PrivateUse1
#elif defined(BACKEND_MLU)
#include <cnrt.h>
#include <pybind11/embed.h>
namespace py = pybind11;
#define DEVICE_TYPE at::DeviceType::PrivateUse1
#elif defined(BACKEND_IX)
#include <cuda.h>
#include <cuda_runtime.h>
#define DEVICE_TYPE at::DeviceType::CUDA
#elif defined(BACKEND_HCU)
#include <hip/hip_runtime.h>
#define DEVICE_TYPE at::DeviceType::CUDA
#else  // CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#define DEVICE_TYPE at::DeviceType::CUDA
#endif

namespace triton_jit {
namespace test {

  /**
   * @brief Result of correctness verification
   */
  struct CorrectnessResult {
    bool passed;             // Overall pass/fail status
    double max_abs_diff;     // Maximum absolute difference
    double max_rel_diff;     // Maximum relative difference
    int64_t num_mismatches;  // Number of elements that don't match
    int64_t total_elements;  // Total elements compared
    std::string message;     // Descriptive message

    CorrectnessResult()
        : passed(false), max_abs_diff(0), max_rel_diff(0), num_mismatches(0), total_elements(0), message("") {
    }
  };

  /**
   * @brief Performance benchmark result
   */
  struct BenchmarkResult {
    double mean_latency_us;     // Mean latency in microseconds
    double std_latency_us;      // Standard deviation
    double min_latency_us;      // Minimum latency
    double max_latency_us;      // Maximum latency
    double throughput_gbps;     // Throughput in GB/s
    double tflops;              // Tera FLOPS
    int warmup_iters;           // Warmup iterations
    int bench_iters;            // Benchmark iterations
    std::string operator_name;  // Name of the operator

    BenchmarkResult()
        : mean_latency_us(0),
          std_latency_us(0),
          min_latency_us(0),
          max_latency_us(0),
          throughput_gbps(0),
          tflops(0),
          warmup_iters(0),
          bench_iters(0),
          operator_name("") {
    }
  };

  /**
   * @brief Test case configuration
   */
  struct TestConfig {
    std::vector<int64_t> input_shape;
    at::ScalarType dtype;
    double atol;  // Absolute tolerance
    double rtol;  // Relative tolerance
    int warmup_iters;
    int bench_iters;
    std::string test_name;

    TestConfig()
        : dtype(at::kFloat), atol(1e-5), rtol(1e-5), warmup_iters(10), bench_iters(100), test_name("") {
    }
  };

  /**
   * @brief Device manager for multi-backend support
   */
  class DeviceManager {
   public:
    DeviceManager();
    ~DeviceManager();

    /**
     * @brief Initialize device
     * @param device_id Device ID to use
     * @return 0 on success, non-zero on failure
     */
    int initialize(int device_id = 0);

    /**
     * @brief Synchronize device
     */
    void synchronize();

    /**
     * @brief Get PyTorch device
     */
    at::Device get_device() const {
      return device_;
    }

    /**
     * @brief Get device ID
     */
    int get_device_id() const {
      return device_id_;
    }

    /**
     * @brief Get backend name
     */
    std::string get_backend_name() const;

    /**
     * @brief Check if device is initialized
     */
    bool is_initialized() const {
      return initialized_;
    }

    /**
     * @brief Cleanup device resources
     */
    void cleanup();

   private:
    at::Device device_;
    int device_id_;
    bool initialized_;

#if defined(BACKEND_MUSA) || defined(BACKEND_MLU)
    std::unique_ptr<py::scoped_interpreter> interpreter_;
#endif
  };

  /**
   * @brief Tensor factory for creating test tensors
   */
  class TensorFactory {
   public:
    explicit TensorFactory(const DeviceManager& dm);

    /**
     * @brief Create a random tensor
     */
    at::Tensor rand(const std::vector<int64_t>& shape, at::ScalarType dtype = at::kFloat);

    /**
     * @brief Create a random tensor in specified range
     */
    at::Tensor rand(const std::vector<int64_t>& shape,
                    double low,
                    double high,
                    at::ScalarType dtype = at::kFloat);

    /**
     * @brief Create a random integer tensor
     */
    at::Tensor randint(const std::vector<int64_t>& shape,
                       int64_t low,
                       int64_t high,
                       at::ScalarType dtype = at::kLong);

    /**
     * @brief Create a zeros tensor
     */
    at::Tensor zeros(const std::vector<int64_t>& shape, at::ScalarType dtype = at::kFloat);

    /**
     * @brief Create an ones tensor
     */
    at::Tensor ones(const std::vector<int64_t>& shape, at::ScalarType dtype = at::kFloat);

    /**
     * @brief Create an empty tensor
     */
    at::Tensor empty(const std::vector<int64_t>& shape, at::ScalarType dtype = at::kFloat);

    /**
     * @brief Create a tensor filled with a value
     */
    at::Tensor full(const std::vector<int64_t>& shape, double value, at::ScalarType dtype = at::kFloat);

    /**
     * @brief Create a range tensor (arange)
     */
    at::Tensor arange(int64_t start, int64_t end, at::ScalarType dtype = at::kLong);

   private:
    const DeviceManager& device_manager_;

    // Helper for MUSA backend manual allocation
    at::Tensor allocate_tensor_musa(const std::vector<int64_t>& shape, at::ScalarType dtype);
  };

  /**
   * @brief Correctness checker for comparing tensors
   */
  class CorrectnessChecker {
   public:
    /**
     * @brief Compare two tensors with tolerances
     */
    static CorrectnessResult compare(const at::Tensor& actual,
                                     const at::Tensor& expected,
                                     double atol = 1e-5,
                                     double rtol = 1e-5);

    /**
     * @brief Check if tensors are close using allclose
     */
    static bool allclose(const at::Tensor& actual,
                         const at::Tensor& expected,
                         double atol = 1e-5,
                         double rtol = 1e-5);

    /**
     * @brief Check tensor shape equality
     */
    static bool shapes_equal(const at::Tensor& a, const at::Tensor& b);

    /**
     * @brief Check tensor dtype equality
     */
    static bool dtypes_equal(const at::Tensor& a, const at::Tensor& b);

    /**
     * @brief Get detailed difference statistics
     */
    static CorrectnessResult detailed_compare(const at::Tensor& actual,
                                              const at::Tensor& expected,
                                              double atol = 1e-5,
                                              double rtol = 1e-5);
  };

  /**
   * @brief Test runner for executing test cases
   */
  class TestRunner {
   public:
    explicit TestRunner(DeviceManager& dm);

    /**
     * @brief Run a correctness test
     */
    template <typename OpFunc, typename RefFunc>
    CorrectnessResult run_correctness_test(const TestConfig& config, OpFunc&& op_func, RefFunc&& ref_func);

    /**
     * @brief Run a benchmark test
     */
    template <typename OpFunc>
    BenchmarkResult run_benchmark(const TestConfig& config,
                                  OpFunc&& op_func,
                                  int64_t bytes_accessed = 0,
                                  int64_t flop_count = 0);

    /**
     * @brief Print test result
     */
    static void print_result(const CorrectnessResult& result);

    /**
     * @brief Print benchmark result
     */
    static void print_result(const BenchmarkResult& result);

   private:
    DeviceManager& device_manager_;
  };

  // ============================================================================
  // Template implementations
  // ============================================================================

  template <typename OpFunc, typename RefFunc>
  CorrectnessResult TestRunner::run_correctness_test(const TestConfig& config,
                                                     OpFunc&& op_func,
                                                     RefFunc&& ref_func) {
    CorrectnessResult result;
    result.message = config.test_name;

    try {
      // Run the operator being tested
      at::Tensor actual = op_func();
      device_manager_.synchronize();

      // Run the reference implementation
      at::Tensor expected = ref_func();
      device_manager_.synchronize();

      // Compare results
      result = CorrectnessChecker::detailed_compare(actual, expected, config.atol, config.rtol);
      result.message = config.test_name + ": " + result.message;

    } catch (const std::exception& e) {
      result.passed = false;
      result.message = config.test_name + ": Exception - " + e.what();
    }

    return result;
  }

  template <typename OpFunc>
  BenchmarkResult TestRunner::run_benchmark(const TestConfig& config,
                                            OpFunc&& op_func,
                                            int64_t bytes_accessed,
                                            int64_t flop_count) {
    BenchmarkResult result;
    result.operator_name = config.test_name;
    result.warmup_iters = config.warmup_iters;
    result.bench_iters = config.bench_iters;

    try {
      // Warmup
      for (int i = 0; i < config.warmup_iters; ++i) {
        op_func();
      }
      device_manager_.synchronize();

      // Benchmark
      std::vector<double> latencies;
      latencies.reserve(config.bench_iters);

      for (int i = 0; i < config.bench_iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        op_func();
        device_manager_.synchronize();
        auto end = std::chrono::high_resolution_clock::now();

        double latency_us = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(latency_us);
      }

      // Calculate statistics
      double sum = 0, sum_sq = 0;
      double min_lat = latencies[0], max_lat = latencies[0];

      for (double lat : latencies) {
        sum += lat;
        sum_sq += lat * lat;
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
      }

      result.mean_latency_us = sum / latencies.size();
      result.std_latency_us =
          std::sqrt(sum_sq / latencies.size() - result.mean_latency_us * result.mean_latency_us);
      result.min_latency_us = min_lat;
      result.max_latency_us = max_lat;

      // Calculate throughput
      if (bytes_accessed > 0 && result.mean_latency_us > 0) {
        result.throughput_gbps = (bytes_accessed / 1e9) / (result.mean_latency_us / 1e6);
      }

      // Calculate TFLOPS
      if (flop_count > 0 && result.mean_latency_us > 0) {
        result.tflops = (flop_count / 1e12) / (result.mean_latency_us / 1e6);
      }

    } catch (const std::exception& e) {
      std::cerr << "Benchmark error: " << e.what() << std::endl;
    }

    return result;
  }

  // ============================================================================
  // Utility macros
  // ============================================================================

#define TEST_ASSERT(cond, msg)                                                                      \
  if (!(cond)) {                                                                                    \
    std::cerr << "ASSERTION FAILED: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    return 1;                                                                                       \
  }

#define TEST_ASSERT_NEAR(a, b, tol, msg)                                                               \
  if (std::abs((a) - (b)) > (tol)) {                                                                   \
    std::cerr << "ASSERTION FAILED: " << msg << " (" << a << " vs " << b << ") at " << __FILE__ << ":" \
              << __LINE__ << std::endl;                                                                \
    return 1;                                                                                          \
  }

#define RUN_TEST(test_expr)                                      \
  {                                                              \
    std::cout << "Running " << #test_expr << "..." << std::endl; \
    int result = (test_expr);                                    \
    if (result != 0) {                                           \
      std::cerr << "FAILED: " << #test_expr << std::endl;        \
      return result;                                             \
    }                                                            \
    std::cout << "PASSED: " << #test_expr << std::endl;          \
  }

}  // namespace test
}  // namespace triton_jit
