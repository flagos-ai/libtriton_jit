// ==============================================================================
// test_exponential_.cpp - Multi-backend Triton JIT Exponential_ Test
// ==============================================================================

#include "exponential_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>
#include <cmath>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_exponential_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: exponential_basic ===" << std::endl;

    constexpr int64_t SIZE = 128 * 1024;
    constexpr double LAMBD = 1.0;

    at::Tensor input = tf.rand({SIZE});
    my_ops::exponential_(input, LAMBD);
    dm.synchronize();

    // Verify basic properties
    at::Tensor cpu_result = input.to(at::kCPU);

    // All values should be positive
    bool all_positive = (cpu_result > 0).all().item<bool>();
    std::cout << "All positive: " << (all_positive ? "YES" : "NO") << std::endl;
    TEST_ASSERT(all_positive, "Exponential values should all be positive");

    // Mean should be approximately 1/lambda
    double mean = cpu_result.mean().item<double>();
    double expected_mean = 1.0 / LAMBD;
    std::cout << "Mean: " << mean << " (expected ~" << expected_mean << ")" << std::endl;

    // Allow 10% tolerance for statistical test
    bool mean_close = std::abs(mean - expected_mean) < 0.1 * expected_mean;
    TEST_ASSERT(mean_close, "Mean should be close to 1/lambda");

    std::cout << "[PASS] exponential_basic" << std::endl;
    return 0;
}

int test_exponential_lambda(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: exponential_lambda ===" << std::endl;

    constexpr int64_t SIZE = 256 * 1024;
    std::vector<double> lambdas = {0.5, 1.0, 2.0};

    for (double lambd : lambdas) {
        at::Tensor input = tf.rand({SIZE});
        my_ops::exponential_(input, lambd);
        dm.synchronize();

        at::Tensor cpu_result = input.to(at::kCPU);
        double mean = cpu_result.mean().item<double>();
        double expected_mean = 1.0 / lambd;

        bool mean_close = std::abs(mean - expected_mean) < 0.15 * expected_mean;
        std::cout << "Lambda=" << lambd << ": mean=" << mean
                  << " (expected ~" << expected_mean << ") "
                  << (mean_close ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(mean_close, "Mean should match expected for lambda");
    }

    return 0;
}

int test_exponential_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: exponential_ ===" << std::endl;

    constexpr int64_t SIZE = 1024 * 1024 * 16;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({SIZE});

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::exponential_(input, 1.0); },
        [&]() { dm.synchronize(); }
    );

    // Read + write
    int64_t bytes = SIZE * sizeof(float) * 2;
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "  Triton JIT Exponential_ Operator Test Suite   " << std::endl;
    std::cout << "================================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_exponential_basic(dm, tf));
    RUN_TEST(test_exponential_lambda(dm, tf));
    RUN_TEST(test_exponential_benchmark(dm, tf));

    std::cout << "\n================================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}
