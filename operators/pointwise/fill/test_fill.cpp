// ==============================================================================
// test_fill.cpp - Multi-backend Triton JIT Fill Operation Test
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "fill_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"
#include "report_generator.h"

#include <iostream>
#include <cstdlib>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;
using namespace triton_jit::report;

// ==============================================================================
//                                TEST CASES
// ==============================================================================

int test_fill_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: fill_basic ===" << std::endl;

    constexpr int64_t SIZE = 128 * 1024;
    constexpr double VALUE = 3.14159;

    // Create input tensor
    at::Tensor input = tf.rand({SIZE});

    // Run fill operation
    at::Tensor result = my_ops::fill_tensor(input, VALUE);
    dm.synchronize();

    // Create expected result
    at::Tensor expected = tf.full({SIZE}, VALUE);
    dm.synchronize();

    // Verify
    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "fill_basic correctness check failed");
    return 0;
}

int test_fill_inplace(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: fill_inplace ===" << std::endl;

    constexpr int64_t SIZE = 64 * 1024;
    constexpr double VALUE = -2.718;

    // Create input tensor
    at::Tensor input = tf.rand({SIZE});

    // Run in-place fill
    my_ops::fill_tensor_(input, VALUE);
    dm.synchronize();

    // Create expected result
    at::Tensor expected = tf.full({SIZE}, VALUE);
    dm.synchronize();

    // Verify
    CorrectnessResult cr = CorrectnessChecker::compare(input, expected, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "fill_inplace correctness check failed");
    return 0;
}

int test_fill_shapes(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: fill_shapes ===" << std::endl;

    std::vector<std::vector<int64_t>> shapes = {
        {1},
        {1024},
        {1024, 1024},
        {32, 64, 128},
        {16, 16, 16, 16}
    };

    for (const auto& shape : shapes) {
        at::Tensor input = tf.rand(shape);
        at::Tensor result = my_ops::fill_tensor(input, 42.0);
        dm.synchronize();

        at::Tensor expected = tf.full(shape, 42.0);
        dm.synchronize();

        CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);

        std::cout << "Shape [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        std::cout << "]: " << (cr.passed ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(cr.passed, "fill_shapes failed for a shape");
    }

    return 0;
}

int test_fill_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: fill ===" << std::endl;

    constexpr int64_t SIZE = 1024 * 1024 * 16;  // 16M elements
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({SIZE});

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        my_ops::fill_tensor(input, 1.0);
    }
    dm.synchronize();

    // Benchmark
    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::fill_tensor(input, 1.0); },
        [&]() { dm.synchronize(); }
    );

    // Calculate metrics
    int64_t bytes = SIZE * sizeof(float);  // Only write, no read
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Mean latency:   " << stats.mean << " us" << std::endl;
    std::cout << "Std latency:    " << stats.std_dev << " us" << std::endl;
    std::cout << "Bandwidth:      " << bandwidth << " GB/s" << std::endl;

    return 0;
}

// ==============================================================================
//                                   MAIN
// ==============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Triton JIT Fill Operator Test Suite  " << std::endl;
    std::cout << "========================================" << std::endl;

    // Initialize device
    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;

    // Create tensor factory
    TensorFactory tf(dm);

    // Run tests
    int result = 0;

    RUN_TEST(test_fill_basic(dm, tf));
    RUN_TEST(test_fill_inplace(dm, tf));
    RUN_TEST(test_fill_shapes(dm, tf));
    RUN_TEST(test_fill_benchmark(dm, tf));

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return result;
}
