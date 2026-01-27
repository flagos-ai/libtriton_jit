// ==============================================================================
// test_max.cpp - Multi-backend Triton JIT Max Operation Test
// ==============================================================================

#include "max_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_max_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: max_basic ===" << std::endl;

    at::Tensor input = tf.rand({16, 4096});
    dm.synchronize();

    auto [result_vals, result_idx] = my_ops::max_dim(input, 1);
    dm.synchronize();

    // Reference
    auto [expected_vals, expected_idx] = input.max(1);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result_vals, expected_vals, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "max_basic values correctness check failed");

    // Check indices
    bool idx_match = result_idx.equal(expected_idx);
    std::cout << "Indices match: " << (idx_match ? "YES" : "NO") << std::endl;
    TEST_ASSERT(idx_match, "max_basic indices check failed");

    return 0;
}

int test_max_keepdim(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: max_keepdim ===" << std::endl;

    at::Tensor input = tf.rand({8, 32, 64});
    dm.synchronize();

    auto [result_vals, result_idx] = my_ops::max_dim(input, 1, true);
    dm.synchronize();

    auto [expected_vals, expected_idx] = input.max(1, true);
    dm.synchronize();

    // Check shapes
    bool shape_match = (result_vals.sizes() == expected_vals.sizes());
    std::cout << "Output shape: " << result_vals.sizes() << std::endl;
    std::cout << "Expected shape: " << expected_vals.sizes() << std::endl;
    TEST_ASSERT(shape_match, "Shapes should match with keepdim=true");

    CorrectnessResult cr = CorrectnessChecker::compare(result_vals, expected_vals, 1e-5, 1e-5);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "max_keepdim correctness check failed");
    return 0;
}

int test_max_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: max ===" << std::endl;

    constexpr int64_t M = 1024;
    constexpr int64_t N = 4096;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({M, N});
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::max_dim(input, 1); },
        [&]() { dm.synchronize(); }
    );

    int64_t bytes = M * N * sizeof(float) + M * (sizeof(float) + sizeof(int64_t));
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Matrix size: " << M << "x" << N << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Triton JIT Max Operator Test Suite   " << std::endl;
    std::cout << "========================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_max_basic(dm, tf));
    RUN_TEST(test_max_keepdim(dm, tf));
    RUN_TEST(test_max_benchmark(dm, tf));

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
