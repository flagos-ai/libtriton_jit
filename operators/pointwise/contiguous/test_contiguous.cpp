// ==============================================================================
// test_contiguous.cpp - Multi-backend Triton JIT Contiguous Test
// ==============================================================================

#include "contiguous_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_contiguous_already_contiguous(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: contiguous_already_contiguous ===" << std::endl;

    at::Tensor input = tf.rand({64, 128});
    dm.synchronize();

    // Already contiguous should return same tensor
    at::Tensor result = my_ops::contiguous(input);
    dm.synchronize();

    // Check it's the same data
    bool same_ptr = (result.data_ptr() == input.data_ptr());
    std::cout << "Same data pointer: " << (same_ptr ? "YES" : "NO") << std::endl;
    TEST_ASSERT(same_ptr, "Already contiguous should return same tensor");

    std::cout << "[PASS] contiguous_already_contiguous" << std::endl;
    return 0;
}

int test_contiguous_transpose(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: contiguous_transpose ===" << std::endl;

    at::Tensor input = tf.rand({64, 128});
    at::Tensor transposed = input.t();  // Non-contiguous
    dm.synchronize();

    std::cout << "Original contiguous: " << input.is_contiguous() << std::endl;
    std::cout << "Transposed contiguous: " << transposed.is_contiguous() << std::endl;

    at::Tensor result = my_ops::contiguous(transposed);
    dm.synchronize();

    std::cout << "Result contiguous: " << result.is_contiguous() << std::endl;
    TEST_ASSERT(result.is_contiguous(), "Result should be contiguous");

    // Verify values match
    at::Tensor expected = transposed.contiguous();
    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "Values should match torch contiguous");
    return 0;
}

int test_contiguous_slice(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: contiguous_slice ===" << std::endl;

    at::Tensor input = tf.rand({128, 256});
    at::Tensor sliced = input.slice(1, 10, 200, 2);  // Non-contiguous due to stride
    dm.synchronize();

    std::cout << "Sliced contiguous: " << sliced.is_contiguous() << std::endl;

    at::Tensor result = my_ops::contiguous(sliced);
    dm.synchronize();

    std::cout << "Result contiguous: " << result.is_contiguous() << std::endl;
    TEST_ASSERT(result.is_contiguous(), "Result should be contiguous");

    at::Tensor expected = sliced.contiguous();
    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "Values should match torch contiguous");
    return 0;
}

int test_contiguous_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: contiguous ===" << std::endl;

    constexpr int64_t M = 4096;
    constexpr int64_t N = 4096;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({M, N});
    at::Tensor transposed = input.t();
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::contiguous(transposed); },
        [&]() { dm.synchronize(); }
    );

    int64_t bytes = M * N * sizeof(float) * 2;  // Read + write
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Matrix size: " << M << "x" << N << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "  Triton JIT Contiguous Operator Test Suite  " << std::endl;
    std::cout << "=============================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_contiguous_already_contiguous(dm, tf));
    RUN_TEST(test_contiguous_transpose(dm, tf));
    RUN_TEST(test_contiguous_slice(dm, tf));
    RUN_TEST(test_contiguous_benchmark(dm, tf));

    std::cout << "\n=============================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "=============================================" << std::endl;

    return 0;
}
