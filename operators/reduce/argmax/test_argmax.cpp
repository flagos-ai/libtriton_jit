// ==============================================================================
// test_argmax.cpp - Multi-backend Triton JIT Argmax Test
// ==============================================================================

#include "argmax_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_argmax_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: argmax_basic ===" << std::endl;

    at::Tensor input = tf.rand({16, 4096});
    dm.synchronize();

    at::Tensor result = my_ops::argmax(input, 1);
    dm.synchronize();

    at::Tensor expected = input.argmax(1);
    dm.synchronize();

    bool match = result.equal(expected);
    std::cout << "Indices match: " << (match ? "YES" : "NO") << std::endl;

    TEST_ASSERT(match, "argmax_basic correctness check failed");
    return 0;
}

int test_argmax_keepdim(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: argmax_keepdim ===" << std::endl;

    at::Tensor input = tf.rand({8, 32, 64});
    dm.synchronize();

    at::Tensor result = my_ops::argmax(input, 1, true);
    dm.synchronize();

    at::Tensor expected = input.argmax(1, true);
    dm.synchronize();

    bool shape_match = (result.sizes() == expected.sizes());
    std::cout << "Output shape: " << result.sizes() << std::endl;
    TEST_ASSERT(shape_match, "Shapes should match");

    bool match = result.equal(expected);
    std::cout << "Indices match: " << (match ? "YES" : "NO") << std::endl;

    TEST_ASSERT(match, "argmax_keepdim correctness check failed");
    return 0;
}

int test_argmax_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: argmax ===" << std::endl;

    constexpr int64_t M = 1024;
    constexpr int64_t N = 4096;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({M, N});
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::argmax(input, 1); },
        [&]() { dm.synchronize(); }
    );

    int64_t bytes = M * N * sizeof(float) + M * sizeof(int64_t);
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Matrix size: " << M << "x" << N << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Triton JIT Argmax Operator Test Suite  " << std::endl;
    std::cout << "==========================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_argmax_basic(dm, tf));
    RUN_TEST(test_argmax_keepdim(dm, tf));
    RUN_TEST(test_argmax_benchmark(dm, tf));

    std::cout << "\n==========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}
