// ==============================================================================
// test_zeros.cpp - Multi-backend Triton JIT Zeros Operation Test
// ==============================================================================

#include "zeros_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_zeros_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: zeros_basic ===" << std::endl;

    constexpr int64_t SIZE = 128 * 1024;

    at::Tensor input = tf.rand({SIZE});
    at::Tensor result = my_ops::zeros_like(input);
    dm.synchronize();

    at::Tensor expected = tf.zeros({SIZE});
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-7, 1e-7);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "zeros_basic correctness check failed");
    return 0;
}

int test_zeros_shapes(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: zeros_shapes ===" << std::endl;

    std::vector<std::vector<int64_t>> shapes = {
        {1}, {1024}, {1024, 1024}, {32, 64, 128}
    };

    for (const auto& shape : shapes) {
        at::Tensor input = tf.rand(shape);
        at::Tensor result = my_ops::zeros_like(input);
        dm.synchronize();

        at::Tensor expected = tf.zeros(shape);
        dm.synchronize();

        CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-7, 1e-7);

        std::cout << "Shape [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        std::cout << "]: " << (cr.passed ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(cr.passed, "zeros_shapes failed for a shape");
    }

    return 0;
}

int test_zeros_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: zeros ===" << std::endl;

    constexpr int64_t SIZE = 1024 * 1024 * 16;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({SIZE});

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::zeros_like(input); },
        [&]() { dm.synchronize(); }
    );

    int64_t bytes = SIZE * sizeof(float);
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Triton JIT Zeros Operator Test Suite   " << std::endl;
    std::cout << "==========================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_zeros_basic(dm, tf));
    RUN_TEST(test_zeros_shapes(dm, tf));
    RUN_TEST(test_zeros_benchmark(dm, tf));

    std::cout << "\n==========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "==========================================" << std::endl;

    return 0;
}
