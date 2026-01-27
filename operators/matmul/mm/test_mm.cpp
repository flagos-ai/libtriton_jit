// ==============================================================================
// test_mm.cpp - Multi-backend Triton JIT MM Test
// ==============================================================================

#include "mm_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_mm_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: mm_basic ===" << std::endl;

    constexpr int64_t M = 512, K = 256, N = 1024;

    at::Tensor a = tf.rand({M, K});
    at::Tensor b = tf.rand({K, N});
    dm.synchronize();

    at::Tensor result = my_ops::mm(a, b);
    dm.synchronize();

    at::Tensor expected = at::mm(a, b);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "mm_basic correctness check failed");
    return 0;
}

int test_mm_shapes(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: mm_shapes ===" << std::endl;

    std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
        {128, 128, 128},
        {256, 512, 256},
        {1024, 256, 512},
        {512, 1024, 2048},
    };

    for (const auto& [M, K, N] : shapes) {
        at::Tensor a = tf.rand({M, K});
        at::Tensor b = tf.rand({K, N});
        dm.synchronize();

        at::Tensor result = my_ops::mm(a, b);
        dm.synchronize();

        at::Tensor expected = at::mm(a, b);
        dm.synchronize();

        CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
        std::cout << "Shape (" << M << ", " << K << ") x (" << K << ", " << N << "): "
                  << (cr.passed ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(cr.passed, "mm_shapes failed for a shape");
    }

    return 0;
}

int test_mm_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: mm ===" << std::endl;

    constexpr int64_t M = 2048, K = 2048, N = 2048;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor a = tf.rand({M, K});
    at::Tensor b = tf.rand({K, N});
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::mm(a, b); },
        [&]() { dm.synchronize(); }
    );

    // 2*M*N*K FLOPs for matmul
    int64_t flops = 2 * M * N * K;
    double tflops = calculate_tflops(flops, stats.mean);

    std::cout << "Matrix: (" << M << ", " << K << ") x (" << K << ", " << N << ")" << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "TFLOPS:       " << tflops << std::endl;

    return 0;
}

int main() {
    std::cout << "=======================================" << std::endl;
    std::cout << "  Triton JIT MM Operator Test Suite   " << std::endl;
    std::cout << "=======================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_mm_basic(dm, tf));
    RUN_TEST(test_mm_shapes(dm, tf));
    RUN_TEST(test_mm_benchmark(dm, tf));

    std::cout << "\n=======================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "=======================================" << std::endl;

    return 0;
}
