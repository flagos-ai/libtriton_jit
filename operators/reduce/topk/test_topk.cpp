// ==============================================================================
// test_topk.cpp - Multi-backend Triton JIT TopK Test
// ==============================================================================

#include "topk_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_topk_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: topk_basic ===" << std::endl;

    at::Tensor input = tf.rand({16, 1024});
    constexpr int64_t k = 10;
    dm.synchronize();

    auto [result_vals, result_idx] = my_ops::topk(input, k, 1);
    dm.synchronize();

    auto [expected_vals, expected_idx] = input.topk(k, 1);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result_vals, expected_vals, 1e-5, 1e-5);
    TestRunner::print_result(cr);
    TEST_ASSERT(cr.passed, "topk_basic values check failed");

    bool idx_match = result_idx.equal(expected_idx);
    std::cout << "Indices match: " << (idx_match ? "YES" : "NO") << std::endl;
    TEST_ASSERT(idx_match, "topk_basic indices check failed");

    return 0;
}

int test_topk_smallest(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: topk_smallest ===" << std::endl;

    at::Tensor input = tf.rand({8, 512});
    constexpr int64_t k = 5;
    dm.synchronize();

    auto [result_vals, result_idx] = my_ops::topk(input, k, 1, false);  // smallest
    dm.synchronize();

    auto [expected_vals, expected_idx] = input.topk(k, 1, false);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result_vals, expected_vals, 1e-5, 1e-5);
    TestRunner::print_result(cr);
    TEST_ASSERT(cr.passed, "topk_smallest values check failed");

    return 0;
}

int test_topk_shapes(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: topk_shapes ===" << std::endl;

    std::vector<std::pair<std::vector<int64_t>, int64_t>> test_cases = {
        {{1024}, 10},
        {{32, 256}, 8},
        {{8, 16, 128}, 5},
    };

    for (const auto& [shape, k] : test_cases) {
        at::Tensor input = tf.rand(shape);
        dm.synchronize();

        auto [result_vals, _] = my_ops::topk(input, k, -1);
        dm.synchronize();

        auto [expected_vals, __] = input.topk(k, -1);
        dm.synchronize();

        CorrectnessResult cr = CorrectnessChecker::compare(result_vals, expected_vals, 1e-5, 1e-5);

        std::cout << "Shape [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        }
        std::cout << "], k=" << k << ": " << (cr.passed ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(cr.passed, "topk_shapes failed for a shape");
    }

    return 0;
}

int test_topk_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: topk ===" << std::endl;

    constexpr int64_t M = 1024;
    constexpr int64_t N = 4096;
    constexpr int64_t K = 32;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({M, N});
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::topk(input, K, 1); },
        [&]() { dm.synchronize(); }
    );

    std::cout << "Matrix size: " << M << "x" << N << ", k=" << K << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Std latency:  " << stats.std_dev << " us" << std::endl;

    return 0;
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  Triton JIT TopK Operator Test Suite   " << std::endl;
    std::cout << "=========================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_topk_basic(dm, tf));
    RUN_TEST(test_topk_smallest(dm, tf));
    RUN_TEST(test_topk_shapes(dm, tf));
    RUN_TEST(test_topk_benchmark(dm, tf));

    std::cout << "\n=========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}
