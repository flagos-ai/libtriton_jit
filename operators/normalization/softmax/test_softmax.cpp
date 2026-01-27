// ==============================================================================
// test_softmax.cpp - Multi-backend Triton JIT Softmax Test
// ==============================================================================

#include "softmax_op.h"
#include "test_framework.h"
#include "benchmark_utils.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_softmax_basic(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: softmax_basic ===" << std::endl;

    at::Tensor input = tf.rand({16, 1024});
    dm.synchronize();

    at::Tensor result = my_ops::softmax(input, -1);
    dm.synchronize();

    at::Tensor expected = at::softmax(input, -1);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-4, 1e-4);
    TestRunner::print_result(cr);

    TEST_ASSERT(cr.passed, "softmax_basic correctness check failed");

    // Verify softmax properties
    at::Tensor row_sums = result.sum(-1);
    bool sums_to_one = at::allclose(row_sums, at::ones_like(row_sums), 1e-4, 1e-4);
    std::cout << "Sums to 1: " << (sums_to_one ? "YES" : "NO") << std::endl;
    TEST_ASSERT(sums_to_one, "Softmax should sum to 1");

    return 0;
}

int test_softmax_dims(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Test: softmax_dims ===" << std::endl;

    at::Tensor input = tf.rand({8, 16, 32});
    dm.synchronize();

    for (int dim = 0; dim < 3; ++dim) {
        at::Tensor result = my_ops::softmax(input, dim);
        dm.synchronize();

        at::Tensor expected = at::softmax(input, dim);
        dm.synchronize();

        CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-4, 1e-4);
        std::cout << "dim=" << dim << ": " << (cr.passed ? "PASS" : "FAIL") << std::endl;

        TEST_ASSERT(cr.passed, "softmax_dims failed for dim");
    }

    return 0;
}

int test_softmax_benchmark(DeviceManager& dm, TensorFactory& tf) {
    std::cout << "\n=== Benchmark: softmax ===" << std::endl;

    constexpr int64_t BATCH = 1024;
    constexpr int64_t DIM = 4096;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    at::Tensor input = tf.rand({BATCH, DIM});
    dm.synchronize();

    BenchmarkRunner runner(WARMUP, ITERS);
    auto stats = runner.run(
        [&]() { my_ops::softmax(input, -1); },
        [&]() { dm.synchronize(); }
    );

    // Read input + write output
    int64_t bytes = BATCH * DIM * sizeof(float) * 2;
    double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

    std::cout << "Shape: (" << BATCH << ", " << DIM << ")" << std::endl;
    std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
    std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

    return 0;
}

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Triton JIT Softmax Operator Test Suite  " << std::endl;
    std::cout << "===========================================" << std::endl;

    DeviceManager dm;
    if (dm.initialize() != 0) {
        std::cerr << "Failed to initialize device" << std::endl;
        return -1;
    }

    std::cout << "Backend: " << dm.get_backend_name() << std::endl;
    TensorFactory tf(dm);

    RUN_TEST(test_softmax_basic(dm, tf));
    RUN_TEST(test_softmax_dims(dm, tf));
    RUN_TEST(test_softmax_benchmark(dm, tf));

    std::cout << "\n===========================================" << std::endl;
    std::cout << "  All tests passed!" << std::endl;
    std::cout << "===========================================" << std::endl;

    return 0;
}
