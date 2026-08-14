// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// ==============================================================================
// test_addmm.cpp - Multi-backend Triton JIT Addmm Test
// ==============================================================================

#include "addmm_op.h"
#include "benchmark_utils.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_addmm_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: addmm_basic ===" << std::endl;

  constexpr int64_t M = 64;
  constexpr int64_t K = 128;
  constexpr int64_t N = 64;

  at::Tensor input = tf.rand({M, N});
  at::Tensor a = tf.rand({M, K});
  at::Tensor b = tf.rand({K, N});
  dm.synchronize();

  at::Tensor result = my_ops::addmm(input, a, b, 0.5, 1.0);
  dm.synchronize();

  at::Tensor expected = at::addmm(input, a, b, 0.5, 1.0);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "addmm_basic correctness check failed");
  return 0;
}

int test_addmm_alpha_beta(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: addmm_alpha_beta ===" << std::endl;

  constexpr int64_t M = 128;
  constexpr int64_t K = 256;
  constexpr int64_t N = 128;

  at::Tensor input = tf.rand({M, N});
  at::Tensor a = tf.rand({M, K});
  at::Tensor b = tf.rand({K, N});
  dm.synchronize();

  std::vector<std::pair<double, double>> alpha_betas = {
      {1.0, 1.0},
      {0.5, 0.5},
      {2.0, 0.0},
      {0.0, 1.0}
  };

  for (const auto& [alpha, beta] : alpha_betas) {
    at::Tensor result = my_ops::addmm(input, a, b, beta, alpha);
    dm.synchronize();

    at::Tensor expected = at::addmm(input, a, b, beta, alpha);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
    std::cout << "alpha=" << alpha << ", beta=" << beta << ": " << (cr.passed ? "PASS" : "FAIL") << std::endl;

    TEST_ASSERT(cr.passed, "addmm_alpha_beta failed");
  }

  return 0;
}

int test_addmm_shapes(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: addmm_shapes ===" << std::endl;

  std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
      { 64,  64,  64},
      {128, 256, 128},
      {256, 512, 256},
  };

  for (const auto& [M, K, N] : shapes) {
    at::Tensor input = tf.rand({M, N});
    at::Tensor a = tf.rand({M, K});
    at::Tensor b = tf.rand({K, N});
    dm.synchronize();

    at::Tensor result = my_ops::addmm(input, a, b, 1.0, 1.0);
    dm.synchronize();

    at::Tensor expected = at::addmm(input, a, b, 1.0, 1.0);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
    std::cout << "Shape (" << M << ", " << K << ", " << N << "): " << (cr.passed ? "PASS" : "FAIL")
              << std::endl;

    TEST_ASSERT(cr.passed, "addmm_shapes failed");
  }

  return 0;
}

int test_addmm_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: addmm ===" << std::endl;

  constexpr int64_t M = 1024;
  constexpr int64_t K = 1024;
  constexpr int64_t N = 1024;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor input = tf.rand({M, N});
  at::Tensor a = tf.rand({M, K});
  at::Tensor b = tf.rand({K, N});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::addmm(input, a, b, 1.0, 1.0); }, [&]() { dm.synchronize(); });

  int64_t flops = 2 * M * N * K + 2 * M * N;  // matmul + add + scale
  double tflops = calculate_tflops(flops, stats.mean);

  std::cout << "Shape: (" << M << ", " << K << ") x (" << K << ", " << N << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "TFLOPS:       " << tflops << std::endl;

  return 0;
}

int main() {
  std::cout << "=========================================" << std::endl;
  std::cout << "  Triton JIT Addmm Operator Test Suite  " << std::endl;
  std::cout << "=========================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_addmm_basic(dm, tf));
  RUN_TEST(test_addmm_alpha_beta(dm, tf));
  RUN_TEST(test_addmm_shapes(dm, tf));
  RUN_TEST(test_addmm_benchmark(dm, tf));

  std::cout << "\n=========================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "=========================================" << std::endl;

  return 0;
}
