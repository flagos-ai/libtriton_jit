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
// test_bmm.cpp - Multi-backend Triton JIT BMM Test
// ==============================================================================

#include "benchmark_utils.h"
#include "bmm_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_bmm_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: bmm_basic ===" << std::endl;

  constexpr int64_t BATCH = 8;
  constexpr int64_t M = 64;
  constexpr int64_t K = 128;
  constexpr int64_t N = 64;

  at::Tensor a = tf.rand({BATCH, M, K});
  at::Tensor b = tf.rand({BATCH, K, N});
  dm.synchronize();

  at::Tensor result = my_ops::bmm(a, b);
  dm.synchronize();

  at::Tensor expected = at::bmm(a, b);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "bmm_basic correctness check failed");
  return 0;
}

int test_bmm_shapes(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: bmm_shapes ===" << std::endl;

  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> shapes = {
      { 1,  64,  64,  64},
      { 4, 128, 256, 128},
      {16,  32,  64,  32},
      {32,  64, 128,  64},
  };

  for (const auto& [B, M, K, N] : shapes) {
    at::Tensor a = tf.rand({B, M, K});
    at::Tensor b = tf.rand({B, K, N});
    dm.synchronize();

    at::Tensor result = my_ops::bmm(a, b);
    dm.synchronize();

    at::Tensor expected = at::bmm(a, b);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-3, 1e-3);
    std::cout << "Shape (" << B << ", " << M << ", " << K << ") x (" << B << ", " << K << ", " << N
              << "): " << (cr.passed ? "PASS" : "FAIL") << std::endl;

    TEST_ASSERT(cr.passed, "bmm_shapes failed");
  }

  return 0;
}

int test_bmm_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: bmm ===" << std::endl;

  constexpr int64_t BATCH = 32;
  constexpr int64_t M = 512;
  constexpr int64_t K = 512;
  constexpr int64_t N = 512;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor a = tf.rand({BATCH, M, K});
  at::Tensor b = tf.rand({BATCH, K, N});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::bmm(a, b); }, [&]() { dm.synchronize(); });

  int64_t flops = BATCH * 2 * M * N * K;
  double tflops = calculate_tflops(flops, stats.mean);

  std::cout << "Shape: (" << BATCH << ", " << M << ", " << K << ") x (" << BATCH << ", " << K << ", " << N
            << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "TFLOPS:       " << tflops << std::endl;

  return 0;
}

int main() {
  std::cout << "=======================================" << std::endl;
  std::cout << "  Triton JIT BMM Operator Test Suite  " << std::endl;
  std::cout << "=======================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_bmm_basic(dm, tf));
  RUN_TEST(test_bmm_shapes(dm, tf));
  RUN_TEST(test_bmm_benchmark(dm, tf));

  std::cout << "\n=======================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "=======================================" << std::endl;

  return 0;
}
