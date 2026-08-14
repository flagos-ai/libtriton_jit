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
// test_sum.cpp - Multi-backend Triton JIT Sum Operation Test
// ==============================================================================

#include "benchmark_utils.h"
#include "sum_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_sum_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: sum_basic ===" << std::endl;

  at::Tensor input = tf.rand({16, 4096});
  dm.synchronize();

  at::Tensor result = my_ops::sum_dim(input, {1}, false, c10::nullopt);
  dm.synchronize();

  // Reference
  at::Tensor expected = at::sum(input, {1}, false, c10::nullopt);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-4, 1e-4);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "sum_basic correctness check failed");
  return 0;
}

int test_sum_keepdim(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: sum_keepdim ===" << std::endl;

  at::Tensor input = tf.rand({8, 32, 64});
  dm.synchronize();

  at::Tensor result = my_ops::sum_dim(input, {1}, true, c10::nullopt);
  dm.synchronize();

  at::Tensor expected = at::sum(input, {1}, true, c10::nullopt);
  dm.synchronize();

  // Check shapes
  bool shape_match = (result.sizes() == expected.sizes());
  std::cout << "Output shape: " << result.sizes() << std::endl;
  std::cout << "Expected shape: " << expected.sizes() << std::endl;
  TEST_ASSERT(shape_match, "Shapes should match with keepdim=true");

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-4, 1e-4);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "sum_keepdim correctness check failed");
  return 0;
}

int test_sum_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: sum ===" << std::endl;

  constexpr int64_t M = 1024;
  constexpr int64_t N = 4096;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor input = tf.rand({M, N});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats =
      runner.run([&]() { my_ops::sum_dim(input, {1}, false, c10::nullopt); }, [&]() { dm.synchronize(); });

  int64_t bytes = M * N * sizeof(float) + M * sizeof(float);
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Matrix size: " << M << "x" << N << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "  Triton JIT Sum Operator Test Suite   " << std::endl;
  std::cout << "========================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_sum_basic(dm, tf));
  RUN_TEST(test_sum_keepdim(dm, tf));
  RUN_TEST(test_sum_benchmark(dm, tf));

  std::cout << "\n========================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
