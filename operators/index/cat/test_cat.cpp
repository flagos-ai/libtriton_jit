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
// test_cat.cpp - Multi-backend Triton JIT Cat Test
// ==============================================================================

#include "benchmark_utils.h"
#include "cat_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_cat_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: cat_basic ===" << std::endl;

  at::Tensor a = tf.rand({16, 32});
  at::Tensor b = tf.rand({16, 64});
  dm.synchronize();

  at::Tensor result = my_ops::cat({a, b}, 1);
  dm.synchronize();

  // Verify output shape
  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.size(0) == 16 && result.size(1) == 96, "Output shape should be [16, 96]");

  // Reference
  at::Tensor expected = at::cat({a, b}, 1);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "cat_basic correctness check failed");
  return 0;
}

int test_cat_dim0(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: cat_dim0 ===" << std::endl;

  at::Tensor a = tf.rand({8, 64});
  at::Tensor b = tf.rand({16, 64});
  at::Tensor c = tf.rand({4, 64});
  dm.synchronize();

  at::Tensor result = my_ops::cat({a, b, c}, 0);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.size(0) == 28 && result.size(1) == 64, "Output shape should be [28, 64]");

  at::Tensor expected = at::cat({a, b, c}, 0);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "cat_dim0 correctness check failed");
  return 0;
}

int test_cat_3d(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: cat_3d ===" << std::endl;

  at::Tensor a = tf.rand({4, 8, 16});
  at::Tensor b = tf.rand({4, 8, 32});
  dm.synchronize();

  at::Tensor result = my_ops::cat({a, b}, 2);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.size(2) == 48, "Last dim should be concatenated");

  at::Tensor expected = at::cat({a, b}, 2);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "cat_3d correctness check failed");
  return 0;
}

int test_cat_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: cat ===" << std::endl;

  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor a = tf.rand({1024, 512});
  at::Tensor b = tf.rand({1024, 512});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::cat({a, b}, 1); }, [&]() { dm.synchronize(); });

  int64_t bytes = 1024 * 1024 * sizeof(float) * 2;  // read both + write result
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "=======================================" << std::endl;
  std::cout << "  Triton JIT Cat Operator Test Suite  " << std::endl;
  std::cout << "=======================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_cat_basic(dm, tf));
  RUN_TEST(test_cat_dim0(dm, tf));
  RUN_TEST(test_cat_3d(dm, tf));
  RUN_TEST(test_cat_benchmark(dm, tf));

  std::cout << "\n=======================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "=======================================" << std::endl;

  return 0;
}
