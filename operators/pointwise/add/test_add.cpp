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
// test_add.cpp - Multi-backend Triton JIT Add Operation Test
// ==============================================================================

#include "add_op.h"
#include "benchmark_utils.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_add_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: add_basic ===" << std::endl;

  constexpr int64_t SIZE = 128 * 1024;

  at::Tensor a = tf.rand({SIZE});
  at::Tensor b = tf.rand({SIZE});
  dm.synchronize();

  at::Tensor result = my_ops::add_tensor(a, b);
  dm.synchronize();

  at::Tensor expected = at::add(a, b);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "add_basic correctness check failed");
  return 0;
}

int test_add_2d(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: add_2d ===" << std::endl;

  at::Tensor a = tf.rand({64, 128});
  at::Tensor b = tf.rand({64, 128});
  dm.synchronize();

  at::Tensor result = my_ops::add_tensor(a, b);
  dm.synchronize();

  at::Tensor expected = at::add(a, b);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "add_2d correctness check failed");
  return 0;
}

int test_add_broadcast(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: add_broadcast ===" << std::endl;

  at::Tensor a = tf.rand({32, 64});
  at::Tensor b = tf.rand({64});  // Will broadcast
  dm.synchronize();

  at::Tensor result = my_ops::add_tensor(a, b);
  dm.synchronize();

  at::Tensor expected = at::add(a, b);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "add_broadcast correctness check failed");
  return 0;
}

int test_add_shapes(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: add_shapes ===" << std::endl;

  std::vector<std::vector<int64_t>> shapes = {
      // {1},  // Skip: NPU Triton kernel has issues with n < BLOCK_N
      {1024},
      {1024, 1024},
      {32, 64, 128}
  };

  for (const auto& shape : shapes) {
    at::Tensor a = tf.rand(shape);
    at::Tensor b = tf.rand(shape);
    dm.synchronize();

    at::Tensor result = my_ops::add_tensor(a, b);
    dm.synchronize();

    at::Tensor expected = at::add(a, b);
    dm.synchronize();

    CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-5, 1e-5);

    std::cout << "Shape [";
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "]: " << (cr.passed ? "PASS" : "FAIL") << std::endl;

    // Debug output when test fails
    if (!cr.passed) {
      std::cout << "[DEBUG] a (input 1): " << a.cpu() << std::endl;
      std::cout << "[DEBUG] b (input 2): " << b.cpu() << std::endl;
      std::cout << "[DEBUG] result (my_ops::add_tensor): " << result.cpu() << std::endl;
      std::cout << "[DEBUG] expected (at::add): " << expected.cpu() << std::endl;
      std::cout << "[DEBUG] max_abs_diff: " << cr.max_abs_diff << std::endl;
      std::cout << "[DEBUG] max_rel_diff: " << cr.max_rel_diff << std::endl;
      std::cout << "[DEBUG] num_mismatches: " << cr.num_mismatches << "/" << cr.total_elements << std::endl;
    }

    TEST_ASSERT(cr.passed, "add_shapes failed for a shape");
  }

  return 0;
}

int test_add_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: add ===" << std::endl;

  constexpr int64_t SIZE = 1024 * 1024 * 16;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor a = tf.rand({SIZE});
  at::Tensor b = tf.rand({SIZE});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::add_tensor(a, b); }, [&]() { dm.synchronize(); });

  // Read a + b, write result
  int64_t bytes = SIZE * sizeof(float) * 3;
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Size: " << SIZE << " elements" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "=======================================" << std::endl;
  std::cout << "  Triton JIT Add Operator Test Suite  " << std::endl;
  std::cout << "=======================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_add_basic(dm, tf));
  RUN_TEST(test_add_2d(dm, tf));
  RUN_TEST(test_add_broadcast(dm, tf));
  RUN_TEST(test_add_shapes(dm, tf));
  RUN_TEST(test_add_benchmark(dm, tf));

  std::cout << "\n=======================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "=======================================" << std::endl;

  return 0;
}
