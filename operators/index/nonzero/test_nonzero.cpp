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
// test_nonzero.cpp - Multi-backend Triton JIT Nonzero Test
// ==============================================================================

#include "benchmark_utils.h"
#include "nonzero_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_nonzero_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: nonzero_basic ===" << std::endl;

  // Create tensor with some zeros and some non-zeros
  at::Tensor input = tf.rand({100});
  // Threshold to create sparse pattern
  input = (input > 0.5).to(at::kFloat);
  dm.synchronize();

  at::Tensor result = my_ops::nonzero(input);
  dm.synchronize();

  std::cout << "Input size: " << input.size(0) << std::endl;
  std::cout << "Output shape: " << result.sizes() << std::endl;

  // Reference
  at::Tensor expected = at::nonzero(input);
  dm.synchronize();

  // Check shapes match
  bool shape_match = (result.sizes() == expected.sizes());
  std::cout << "Shape match: " << (shape_match ? "YES" : "NO") << std::endl;

  if (shape_match && result.numel() > 0) {
    bool match = result.equal(expected);
    std::cout << "Values match: " << (match ? "YES" : "NO") << std::endl;
    TEST_ASSERT(match, "nonzero_basic values check failed");
  }

  std::cout << "[PASS] nonzero_basic" << std::endl;
  return 0;
}

int test_nonzero_2d(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: nonzero_2d ===" << std::endl;

  at::Tensor input = tf.rand({32, 64});
  input = (input > 0.7).to(at::kFloat);
  dm.synchronize();

  at::Tensor result = my_ops::nonzero(input);
  dm.synchronize();

  std::cout << "Input shape: " << input.sizes() << std::endl;
  std::cout << "Output shape: " << result.sizes() << std::endl;

  at::Tensor expected = at::nonzero(input);
  dm.synchronize();

  bool shape_match = (result.sizes() == expected.sizes());
  std::cout << "Shape match: " << (shape_match ? "YES" : "NO") << std::endl;

  if (shape_match && result.numel() > 0) {
    bool match = result.equal(expected);
    std::cout << "Values match: " << (match ? "YES" : "NO") << std::endl;
    TEST_ASSERT(match, "nonzero_2d values check failed");
  }

  std::cout << "[PASS] nonzero_2d" << std::endl;
  return 0;
}

int test_nonzero_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: nonzero ===" << std::endl;

  constexpr int64_t SIZE = 1024 * 1024;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor input = tf.rand({SIZE});
  input = (input > 0.9).to(at::kFloat);  // ~10% nonzero
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::nonzero(input); }, [&]() { dm.synchronize(); });

  std::cout << "Input size: " << SIZE << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Std latency:  " << stats.std_dev << " us" << std::endl;

  return 0;
}

int main() {
  std::cout << "==========================================" << std::endl;
  std::cout << "  Triton JIT Nonzero Operator Test Suite " << std::endl;
  std::cout << "==========================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_nonzero_basic(dm, tf));
  RUN_TEST(test_nonzero_2d(dm, tf));
  RUN_TEST(test_nonzero_benchmark(dm, tf));

  std::cout << "\n==========================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "==========================================" << std::endl;

  return 0;
}
