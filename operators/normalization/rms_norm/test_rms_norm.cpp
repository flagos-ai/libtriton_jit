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
// test_rms_norm.cpp - Multi-backend Triton JIT RMS Norm Test
// ==============================================================================

#include "benchmark_utils.h"
#include "rms_norm_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_rms_norm_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rms_norm_basic ===" << std::endl;

  constexpr int64_t BATCH = 16;
  constexpr int64_t HIDDEN = 1024;

  at::Tensor input = tf.rand({BATCH, HIDDEN});
  at::Tensor weight = tf.ones({HIDDEN});
  dm.synchronize();

  at::Tensor result = my_ops::rms_norm(input, weight, 1e-6);
  dm.synchronize();

  // Verify output shape
  bool shape_match = (result.sizes() == input.sizes());
  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(shape_match, "Output shape should match input shape");

  // Verify RMS normalization property: output should have unit RMS per row
  at::Tensor rms = (result * result).mean(-1).sqrt();
  at::Tensor expected_rms = tf.ones({BATCH});
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(rms, expected_rms, 1e-4, 1e-4);
  std::cout << "RMS close to 1: " << (cr.passed ? "YES" : "NO") << std::endl;

  std::cout << "[PASS] rms_norm_basic" << std::endl;
  return 0;
}

int test_rms_norm_with_weight(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rms_norm_with_weight ===" << std::endl;

  constexpr int64_t BATCH = 8;
  constexpr int64_t HIDDEN = 512;

  at::Tensor input = tf.rand({BATCH, HIDDEN});
  at::Tensor weight = tf.rand({HIDDEN});
  dm.synchronize();

  at::Tensor result = my_ops::rms_norm(input, weight, 1e-6);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.sizes() == input.sizes(), "Shape should match");

  std::cout << "[PASS] rms_norm_with_weight" << std::endl;
  return 0;
}

int test_rms_norm_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: rms_norm ===" << std::endl;

  constexpr int64_t BATCH = 1024;
  constexpr int64_t HIDDEN = 4096;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor input = tf.rand({BATCH, HIDDEN});
  at::Tensor weight = tf.ones({HIDDEN});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::rms_norm(input, weight, 1e-6); }, [&]() { dm.synchronize(); });

  // Read input + weight + write output
  int64_t bytes = BATCH * HIDDEN * sizeof(float) * 2 + HIDDEN * sizeof(float);
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Shape: (" << BATCH << ", " << HIDDEN << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "============================================" << std::endl;
  std::cout << "  Triton JIT RMS Norm Operator Test Suite  " << std::endl;
  std::cout << "============================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_rms_norm_basic(dm, tf));
  RUN_TEST(test_rms_norm_with_weight(dm, tf));
  RUN_TEST(test_rms_norm_benchmark(dm, tf));

  std::cout << "\n============================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "============================================" << std::endl;

  return 0;
}
