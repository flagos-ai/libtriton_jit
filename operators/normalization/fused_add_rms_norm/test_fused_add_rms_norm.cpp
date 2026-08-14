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
// test_fused_add_rms_norm.cpp - Multi-backend Fused Add RMS Norm Test
// ==============================================================================

#include "benchmark_utils.h"
#include "fused_add_rms_norm_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_fused_add_rms_norm_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: fused_add_rms_norm_basic ===" << std::endl;

  constexpr int64_t BATCH = 16;
  constexpr int64_t HIDDEN = 1024;

  at::Tensor input = tf.rand({BATCH, HIDDEN});
  at::Tensor residual = tf.rand({BATCH, HIDDEN});
  at::Tensor weight = tf.ones({HIDDEN});
  dm.synchronize();

  auto [output, res_out] = my_ops::fused_add_rms_norm(input, residual, weight, 1e-6);
  dm.synchronize();

  std::cout << "Output shape: " << output.sizes() << std::endl;
  std::cout << "Residual out shape: " << res_out.sizes() << std::endl;

  TEST_ASSERT(output.sizes() == input.sizes(), "Output shape should match input");
  TEST_ASSERT(res_out.sizes() == input.sizes(), "Residual out shape should match input");

  // Verify residual = input + residual
  at::Tensor expected_res = input + residual;
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(res_out, expected_res, 1e-4, 1e-4);
  std::cout << "Residual correct: " << (cr.passed ? "YES" : "NO") << std::endl;
  TEST_ASSERT(cr.passed, "Residual should equal input + residual");

  std::cout << "[PASS] fused_add_rms_norm_basic" << std::endl;
  return 0;
}

int test_fused_add_rms_norm_with_weight(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: fused_add_rms_norm_with_weight ===" << std::endl;

  constexpr int64_t BATCH = 8;
  constexpr int64_t HIDDEN = 512;

  at::Tensor input = tf.rand({BATCH, HIDDEN});
  at::Tensor residual = tf.rand({BATCH, HIDDEN});
  at::Tensor weight = tf.rand({HIDDEN});
  dm.synchronize();

  auto [output, res_out] = my_ops::fused_add_rms_norm(input, residual, weight, 1e-6);
  dm.synchronize();

  std::cout << "Output shape: " << output.sizes() << std::endl;
  TEST_ASSERT(output.sizes() == input.sizes(), "Shape should match");

  std::cout << "[PASS] fused_add_rms_norm_with_weight" << std::endl;
  return 0;
}

int test_fused_add_rms_norm_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: fused_add_rms_norm ===" << std::endl;

  constexpr int64_t BATCH = 1024;
  constexpr int64_t HIDDEN = 4096;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor input = tf.rand({BATCH, HIDDEN});
  at::Tensor residual = tf.rand({BATCH, HIDDEN});
  at::Tensor weight = tf.ones({HIDDEN});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::fused_add_rms_norm(input, residual, weight, 1e-6); },
                          [&]() { dm.synchronize(); });

  // Read input + residual + weight, write output + residual_out
  int64_t bytes = BATCH * HIDDEN * sizeof(float) * 4 + HIDDEN * sizeof(float);
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Shape: (" << BATCH << ", " << HIDDEN << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "===================================================" << std::endl;
  std::cout << "  Triton JIT Fused Add RMS Norm Operator Test Suite" << std::endl;
  std::cout << "===================================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_fused_add_rms_norm_basic(dm, tf));
  RUN_TEST(test_fused_add_rms_norm_with_weight(dm, tf));
  RUN_TEST(test_fused_add_rms_norm_benchmark(dm, tf));

  std::cout << "\n===================================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "===================================================" << std::endl;

  return 0;
}
