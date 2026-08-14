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
// test_rwkv_ka_fusion.cpp - Multi-backend RWKV KA Fusion Test
// ==============================================================================

#include "benchmark_utils.h"
#include "rwkv_ka_fusion_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_rwkv_ka_fusion_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rwkv_ka_fusion_basic ===" << std::endl;

  constexpr int64_t BATCH = 8;
  constexpr int64_t SEQ_LEN = 128;
  constexpr int64_t HIDDEN = 256;

  at::Tensor k = tf.rand({BATCH, SEQ_LEN, HIDDEN});
  at::Tensor a = tf.rand({BATCH, SEQ_LEN, HIDDEN});
  dm.synchronize();

  at::Tensor result = my_ops::rwkv_ka_fusion(k, a);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.sizes() == k.sizes(), "Output shape should match input shape");

  std::cout << "[PASS] rwkv_ka_fusion_basic" << std::endl;
  return 0;
}

int test_rwkv_ka_fusion_shapes(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rwkv_ka_fusion_shapes ===" << std::endl;

  std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
      { 1,  64, 128},
      { 4, 256, 512},
      {16, 128, 256},
  };

  for (const auto& [B, S, H] : shapes) {
    at::Tensor k = tf.rand({B, S, H});
    at::Tensor a = tf.rand({B, S, H});
    dm.synchronize();

    at::Tensor result = my_ops::rwkv_ka_fusion(k, a);
    dm.synchronize();

    bool shape_match = (result.sizes() == k.sizes());
    std::cout << "Shape (" << B << ", " << S << ", " << H << "): " << (shape_match ? "PASS" : "FAIL")
              << std::endl;

    TEST_ASSERT(shape_match, "rwkv_ka_fusion_shapes failed");
  }

  return 0;
}

int test_rwkv_ka_fusion_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: rwkv_ka_fusion ===" << std::endl;

  constexpr int64_t BATCH = 32;
  constexpr int64_t SEQ_LEN = 512;
  constexpr int64_t HIDDEN = 1024;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor k = tf.rand({BATCH, SEQ_LEN, HIDDEN});
  at::Tensor a = tf.rand({BATCH, SEQ_LEN, HIDDEN});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::rwkv_ka_fusion(k, a); }, [&]() { dm.synchronize(); });

  // Read k + a, write output
  int64_t bytes = BATCH * SEQ_LEN * HIDDEN * sizeof(float) * 3;
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Shape: (" << BATCH << ", " << SEQ_LEN << ", " << HIDDEN << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "================================================" << std::endl;
  std::cout << "  Triton JIT RWKV KA Fusion Operator Test Suite " << std::endl;
  std::cout << "================================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_rwkv_ka_fusion_basic(dm, tf));
  RUN_TEST(test_rwkv_ka_fusion_shapes(dm, tf));
  RUN_TEST(test_rwkv_ka_fusion_benchmark(dm, tf));

  std::cout << "\n================================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "================================================" << std::endl;

  return 0;
}
