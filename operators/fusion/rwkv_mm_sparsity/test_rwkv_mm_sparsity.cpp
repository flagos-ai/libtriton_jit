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
// test_rwkv_mm_sparsity.cpp - Multi-backend RWKV MM Sparsity Test
// ==============================================================================

#include "benchmark_utils.h"
#include "rwkv_mm_sparsity_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_rwkv_mm_sparsity_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rwkv_mm_sparsity_basic ===" << std::endl;

  constexpr int64_t M = 256;
  constexpr int64_t K = 128;
  constexpr int64_t N = 256;
  constexpr int64_t BLOCK_M = 64;

  at::Tensor a = tf.rand({M, K});
  at::Tensor b = tf.rand({K, N});
  at::Tensor mask = at::ones({M / BLOCK_M}, at::TensorOptions().dtype(at::kInt).device(dm.get_device()));
  dm.synchronize();

  at::Tensor result = my_ops::rwkv_mm_sparsity(a, b, mask);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.size(0) == M && result.size(1) == N, "Output shape should be [M, N]");

  std::cout << "[PASS] rwkv_mm_sparsity_basic" << std::endl;
  return 0;
}

int test_rwkv_mm_sparsity_sparse_mask(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rwkv_mm_sparsity_sparse_mask ===" << std::endl;

  constexpr int64_t M = 512;
  constexpr int64_t K = 256;
  constexpr int64_t N = 512;
  constexpr int64_t BLOCK_M = 64;

  at::Tensor a = tf.rand({M, K});
  at::Tensor b = tf.rand({K, N});
  // Create sparse mask (only compute some blocks)
  at::Tensor mask = at::zeros({M / BLOCK_M}, at::TensorOptions().dtype(at::kInt).device(dm.get_device()));
  // Enable only first half of blocks
  mask.slice(0, 0, M / BLOCK_M / 2).fill_(1);
  dm.synchronize();

  at::Tensor result = my_ops::rwkv_mm_sparsity(a, b, mask);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;

  std::cout << "[PASS] rwkv_mm_sparsity_sparse_mask" << std::endl;
  return 0;
}

int test_rwkv_mm_sparsity_shapes(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: rwkv_mm_sparsity_shapes ===" << std::endl;

  std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> shapes = {
      {128,  64, 128, 32},
      {256, 128, 256, 64},
      {512, 256, 512, 64},
  };

  for (const auto& [M, K, N, BLOCK_M] : shapes) {
    at::Tensor a = tf.rand({M, K});
    at::Tensor b = tf.rand({K, N});
    at::Tensor mask = at::ones({M / BLOCK_M}, at::TensorOptions().dtype(at::kInt).device(dm.get_device()));
    dm.synchronize();

    at::Tensor result = my_ops::rwkv_mm_sparsity(a, b, mask);
    dm.synchronize();

    bool shape_match = (result.size(0) == M && result.size(1) == N);
    std::cout << "Shape (" << M << ", " << K << ") x (" << K << ", " << N
              << "): " << (shape_match ? "PASS" : "FAIL") << std::endl;

    TEST_ASSERT(shape_match, "rwkv_mm_sparsity_shapes failed");
  }

  return 0;
}

int test_rwkv_mm_sparsity_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: rwkv_mm_sparsity ===" << std::endl;

  constexpr int64_t M = 1024;
  constexpr int64_t K = 512;
  constexpr int64_t N = 1024;
  constexpr int64_t BLOCK_M = 64;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor a = tf.rand({M, K});
  at::Tensor b = tf.rand({K, N});
  at::Tensor mask = at::ones({M / BLOCK_M}, at::TensorOptions().dtype(at::kInt).device(dm.get_device()));
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::rwkv_mm_sparsity(a, b, mask); }, [&]() { dm.synchronize(); });

  int64_t flops = 2 * M * N * K;
  double tflops = calculate_tflops(flops, stats.mean);

  std::cout << "Shape: (" << M << ", " << K << ") x (" << K << ", " << N << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "TFLOPS:       " << tflops << std::endl;

  return 0;
}

int main() {
  std::cout << "==================================================" << std::endl;
  std::cout << "  Triton JIT RWKV MM Sparsity Operator Test Suite " << std::endl;
  std::cout << "==================================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_rwkv_mm_sparsity_basic(dm, tf));
  RUN_TEST(test_rwkv_mm_sparsity_sparse_mask(dm, tf));
  RUN_TEST(test_rwkv_mm_sparsity_shapes(dm, tf));
  RUN_TEST(test_rwkv_mm_sparsity_benchmark(dm, tf));

  std::cout << "\n==================================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "==================================================" << std::endl;

  return 0;
}
