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
// test_reshape_and_cache_flash.cpp - Multi-backend Reshape and Cache Test
// ==============================================================================

#include "benchmark_utils.h"
#include "reshape_and_cache_flash_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_reshape_and_cache_flash_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: reshape_and_cache_flash_basic ===" << std::endl;

  constexpr int64_t NUM_TOKENS = 32;
  constexpr int64_t NUM_HEADS = 8;
  constexpr int64_t HEAD_DIM = 64;
  constexpr int64_t NUM_BLOCKS = 16;
  constexpr int64_t BLOCK_SIZE = 16;

  at::Tensor key = tf.rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM});
  at::Tensor value = tf.rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM});
  at::Tensor key_cache = tf.zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM});
  at::Tensor value_cache = tf.zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM});
  at::Tensor slot_mapping =
      at::arange(NUM_TOKENS, at::TensorOptions().dtype(at::kLong).device(dm.get_device()));
  dm.synchronize();

  std::cout << "Key shape: " << key.sizes() << std::endl;
  std::cout << "Cache shape: " << key_cache.sizes() << std::endl;

  my_ops::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping);
  dm.synchronize();

  // Verify cache was updated (not all zeros)
  float key_cache_sum = key_cache.abs().sum().item<float>();
  float value_cache_sum = value_cache.abs().sum().item<float>();

  std::cout << "Key cache sum: " << key_cache_sum << std::endl;
  std::cout << "Value cache sum: " << value_cache_sum << std::endl;

  TEST_ASSERT(key_cache_sum > 0, "Key cache should be updated");
  TEST_ASSERT(value_cache_sum > 0, "Value cache should be updated");

  std::cout << "[PASS] reshape_and_cache_flash_basic" << std::endl;
  return 0;
}

int test_reshape_and_cache_flash_partial(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: reshape_and_cache_flash_partial ===" << std::endl;

  constexpr int64_t NUM_TOKENS = 16;
  constexpr int64_t NUM_HEADS = 4;
  constexpr int64_t HEAD_DIM = 32;
  constexpr int64_t NUM_BLOCKS = 8;
  constexpr int64_t BLOCK_SIZE = 8;

  at::Tensor key = tf.rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM});
  at::Tensor value = tf.rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM});
  at::Tensor key_cache = tf.zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM});
  at::Tensor value_cache = tf.zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM});
  // Only map half the tokens
  at::Tensor slot_mapping =
      at::arange(NUM_TOKENS / 2, at::TensorOptions().dtype(at::kLong).device(dm.get_device()));
  dm.synchronize();

  my_ops::reshape_and_cache_flash(key.slice(0, 0, NUM_TOKENS / 2),
                                  value.slice(0, 0, NUM_TOKENS / 2),
                                  key_cache,
                                  value_cache,
                                  slot_mapping);
  dm.synchronize();

  std::cout << "[PASS] reshape_and_cache_flash_partial" << std::endl;
  return 0;
}

int test_reshape_and_cache_flash_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: reshape_and_cache_flash ===" << std::endl;

  constexpr int64_t NUM_TOKENS = 512;
  constexpr int64_t NUM_HEADS = 32;
  constexpr int64_t HEAD_DIM = 128;
  constexpr int64_t NUM_BLOCKS = 1024;
  constexpr int64_t BLOCK_SIZE = 16;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor key = tf.rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM});
  at::Tensor value = tf.rand({NUM_TOKENS, NUM_HEADS, HEAD_DIM});
  at::Tensor key_cache = tf.zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM});
  at::Tensor value_cache = tf.zeros({NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_DIM});
  at::Tensor slot_mapping =
      at::arange(NUM_TOKENS, at::TensorOptions().dtype(at::kLong).device(dm.get_device()));
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats =
      runner.run([&]() { my_ops::reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping); },
                 [&]() { dm.synchronize(); });

  // Read key + value, write to cache
  int64_t bytes = NUM_TOKENS * NUM_HEADS * HEAD_DIM * sizeof(float) * 4;
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Tokens: " << NUM_TOKENS << ", Heads: " << NUM_HEADS << ", HeadDim: " << HEAD_DIM << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "=======================================================" << std::endl;
  std::cout << "  Triton JIT Reshape and Cache Flash Operator Test Suite" << std::endl;
  std::cout << "=======================================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_reshape_and_cache_flash_basic(dm, tf));
  RUN_TEST(test_reshape_and_cache_flash_partial(dm, tf));
  RUN_TEST(test_reshape_and_cache_flash_benchmark(dm, tf));

  std::cout << "\n=======================================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "=======================================================" << std::endl;

  return 0;
}
