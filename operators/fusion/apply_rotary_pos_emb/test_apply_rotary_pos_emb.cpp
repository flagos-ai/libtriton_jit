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
// test_apply_rotary_pos_emb.cpp - Multi-backend Rotary Position Embedding Test
// ==============================================================================

#include "apply_rotary_pos_emb_op.h"
#include "benchmark_utils.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_apply_rotary_pos_emb_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: apply_rotary_pos_emb_basic ===" << std::endl;

  constexpr int64_t SEQ_LEN = 128;
  constexpr int64_t NUM_HEADS = 8;
  constexpr int64_t HEAD_DIM = 64;

  at::Tensor q = tf.rand({SEQ_LEN, NUM_HEADS, HEAD_DIM});
  at::Tensor k = tf.rand({SEQ_LEN, NUM_HEADS, HEAD_DIM});
  at::Tensor cos = tf.rand({SEQ_LEN, HEAD_DIM / 2});
  at::Tensor sin = tf.rand({SEQ_LEN, HEAD_DIM / 2});
  dm.synchronize();

  auto [q_out, k_out] = my_ops::apply_rotary_pos_emb(q, k, cos, sin, HEAD_DIM);
  dm.synchronize();

  std::cout << "Q output shape: " << q_out.sizes() << std::endl;
  std::cout << "K output shape: " << k_out.sizes() << std::endl;

  TEST_ASSERT(q_out.sizes() == q.sizes(), "Q output shape should match input");
  TEST_ASSERT(k_out.sizes() == k.sizes(), "K output shape should match input");

  std::cout << "[PASS] apply_rotary_pos_emb_basic" << std::endl;
  return 0;
}

int test_apply_rotary_pos_emb_shapes(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: apply_rotary_pos_emb_shapes ===" << std::endl;

  std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
      { 64,  4,  32},
      {256,  8,  64},
      {512, 16, 128},
  };

  for (const auto& [S, H, D] : shapes) {
    at::Tensor q = tf.rand({S, H, D});
    at::Tensor k = tf.rand({S, H, D});
    at::Tensor cos = tf.rand({S, D / 2});
    at::Tensor sin = tf.rand({S, D / 2});
    dm.synchronize();

    auto [q_out, k_out] = my_ops::apply_rotary_pos_emb(q, k, cos, sin, D);
    dm.synchronize();

    bool shape_match = (q_out.sizes() == q.sizes() && k_out.sizes() == k.sizes());
    std::cout << "Shape (" << S << ", " << H << ", " << D << "): " << (shape_match ? "PASS" : "FAIL")
              << std::endl;

    TEST_ASSERT(shape_match, "apply_rotary_pos_emb_shapes failed");
  }

  return 0;
}

int test_apply_rotary_pos_emb_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: apply_rotary_pos_emb ===" << std::endl;

  constexpr int64_t SEQ_LEN = 2048;
  constexpr int64_t NUM_HEADS = 32;
  constexpr int64_t HEAD_DIM = 128;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor q = tf.rand({SEQ_LEN, NUM_HEADS, HEAD_DIM});
  at::Tensor k = tf.rand({SEQ_LEN, NUM_HEADS, HEAD_DIM});
  at::Tensor cos = tf.rand({SEQ_LEN, HEAD_DIM / 2});
  at::Tensor sin = tf.rand({SEQ_LEN, HEAD_DIM / 2});
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::apply_rotary_pos_emb(q, k, cos, sin, HEAD_DIM); },
                          [&]() { dm.synchronize(); });

  // Read q + k + cos + sin, write q_out + k_out
  int64_t bytes =
      SEQ_LEN * NUM_HEADS * HEAD_DIM * sizeof(float) * 4 + SEQ_LEN * (HEAD_DIM / 2) * sizeof(float) * 2;
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Shape: (" << SEQ_LEN << ", " << NUM_HEADS << ", " << HEAD_DIM << ")" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "======================================================" << std::endl;
  std::cout << "  Triton JIT Apply Rotary Pos Emb Operator Test Suite " << std::endl;
  std::cout << "======================================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_apply_rotary_pos_emb_basic(dm, tf));
  RUN_TEST(test_apply_rotary_pos_emb_shapes(dm, tf));
  RUN_TEST(test_apply_rotary_pos_emb_benchmark(dm, tf));

  std::cout << "\n======================================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "======================================================" << std::endl;

  return 0;
}
