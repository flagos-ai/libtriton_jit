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
// test_embedding.cpp - Multi-backend Triton JIT Embedding Test
// ==============================================================================

#include "benchmark_utils.h"
#include "embedding_op.h"
#include "test_framework.h"

#include <iostream>

using namespace triton_jit::test;
using namespace triton_jit::benchmark;

int test_embedding_basic(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: embedding_basic ===" << std::endl;

  constexpr int64_t NUM_EMBEDDINGS = 1000;
  constexpr int64_t EMBEDDING_DIM = 256;
  constexpr int64_t BATCH = 16;
  constexpr int64_t SEQ_LEN = 32;

  at::Tensor weight = tf.rand({NUM_EMBEDDINGS, EMBEDDING_DIM});
  at::Tensor indices = at::randint(0,
                                   NUM_EMBEDDINGS,
                                   {BATCH, SEQ_LEN},
                                   at::TensorOptions().dtype(at::kLong).device(dm.get_device()));
  dm.synchronize();

  at::Tensor result = my_ops::embedding(indices, weight);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.size(0) == BATCH && result.size(1) == SEQ_LEN && result.size(2) == EMBEDDING_DIM,
              "Output shape should be [BATCH, SEQ_LEN, EMBEDDING_DIM]");

  // Reference
  at::Tensor expected = at::embedding(weight, indices);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "embedding_basic correctness check failed");
  return 0;
}

int test_embedding_1d(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Test: embedding_1d ===" << std::endl;

  constexpr int64_t NUM_EMBEDDINGS = 500;
  constexpr int64_t EMBEDDING_DIM = 128;
  constexpr int64_t SIZE = 64;

  at::Tensor weight = tf.rand({NUM_EMBEDDINGS, EMBEDDING_DIM});
  at::Tensor indices =
      at::randint(0, NUM_EMBEDDINGS, {SIZE}, at::TensorOptions().dtype(at::kLong).device(dm.get_device()));
  dm.synchronize();

  at::Tensor result = my_ops::embedding(indices, weight);
  dm.synchronize();

  std::cout << "Output shape: " << result.sizes() << std::endl;
  TEST_ASSERT(result.size(0) == SIZE && result.size(1) == EMBEDDING_DIM,
              "Output shape should be [SIZE, EMBEDDING_DIM]");

  at::Tensor expected = at::embedding(weight, indices);
  dm.synchronize();

  CorrectnessResult cr = CorrectnessChecker::compare(result, expected, 1e-6, 1e-6);
  TestRunner::print_result(cr);

  TEST_ASSERT(cr.passed, "embedding_1d correctness check failed");
  return 0;
}

int test_embedding_benchmark(DeviceManager& dm, TensorFactory& tf) {
  std::cout << "\n=== Benchmark: embedding ===" << std::endl;

  constexpr int64_t NUM_EMBEDDINGS = 32000;
  constexpr int64_t EMBEDDING_DIM = 4096;
  constexpr int64_t BATCH = 32;
  constexpr int64_t SEQ_LEN = 512;
  constexpr int WARMUP = 10;
  constexpr int ITERS = 100;

  at::Tensor weight = tf.rand({NUM_EMBEDDINGS, EMBEDDING_DIM});
  at::Tensor indices = at::randint(0,
                                   NUM_EMBEDDINGS,
                                   {BATCH, SEQ_LEN},
                                   at::TensorOptions().dtype(at::kLong).device(dm.get_device()));
  dm.synchronize();

  BenchmarkRunner runner(WARMUP, ITERS);
  auto stats = runner.run([&]() { my_ops::embedding(indices, weight); }, [&]() { dm.synchronize(); });

  // Output bytes
  int64_t bytes = BATCH * SEQ_LEN * EMBEDDING_DIM * sizeof(float);
  double bandwidth = calculate_bandwidth_gbps(bytes, stats.mean);

  std::cout << "Weight: [" << NUM_EMBEDDINGS << ", " << EMBEDDING_DIM << "]" << std::endl;
  std::cout << "Indices: [" << BATCH << ", " << SEQ_LEN << "]" << std::endl;
  std::cout << "Mean latency: " << stats.mean << " us" << std::endl;
  std::cout << "Bandwidth:    " << bandwidth << " GB/s" << std::endl;

  return 0;
}

int main() {
  std::cout << "============================================" << std::endl;
  std::cout << "  Triton JIT Embedding Operator Test Suite " << std::endl;
  std::cout << "============================================" << std::endl;

  DeviceManager dm;
  if (dm.initialize() != 0) {
    std::cerr << "Failed to initialize device" << std::endl;
    return -1;
  }

  std::cout << "Backend: " << dm.get_backend_name() << std::endl;
  TensorFactory tf(dm);

  RUN_TEST(test_embedding_basic(dm, tf));
  RUN_TEST(test_embedding_1d(dm, tf));
  RUN_TEST(test_embedding_benchmark(dm, tf));

  std::cout << "\n============================================" << std::endl;
  std::cout << "  All tests passed!" << std::endl;
  std::cout << "============================================" << std::endl;

  return 0;
}
