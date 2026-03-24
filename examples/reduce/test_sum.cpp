#include <gtest/gtest.h>
#include "sum_op.h"
#include "torch/torch.h"
#include "triton_jit/backend_config.h"

static at::Device test_device() {
#if defined(BACKEND_NPU)
  return at::Device("npu:0");
#elif defined(BACKEND_MUSA)
  return at::Device("musa:0");
#else
  return at::kCUDA;
#endif
}

TEST(sum_test, dim1) {
  at::Tensor tensor = at::rand({16, 4 * 1024}, test_device());

  at::Tensor result = my_ops::sum_dim(tensor, {1}, false, c10::nullopt);
  at::Tensor expected = at::sum(tensor, {1}, false, c10::nullopt);
  EXPECT_TRUE(torch::allclose(result, expected, 1e-3, 1e-3));
}

TEST(sum_test, dim0) {
  at::Tensor tensor = at::rand({256, 512}, test_device());

  at::Tensor result = my_ops::sum_dim(tensor, {0}, false, c10::nullopt);
  at::Tensor expected = at::sum(tensor, {0}, false, c10::nullopt);
  EXPECT_TRUE(torch::allclose(result, expected, 1e-3, 1e-3));
}

TEST(sum_test, keepdim) {
  at::Tensor tensor = at::rand({16, 4 * 1024}, test_device());

  at::Tensor result = my_ops::sum_dim(tensor, {1}, true, c10::nullopt);
  at::Tensor expected = at::sum(tensor, {1}, true, c10::nullopt);
  EXPECT_TRUE(torch::allclose(result, expected, 1e-3, 1e-3));
  EXPECT_EQ(result.sizes(), expected.sizes());
}
