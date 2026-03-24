#include <gtest/gtest.h>
#include "add_op.h"
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

TEST(add_test, basic) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  at::Tensor b = at::rand({128 * 1024}, test_device());

  at::Tensor result = my_ops::add_tensor(a, b);
  at::Tensor expected = at::add(a, b);
  EXPECT_TRUE(torch::allclose(result, expected));
}

TEST(add_test, broadcast) {
  at::Tensor a = at::rand({256, 128}, test_device());
  at::Tensor b = at::rand({128}, test_device());

  at::Tensor result = my_ops::add_tensor(a, b);
  at::Tensor expected = at::add(a, b);
  EXPECT_TRUE(torch::allclose(result, expected));
}
