#include <gtest/gtest.h>
#include "axpy_op.h"
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

TEST(axpy_test, scalar_int) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  at::Tensor b = at::rand({128 * 1024}, test_device());

  at::Tensor result = my_ops::axpy(a, b, c10::Scalar(10));
  at::Tensor expected = at::add(c10::Scalar(10) * a, b);
  EXPECT_TRUE(torch::allclose(result, expected));
}

TEST(axpy_test, scalar_double) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  at::Tensor b = at::rand({128 * 1024}, test_device());

  at::Tensor result = my_ops::axpy(a, b, c10::Scalar(3.14));
  at::Tensor expected = at::add(c10::Scalar(3.14) * a, b);
  EXPECT_TRUE(torch::allclose(result, expected));
}

TEST(axpy_test, optional_scalar_nullopt) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  at::Tensor b = at::rand({128 * 1024}, test_device());

  at::Tensor result = my_ops::axpy2(a, b, std::nullopt);
  at::Tensor expected = at::add(a, b);
  EXPECT_TRUE(torch::allclose(result, expected));
}

TEST(axpy_test, optional_scalar_has_value) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  at::Tensor b = at::rand({128 * 1024}, test_device());

  std::optional<c10::Scalar> alpha = c10::Scalar(3.14);
  at::Tensor result = my_ops::axpy2(a, b, alpha);
  at::Tensor expected = at::add(alpha.value() * a, b);
  EXPECT_TRUE(torch::allclose(result, expected));
}

TEST(axpy_test, optional_tensor_has_value) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  std::optional<at::Tensor> b = at::rand({128 * 1024}, test_device());

  c10::Scalar alpha(3.14);
  at::Tensor result = my_ops::axpy3(a, b, alpha);
  at::Tensor expected = at::add(alpha * a, b.value());
  EXPECT_TRUE(torch::allclose(result, expected));
}

TEST(axpy_test, optional_tensor_nullopt) {
  at::Tensor a = at::rand({128 * 1024}, test_device());
  std::optional<at::Tensor> b = std::nullopt;

  c10::Scalar alpha(3.14);
  at::Tensor result = my_ops::axpy3(a, b, alpha);
  at::Tensor expected = alpha * a;
  EXPECT_TRUE(torch::allclose(result, expected));
}
