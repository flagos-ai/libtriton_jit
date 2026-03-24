#pragma once

#include <optional>
#include "torch/torch.h"

namespace my_ops {

at::Tensor axpy(const at::Tensor& x, const at::Tensor& y, const c10::Scalar& alpha);
at::Tensor axpy2(const at::Tensor& x, const at::Tensor& y, const std::optional<c10::Scalar>& alpha);
at::Tensor axpy3(const at::Tensor& x,
                 const std::optional<at::Tensor>& y,
                 const std::optional<c10::Scalar>& alpha);

}  // namespace my_ops
