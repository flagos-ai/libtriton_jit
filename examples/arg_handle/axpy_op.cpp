#include "axpy_op.h"
#include "common/backend_ops.h"
#include "common/op_registration.h"
#include "triton_jit/triton_jit_function.h"

namespace my_ops {
using namespace triton_jit;

at::Tensor axpy(const at::Tensor& x, const at::Tensor& y, const c10::Scalar& alpha) {
  auto res = torch::broadcast_tensors({x, y});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor& xx = res[0];
  const at::Tensor& yy = res[1];

  at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.scalar_type());
  at::Tensor out = triton_jit::ops::backend_empty(xx.sizes(), out_dtype, x.device());

  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy_kernel");

  constexpr int64_t tile_size = 1024;
  constexpr int num_warps = 8;
  constexpr int num_stages = 1;
  const int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::DeviceGuard guard(out.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(x);

  f(stream, num_blocks, 1, 1, num_warps, num_stages, x, y, out, alpha, n, tile_size);
  return out;
}

at::Tensor axpy2(const at::Tensor& x, const at::Tensor& y, const std::optional<c10::Scalar>& alpha) {
  auto res = torch::broadcast_tensors({x, y});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor& xx = res[0];
  const at::Tensor& yy = res[1];

  at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.scalar_type());
  at::Tensor out = triton_jit::ops::backend_empty(xx.sizes(), out_dtype, x.device());

  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy2_kernel");

  constexpr int64_t tile_size = 1024;
  constexpr int num_warps = 8;
  constexpr int num_stages = 1;
  const int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::DeviceGuard guard(out.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(x);

  f(stream, num_blocks, 1, 1, num_warps, num_stages, x, y, out, alpha, n, tile_size);
  return out;
}

at::Tensor axpy3(const at::Tensor& x,
                 const std::optional<at::Tensor>& y,
                 const std::optional<c10::Scalar>& alpha) {
  at::Tensor out = [&]() {
    if (!y.has_value()) {
      return at::empty_like(x);
    } else {
      auto res = torch::broadcast_tensors({x, y.value()});
      res[0] = res[0].contiguous();
      res[1] = res[1].contiguous();
      at::ScalarType out_dtype = at::promote_types(x.scalar_type(), y.value().scalar_type());
      return triton_jit::ops::backend_empty(res[0].sizes(), out_dtype, x.device());
    }
  }();

  const TritonJITFunction& f = TritonJITFunction::get_instance(std::string("axpy.py"), "axpy3_kernel");

  constexpr int64_t tile_size = 1024;
  constexpr int num_warps = 8;
  constexpr int num_stages = 1;
  const int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::DeviceGuard guard(out.device());
  triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(x);

  f(stream, num_blocks, 1, 1, num_warps, num_stages, x, y, out, alpha, n, tile_size);
  return out;
}

TORCH_LIBRARY(axpy_ops, m) {
  m.def("axpy(Tensor self, Tensor other, Scalar alpha) -> Tensor");
  m.def("axpy2(Tensor self, Tensor other, Scalar? alpha) -> Tensor");
  m.def("axpy3(Tensor self, Tensor? other, Scalar? alpha) -> Tensor");
}

REGISTER_TRITON_OP(axpy_ops, "axpy", axpy)
REGISTER_TRITON_OP(axpy_ops, "axpy2", axpy2)
REGISTER_TRITON_OP(axpy_ops, "axpy3", axpy3)

}  // namespace my_ops
