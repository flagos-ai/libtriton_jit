

#include "add_op.h"
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

#if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
#include "torch_npu/csrc/core/npu/NPUStream.h"
#define HAS_TORCH_NPU 1
#else
#define HAS_TORCH_NPU 0
#endif

namespace my_ops {
using namespace triton_jit;

at::Tensor add_tensor(const at::Tensor &a_, const at::Tensor &b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  res[0] = res[0].contiguous();
  res[1] = res[1].contiguous();
  const at::Tensor &a = res[0];
  const at::Tensor &b = res[1];

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string("add.py"), "binary_pointwise_kernel");

  // add utility to build this automatically
  int64_t tile_size = 1024;
  const int num_warps = 8;
  const int num_stages = 1;
  int64_t n = out.numel();
  const unsigned int num_blocks = (n + tile_size - 1) / tile_size;

  c10::DeviceGuard guard(out.device());

  // Check device type and get appropriate stream
  if (a.device().is_cuda()) {
    // CUDA backend
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());
    f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
  }
#if HAS_TORCH_NPU
  else if (a.device().type() == c10::DeviceType::PrivateUse1) {
    // NPU backend (PrivateUse1)
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream(true);
    f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);

    // Synchronize NPU stream
    if (stream != nullptr) {
      aclrtSynchronizeStream(stream);
    }
  }
#endif
  else {
    throw std::runtime_error("Unsupported device type for add_tensor");
  }

  return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("add_tensor", TORCH_FN(add_tensor));
}

#if HAS_TORCH_NPU
TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
  m.impl("add_tensor", TORCH_FN(add_tensor));
}
#endif
}  // namespace my_ops
