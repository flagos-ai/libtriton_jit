#include "add_op.h"
#include "torch/torch.h"
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
	
	// Get NPU stream
	aclrtStream stream = nullptr;
#if HAS_TORCH_NPU
	stream = c10_npu::getCurrentNPUStream(a.device().index()).stream();
#endif

	// Execute kernel
	c10::DeviceGuard guard(out.device());
	f(stream, num_blocks, 1, 1, num_warps, num_stages, a, b, out, n, tile_size);
	
	return out;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("add_tensor(Tensor self, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
	m.impl("add_tensor", TORCH_FN(add_tensor));
}
}  // namespace my_ops
