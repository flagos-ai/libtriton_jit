// ==============================================================================
// sum_op.cpp - Multi-backend Triton JIT Sum Reduction Operation
// Supported backends: CUDA, IX, NPU, MUSA
// ==============================================================================

#include "sum_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include <filesystem>
#include "ATen/WrapDimUtils.h"
#include "ATen/native/ReduceOpsUtils.h"
#include "c10/util/DimVector.h"

#include "operators/common/backend_ops.h"
#include "operators/common/op_registration.h"
#include "operators/common/kernel_config.h"

// ==============================================================================
//                         HELPER FUNCTIONS
// ==============================================================================

std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(
    const at::Tensor &tensor, at::OptionalIntArrayRef reduction_axes_opt) {
    int64_t dim = tensor.dim();
    c10::DimVector reduction_axes;

    if (reduction_axes_opt.has_value()) {
        reduction_axes = reduction_axes_opt.value().vec();
    }

    std::unordered_set<int64_t> reduction_set(reduction_axes.begin(), reduction_axes.end());

    c10::DimVector left_axes, right_axes;
    int64_t non_reduction_size = 1, reduction_size = 1;

    for (int64_t i = 0; i < dim; ++i) {
        if (reduction_set.count(i)) {
            right_axes.push_back(i);
            reduction_size *= tensor.size(i);
        } else {
            left_axes.push_back(i);
            non_reduction_size *= tensor.size(i);
        }
    }

    // Concatenate left and right axes to form the new permutation order
    c10::DimVector permute_order = left_axes;
    permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

    return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}

// ==============================================================================
//                         KERNEL IMPLEMENTATION
// ==============================================================================

namespace my_ops {
using namespace triton_jit;

// signature
// sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType?
// dtype=None) -> Tensor
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim,
                   ::std::optional<at::ScalarType> dtype) {
    // permute reduction dims to the right, and make the input contiguous
    at::DimVector dims_ = at::native::make_dim_vector(dim, self.dim());
    at::maybe_wrap_dims(dims_, self.dim());
    at::DimVector shape = at::meta::get_reduction_shape(self, dims_, keepdim, false);
    c10::ScalarType out_dtype = at::native::get_dtype_from_self(self, dtype, true);
    at::Tensor out = at::empty(shape, self.options());
    auto [permuted_self, non_reduction_size, reduction_size] = permute_reduction_axes_right(self, dims_);
    permuted_self = permuted_self.contiguous();

    // sum_dim_kernel(inp, out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr)
    const TritonJITFunction &f = TritonJITFunction::get_instance("./sum.py", "sum_dim_kernel");

    constexpr auto cfg = triton_jit::ops::default_reduce_sum_config();

    const unsigned int num_blocks = (non_reduction_size + cfg.BLOCK_M - 1) / cfg.BLOCK_M;

    // ------------------------- Kernel Launch ---------------------------------
    c10::DeviceGuard guard(out.device());
    triton_jit::ops::RawStream stream = triton_jit::ops::get_device_stream(permuted_self);

    f(stream,
      num_blocks,
      1,
      1,
      cfg.num_warps,
      cfg.num_stages,
      permuted_self,
      out,
      non_reduction_size,
      reduction_size,
      cfg.BLOCK_M,
      cfg.BLOCK_N);
    return out;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(my_ops, m) {
    m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
}

REGISTER_TRITON_OP(my_ops, "sum.dim_IntList", sum_dim)

}  // namespace my_ops
