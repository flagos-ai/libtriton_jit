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

// ==============================================================================
//                         BACKEND DETECTION & HEADERS
// ==============================================================================

#if defined(BACKEND_NPU)
    // ----------------------------- NPU Backend -----------------------------
    #if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
        #include "torch_npu/csrc/core/npu/NPUStream.h"
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
        #warning "torch_npu headers not found, NPU stream support disabled"
    #endif

#elif defined(BACKEND_MUSA)
    // ----------------------------- MUSA Backend ----------------------------
    #include <musa.h>
    #if __has_include("torch_musa/csrc/core/musa/MUSAStream.h")
        #include "torch_musa/csrc/core/musa/MUSAStream.h"
        #define HAS_TORCH_MUSA 1
    #else
        #define HAS_TORCH_MUSA 0
        #warning "torch_musa headers not found, MUSA stream support disabled"
    #endif

#else
    // ----------------------- CUDA / IX Backend (Default) -------------------
    #include "c10/cuda/CUDAStream.h"

#endif

// ==============================================================================
//                         BACKEND-SPECIFIC TYPES & UTILITIES
// ==============================================================================

namespace {

// ------------------------------ Stream Types ---------------------------------

#if defined(BACKEND_NPU)
    using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
    using RawStream = MUstream;
#else
    using RawStream = CUstream;
#endif

// ----------------------------- Stream Getter ---------------------------------

inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& tensor) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif

#elif defined(BACKEND_MUSA)
    #if HAS_TORCH_MUSA
        return c10::musa::getCurrentMUSAStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif

#else  // CUDA / IX
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

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

    // def sum_kernel(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr):
    const TritonJITFunction &f = TritonJITFunction::get_instance("./sum.py", "sum_kernel");
    int64_t tile_m = 4;
    int64_t tile_n = 512;
    const int num_warps = 8;
    const int num_stages = 2;
    const unsigned int num_blocks = (non_reduction_size + tile_m - 1) / tile_m;

    // ------------------------- Kernel Launch ---------------------------------
    c10::DeviceGuard guard(out.device());
    RawStream stream = get_device_stream(permuted_self);

    f(stream,
      num_blocks,
      1,
      1,
      num_warps,
      num_stages,
      permuted_self,
      out,
      non_reduction_size,
      reduction_size,
      tile_m,
      tile_n,
      num_stages);
    return out;
}

// ==============================================================================
//                         TORCH LIBRARY REGISTRATION
// ==============================================================================

TORCH_LIBRARY(my_ops, m) {
    m.def("sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor");
}

// Backend-specific dispatch key registration
#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    // NPU and MUSA use PrivateUse1 dispatch key
    TORCH_LIBRARY_IMPL(my_ops, PrivateUse1, m) {
        m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
    }
#else
    // CUDA and IX use CUDA dispatch key
    TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
        m.impl("sum.dim_IntList", TORCH_FN(sum_dim));
    }
#endif

}  // namespace my_ops
