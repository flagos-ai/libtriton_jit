// ==============================================================================
// max_op.cpp - Multi-backend Triton JIT Max Reduction Operation
// ==============================================================================

#include "max_op.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include "ATen/WrapDimUtils.h"

#if defined(BACKEND_NPU)
    #if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
        #include "torch_npu/csrc/core/npu/NPUStream.h"
        #define HAS_TORCH_NPU 1
    #else
        #define HAS_TORCH_NPU 0
    #endif
#elif defined(BACKEND_MUSA)
    #include <musa_runtime.h>
#else
    #include "c10/cuda/CUDAStream.h"
#endif

namespace {

#if defined(BACKEND_NPU)
    using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
    using RawStream = musaStream_t;
#else
    using RawStream = CUstream;
#endif

inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& tensor) {
#if defined(BACKEND_NPU)
    #if HAS_TORCH_NPU
        return c10_npu::getCurrentNPUStream(tensor.device().index()).stream();
    #else
        return nullptr;
    #endif
#elif defined(BACKEND_MUSA)
    return nullptr;
#else
    auto cuda_stream = c10::cuda::getCurrentCUDAStream(tensor.device().index());
    return static_cast<CUstream>(cuda_stream.stream());
#endif
}

}  // anonymous namespace

namespace my_ops {
using namespace triton_jit;

std::tuple<at::Tensor, at::Tensor> max_dim(const at::Tensor& self, int64_t dim, bool keepdim) {
    // Wrap dimension
    dim = at::maybe_wrap_dim(dim, self.dim());

    // Permute to put reduction dim last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    at::Tensor permuted = self.permute(perm).contiguous();
    int64_t M = permuted.numel() / permuted.size(-1);
    int64_t N = permuted.size(-1);

    // Compute output shape
    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i != dim) out_shape.push_back(self.size(i));
    }

    // Allocate outputs
#if defined(BACKEND_MUSA)
    void* val_ptr = nullptr;
    void* idx_ptr = nullptr;
    size_t val_bytes = M * at::elementSize(self.scalar_type());
    size_t idx_bytes = M * sizeof(int64_t);
    musaMalloc(&val_ptr, val_bytes);
    musaMalloc(&idx_ptr, idx_bytes);

    auto val_opts = at::TensorOptions().dtype(self.scalar_type()).device(self.device());
    auto idx_opts = at::TensorOptions().dtype(at::kLong).device(self.device());
    auto val_deleter = [](void* ptr) { musaFree(ptr); };
    auto idx_deleter = [](void* ptr) { musaFree(ptr); };

    at::Tensor out_vals = at::from_blob(val_ptr, {M}, val_deleter, val_opts);
    at::Tensor out_idx = at::from_blob(idx_ptr, {M}, idx_deleter, idx_opts);
#else
    at::Tensor out_vals = at::empty({M}, self.options());
    at::Tensor out_idx = at::empty({M}, at::TensorOptions().dtype(at::kLong).device(self.device()));
#endif

    // Launch kernel
    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("max.py"), "max_with_indices_kernel");

    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 512;
    constexpr int num_warps = 8;
    constexpr int num_stages = 1;
    const unsigned int num_blocks = (M + BLOCK_M - 1) / BLOCK_M;

    c10::DeviceGuard guard(self.device());
    RawStream stream = get_device_stream(permuted);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      permuted.view({M, N}), out_vals, out_idx,
      M, N, BLOCK_M, BLOCK_N);

    // Reshape output - out_shape is already in correct order (original dims except reduced)
    if (!out_shape.empty()) {
        out_vals = out_vals.view(out_shape);
        out_idx = out_idx.view(out_shape);
    }

    if (keepdim) {
        out_vals = out_vals.unsqueeze(dim);
        out_idx = out_idx.unsqueeze(dim);
    }

    return {out_vals, out_idx};
}

at::Tensor max(const at::Tensor& self) {
    auto [vals, _] = max_dim(self.flatten(), 0, false);
    return vals;
}

TORCH_LIBRARY(max_ops, m) {
    m.def("max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)");
    m.def("max(Tensor self) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(max_ops, PrivateUse1, m) {
        m.impl("max.dim", TORCH_FN(max_dim));
        m.impl("max", TORCH_FN(max));
    }
#else
    TORCH_LIBRARY_IMPL(max_ops, CUDA, m) {
        m.impl("max.dim", TORCH_FN(max_dim));
        m.impl("max", TORCH_FN(max));
    }
#endif

}  // namespace my_ops
