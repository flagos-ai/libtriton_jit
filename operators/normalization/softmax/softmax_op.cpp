// ==============================================================================
// softmax_op.cpp - Multi-backend Triton JIT Softmax
// ==============================================================================

#include "softmax_op.h"
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

int64_t next_power_of_2(int64_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

}  // anonymous namespace

namespace my_ops {
using namespace triton_jit;

at::Tensor softmax(const at::Tensor& input, int64_t dim) {
    dim = at::maybe_wrap_dim(dim, input.dim());

    // Permute to put softmax dim last
    std::vector<int64_t> perm;
    for (int64_t i = 0; i < input.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    at::Tensor x_permuted = input.permute(perm).contiguous();

    int64_t n_rows = x_permuted.numel() / x_permuted.size(-1);
    int64_t n_cols = x_permuted.size(-1);
    at::Tensor x_flat = x_permuted.view({n_rows, n_cols});

#if defined(BACKEND_MUSA)
    void* out_ptr = nullptr;
    musaMalloc(&out_ptr, n_rows * n_cols * at::elementSize(input.scalar_type()));
    auto opts = at::TensorOptions().dtype(input.scalar_type()).device(input.device());
    auto deleter = [](void* ptr) { musaFree(ptr); };
    at::Tensor output = at::from_blob(out_ptr, {n_rows, n_cols}, deleter, opts);
#else
    at::Tensor output = at::empty({n_rows, n_cols}, input.options());
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("softmax.py"), "softmax_kernel");

    int64_t BLOCK_SIZE = next_power_of_2(n_cols);
    constexpr int num_warps = 4;
    constexpr int num_stages = 1;

    c10::DeviceGuard guard(input.device());
    RawStream stream = get_device_stream(x_flat);

    f(stream, n_rows, 1, 1, num_warps, num_stages,
      x_flat, output,
      x_flat.stride(0), output.stride(0),
      n_cols, BLOCK_SIZE);

    // Reshape and inverse permute
    output = output.view(x_permuted.sizes());

    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        inv_perm[perm[i]] = i;
    }
    output = output.permute(inv_perm).contiguous();

    return output;
}

TORCH_LIBRARY(softmax_ops, m) {
    m.def("softmax(Tensor self, int dim) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(softmax_ops, PrivateUse1, m) {
        m.impl("softmax", TORCH_FN(softmax));
    }
#else
    TORCH_LIBRARY_IMPL(softmax_ops, CUDA, m) {
        m.impl("softmax", TORCH_FN(softmax));
    }
#endif

}  // namespace my_ops
