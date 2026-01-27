// ==============================================================================
// argmax_op.cpp - Multi-backend Triton JIT Argmax Operation
// ==============================================================================

#include "argmax_op.h"
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

at::Tensor argmax(const at::Tensor& self, int64_t dim, bool keepdim) {
    dim = at::maybe_wrap_dim(dim, self.dim());

    std::vector<int64_t> perm;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i != dim) perm.push_back(i);
    }
    perm.push_back(dim);

    at::Tensor permuted = self.permute(perm).contiguous();
    int64_t M = permuted.numel() / permuted.size(-1);
    int64_t N = permuted.size(-1);

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < self.dim(); ++i) {
        if (i != dim) out_shape.push_back(self.size(i));
    }

#if defined(BACKEND_MUSA)
    void* ptr = nullptr;
    musaMalloc(&ptr, M * sizeof(int64_t));
    auto opts = at::TensorOptions().dtype(at::kLong).device(self.device());
    auto deleter = [](void* p) { musaFree(p); };
    at::Tensor out = at::from_blob(ptr, {M}, deleter, opts);
#else
    at::Tensor out = at::empty({M}, at::TensorOptions().dtype(at::kLong).device(self.device()));
#endif

    const TritonJITFunction& f = TritonJITFunction::get_instance(
        std::string("argmax.py"), "argmax_kernel");

    constexpr int64_t BLOCK_M = 4;
    constexpr int64_t BLOCK_N = 512;
    constexpr int num_warps = 8;
    constexpr int num_stages = 1;
    const unsigned int num_blocks = (M + BLOCK_M - 1) / BLOCK_M;

    c10::DeviceGuard guard(self.device());
    RawStream stream = get_device_stream(permuted);

    f(stream, num_blocks, 1, 1, num_warps, num_stages,
      permuted.view({M, N}), out, M, N, BLOCK_M, BLOCK_N);

    // Reshape output - out_shape is already in correct order
    if (!out_shape.empty()) {
        out = out.view(out_shape);
    }

    if (keepdim) {
        out = out.unsqueeze(dim);
    }

    return out;
}

TORCH_LIBRARY(argmax_ops, m) {
    m.def("argmax(Tensor self, int dim, bool keepdim=False) -> Tensor");
}

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA)
    TORCH_LIBRARY_IMPL(argmax_ops, PrivateUse1, m) {
        m.impl("argmax", TORCH_FN(argmax));
    }
#else
    TORCH_LIBRARY_IMPL(argmax_ops, CUDA, m) {
        m.impl("argmax", TORCH_FN(argmax));
    }
#endif

}  // namespace my_ops
