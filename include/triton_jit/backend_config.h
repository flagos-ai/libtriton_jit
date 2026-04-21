#pragma once

#if defined(BACKEND_NPU)
#include "triton_jit/backends/npu_backend.h"
#elif defined(BACKEND_MUSA)
#include "triton_jit/backends/musa_backend.h"
#elif defined(BACKEND_IX)
#include "triton_jit/backends/ix_backend.h"
#elif defined(BACKEND_MLU)
#include "triton_jit/backends/mlu_backend.h"
#elif defined(BACKEND_HCU)
#include "triton_jit/backends/hcu_backend.h"
#else
#include "triton_jit/backends/cuda_backend.h"
#endif

namespace triton_jit {

// Forward declarations of template classes
template <BackendPolicy Backend>
class TritonKernelImpl;

template <BackendPolicy Backend>
class TritonJITFunctionImpl;

#if defined(BACKEND_NPU)
/// Default backend for NPU (Ascend)
using DefaultBackend = NpuBackend;

#elif defined(BACKEND_MUSA)
/// Default backend for MUSA (Moore Threads)
using DefaultBackend = MusaBackend;

#elif defined(BACKEND_MLU)
/// Default backend for MLU (Cambricon)
using DefaultBackend = MluBackend;

#elif defined(BACKEND_CUDA)
/// Default backend for CUDA
using DefaultBackend = CudaBackend;

#elif defined(BACKEND_IX)
/// Default backend for IX (Tianshu)
using DefaultBackend = IxBackend;

#elif defined(BACKEND_HCU)
/// Default backend for HCU (Hygon, HIP-compatible)
using DefaultBackend = HcuBackend;

#else
// Default to CUDA if no backend specified
#warning "No backend specified, defaulting to CUDA. Use -DBACKEND=CUDA explicitly."
using DefaultBackend = CudaBackend;

#endif

using TritonKernel = TritonKernelImpl<DefaultBackend>;
using TritonJITFunction = TritonJITFunctionImpl<DefaultBackend>;
using DefaultStreamType = DefaultBackend::StreamType;
using DefaultContextType = DefaultBackend::ContextType;
using DefaultKernelHandle = DefaultBackend::KernelHandle;

}  // namespace triton_jit
