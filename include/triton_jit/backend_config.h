// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#if defined(BACKEND_NPU)
#include "triton_jit/backends/npu_backend.h"
#elif defined(BACKEND_MUSA)
#include "triton_jit/backends/musa_backend.h"
#elif defined(BACKEND_MACA)
#include "triton_jit/backends/maca_backend.h"
#elif defined(BACKEND_IX)
#include "triton_jit/backends/ix_backend.h"
#elif defined(BACKEND_MLU)
#include "triton_jit/backends/mlu_backend.h"
#elif defined(BACKEND_GCU)
#include "triton_jit/backends/gcu_backend.h"
#elif defined(BACKEND_HCU)
#include "triton_jit/backends/hcu_backend.h"
#else
#include "triton_jit/backends/cuda_backend.h"
#endif

namespace triton_jit {

#if defined(BACKEND_NPU)
/// Default backend for NPU (Ascend)
using DefaultBackend = NpuBackend;

#elif defined(BACKEND_MUSA)
/// Default backend for MUSA (Moore Threads)
using DefaultBackend = MusaBackend;

#elif defined(BACKEND_MACA)
/// Default backend for MACA (MetaX)
using DefaultBackend = MacaBackend;

#elif defined(BACKEND_MLU)
/// Default backend for MLU (Cambricon)
using DefaultBackend = MluBackend;

#elif defined(BACKEND_CUDA)
/// Default backend for CUDA
using DefaultBackend = CudaBackend;

#elif defined(BACKEND_IX)
/// Default backend for IX (Tianshu)
using DefaultBackend = IxBackend;

#elif defined(BACKEND_GCU)
/// Default backend for GCU (Enflame)
using DefaultBackend = GcuBackend;

#elif defined(BACKEND_HCU)
/// Default backend for HCU (Hygon, HIP-compatible)
using DefaultBackend = HcuBackend;

#else
// Default to CUDA if no backend specified
#warning "No backend specified, defaulting to CUDA. Use -DBACKEND=CUDA explicitly."
using DefaultBackend = CudaBackend;

#endif

// Forward declarations of template classes
template <BackendPolicy Backend>
class TritonKernelImpl;

template <BackendPolicy Backend>
class TritonJITFunctionImpl;

using TritonKernel = TritonKernelImpl<DefaultBackend>;
using TritonJITFunction = TritonJITFunctionImpl<DefaultBackend>;
using DefaultStreamType = DefaultBackend::StreamType;
using DefaultContextType = DefaultBackend::ContextType;
using DefaultKernelHandle = DefaultBackend::KernelHandle;

}  // namespace triton_jit
