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

#include <stdexcept>
#include <string>

#include <ATen/ATen.h>

#include "triton_jit/backend_config.h"

// ---- Backend-specific headers (centralized, operator files no longer need these) ----
#if defined(BACKEND_NPU)
#if __has_include("torch_npu/csrc/core/npu/NPUStream.h")
#include "torch_npu/csrc/core/npu/NPUStream.h"
#define HAS_TORCH_NPU 1
#else
#define HAS_TORCH_NPU 0
#endif
#elif defined(BACKEND_MUSA)
#include <musa_runtime.h>
#elif defined(BACKEND_MACA)
#include <mcr/mc_runtime.h>
#include "c10/cuda/CUDAStream.h"
#elif defined(BACKEND_GCU)
#include <tops_runtime_api.h>
#elif defined(BACKEND_HCU)
#include <hip/hip_runtime.h>
#include "c10/hip/HIPStream.h"
#else
#include "c10/cuda/CUDAStream.h"
#endif

namespace triton_jit::ops {

// ---- Stream type alias ----
#if defined(BACKEND_NPU)
using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
using RawStream = musaStream_t;
#elif defined(BACKEND_MACA)
using RawStream = mcStream_t;
#elif defined(BACKEND_GCU)
using RawStream = topsStream_t;
#elif defined(BACKEND_HCU)
using RawStream = hipStream_t;
#else
using RawStream = CUstream;
#endif

// ---- Stream getter ----
inline RawStream get_device_stream([[maybe_unused]] const at::Tensor& t) {
#if defined(BACKEND_NPU)
#if HAS_TORCH_NPU
  return c10_npu::getCurrentNPUStream(t.device().index()).stream();
#else
  return nullptr;
#endif
#elif defined(BACKEND_MUSA)
  return nullptr;
#elif defined(BACKEND_MACA)
  return reinterpret_cast<mcStream_t>(c10::cuda::getCurrentCUDAStream(t.device().index()).stream());
#elif defined(BACKEND_GCU)
  return nullptr;
#elif defined(BACKEND_HCU)
  return static_cast<hipStream_t>(c10::hip::getCurrentHIPStream(t.device().index()).stream());
#else
  return static_cast<CUstream>(c10::cuda::getCurrentCUDAStream(t.device().index()).stream());
#endif
}

// ---- Tensor allocation (wraps MUSA musaMalloc difference) ----
inline at::Tensor backend_empty(at::IntArrayRef sizes, at::ScalarType dtype, at::Device device) {
#if defined(BACKEND_MUSA)
  void* p = nullptr;
  size_t bytes = 1;
  for (auto s : sizes) bytes *= static_cast<size_t>(s);
  bytes *= at::elementSize(dtype);
  if (musaMalloc(&p, bytes) != musaSuccess) throw std::runtime_error("musaMalloc failed");
  return at::from_blob(
      p,
      sizes,
      [](void* ptr) { musaFree(ptr); },
      at::TensorOptions().dtype(dtype).device(device));
#else
  return at::empty(sizes, at::TensorOptions().dtype(dtype).device(device));
#endif
}

}  // namespace triton_jit::ops
