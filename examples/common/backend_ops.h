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
#else
#include "c10/cuda/CUDAStream.h"
#endif

namespace triton_jit::ops {

// ---- Stream type alias ----
#if defined(BACKEND_NPU)
using RawStream = aclrtStream;
#elif defined(BACKEND_MUSA)
using RawStream = musaStream_t;
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
