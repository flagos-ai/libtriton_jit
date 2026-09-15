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

#include "triton_jit/freeze.h"

#include <atomic>

#if defined(BACKEND_CUDA) || defined(BACKEND_IX)
#include <cuda.h>
#elif defined(BACKEND_HCU)
#include <hip/hip_runtime_api.h>
#endif

namespace triton_jit {

namespace {
  std::atomic<int> freeze_depth {0};
}  // namespace

bool is_frozen() noexcept {
  return freeze_depth.load(std::memory_order_acquire) > 0;
}

bool stream_is_capturing(void* stream) noexcept {
#if defined(BACKEND_CUDA) || defined(BACKEND_IX)
  CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
  if (cuStreamIsCapturing(static_cast<CUstream>(stream), &status) != CUDA_SUCCESS) {
    return false;
  }
  return status != CU_STREAM_CAPTURE_STATUS_NONE;
#elif defined(BACKEND_HCU)
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  if (hipStreamIsCapturing(static_cast<hipStream_t>(stream), &status) != hipSuccess) {
    return false;
  }
  return status != hipStreamCaptureStatusNone;
#else
  (void)stream;
  return false;
#endif
}

ScopedFreeze::ScopedFreeze() {
  freeze_depth.fetch_add(1, std::memory_order_acq_rel);
}

ScopedFreeze::~ScopedFreeze() {
  freeze_depth.fetch_sub(1, std::memory_order_acq_rel);
}

namespace detail {

  void refuse_if_frozen(void* stream, const std::string& what, ColdWork work) {
    const char* missing = work == ColdWork::kConfig     ? "tuned configuration"
                          : work == ColdWork::kFunction ? "kernel signature"
                                                        : "compiled/loaded GPU program";
    if (is_frozen()) {
      throw FrozenMissError(
          what + ": missing " + missing + "; ScopedFreeze is active; prepare this invocation before freezing",
          work,
          ColdRestriction::kExplicitFreeze);
    }
    if (stream_is_capturing(stream)) {
      throw FrozenMissError(what + ": missing " + missing +
                                "; the stream is being captured; prepare this invocation before capture",
                            work,
                            ColdRestriction::kCapture);
    }
  }

}  // namespace detail

}  // namespace triton_jit
