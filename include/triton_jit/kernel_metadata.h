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

#include <string>

#include "triton_jit/backends/npu_types.h"

namespace triton_jit {

// GPU metadata for CUDA/IX/MUSA backends
struct GpuKernelMeta {
  unsigned int shared = 0;
  unsigned int arch = 0;
};

// HCU metadata for HCU backend
struct HcuKernelMetadata {
  unsigned int shared = 0;
  std::string arch;
};

// MLU metadata for MLU backend
struct MluKernelMetadata {
  // MLU does not use `shared` but keep this for compatibility.
  unsigned int shared = 0;
  unsigned int arch = 0;
  int num_warps = 1;
  bool promote_shared = false;
};

// Load GPU kernel metadata from {dir}/{kernel_name}.json
// Returns default values if file not found.
GpuKernelMeta load_gpu_metadata(const std::string& dir, const std::string& kernel_name);

// Load NPU kernel metadata from {dir}/{kernel_name}.json
// Returns default values if file not found.
NpuKernelMetadata load_npu_metadata(const std::string& dir, const std::string& kernel_name);

HcuKernelMetadata load_hcu_metadata(const std::string& dir, const std::string& kernel_name);

// Load MLU kernel metadata from {dir}/{kernel_name}.json
// Returns default values if file not found.
MluKernelMetadata load_mlu_metadata(const std::string& dir, const std::string& kernel_name);

// Load only the shared memory field from metadata JSON.
// Returns 0 if file not found or field missing.
unsigned int load_shared_memory(const std::string& dir, const std::string& kernel_name);

}  // namespace triton_jit
