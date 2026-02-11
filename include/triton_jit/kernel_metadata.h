#pragma once

#include <string>

#include "triton_jit/backends/npu_types.h"

namespace triton_jit {

// GPU metadata for CUDA/IX/MUSA backends
struct GpuKernelMeta {
    unsigned int shared = 0;
    unsigned int arch = 0;
};

// Load GPU kernel metadata from {dir}/{kernel_name}.json
// Returns default values if file not found.
GpuKernelMeta load_gpu_metadata(const std::string& dir,
                                const std::string& kernel_name);

// Load NPU kernel metadata from {dir}/{kernel_name}.json
// Returns default values if file not found.
NpuKernelMetadata load_npu_metadata(const std::string& dir,
                                    const std::string& kernel_name);

// Load only the shared memory field from metadata JSON.
// Returns 0 if file not found or field missing.
unsigned int load_shared_memory(const std::string& dir,
                                const std::string& kernel_name);

} // namespace triton_jit
