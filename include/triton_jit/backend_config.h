/**
 * @file backend_config.h
 * @brief Backend configuration and Type Aliases
 *
 * This file provides:
 * 1. Backend selection through compile-time macros
 * 2. Type aliases for user code compatibility
 * 3. Default backend configuration
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include "triton_jit/backends/cuda_backend.h"
// Future backends can be added here:
// #include "triton_jit/backends/npu_backend.h"
// #include "triton_jit/backends/rocm_backend.h"

namespace triton_jit {

// Forward declarations of template classes
template<BackendPolicy Backend>
class TritonKernelImpl;

template<BackendPolicy Backend>
class TritonJITFunctionImpl;

// ========== Backend Selection ==========

/**
 * @brief Compile-time backend selection
 *
 * The backend is selected via CMake option:
 *   -DBACKEND=CUDA  (default for this version)
 *   -DBACKEND=NPU   (future support)
 *
 * The selected backend determines:
 * - StreamType (CUstream, aclrtStream, etc.)
 * - Kernel loading mechanism
 * - Launch API
 */

#if defined(BACKEND_CUDA)
    /// Default backend for CUDA
    using DefaultBackend = CudaBackend;

    #define BACKEND_NAME "CUDA"
    #define BACKEND_VERSION "2.0.0-cuda"

#elif defined(BACKEND_NPU)
    /// Default backend for NPU (Ascend)
    // Future implementation
    // using DefaultBackend = NpuBackend;
    #error "NPU Backend not yet implemented. Use BACKEND_CUDA for now."

    #define BACKEND_NAME "NPU"
    #define BACKEND_VERSION "2.0.0-npu"

#else
    // Default to CUDA if no backend specified
    #warning "No backend specified, defaulting to CUDA. Use -DBACKEND=CUDA explicitly."
    using DefaultBackend = CudaBackend;

    #define BACKEND_NAME "CUDA"
    #define BACKEND_VERSION "2.0.0-cuda-default"

#endif

// ========== Type Aliases for User Code ==========

/**
 * @brief Type aliases for backward compatibility
 *
 * User code can continue using TritonKernel and TritonJITFunction
 * without template parameters. The backend is selected at compile time.
 */

/// Kernel class with default backend
using TritonKernel = TritonKernelImpl<DefaultBackend>;

/// JIT function class with default backend
using TritonJITFunction = TritonJITFunctionImpl<DefaultBackend>;

/// Stream type for the default backend
using DefaultStreamType = DefaultBackend::StreamType;

/// Context type for the default backend
using DefaultContextType = DefaultBackend::ContextType;

/// Kernel handle type for the default backend
using DefaultKernelHandle = DefaultBackend::KernelHandle;

// ========== Backend Info ==========

/**
 * @brief Get backend name at runtime
 * @return Backend name string
 */
inline const char* get_backend_name() {
    return BACKEND_NAME;
}

/**
 * @brief Get backend version at runtime
 * @return Backend version string
 */
inline const char* get_backend_version() {
    return BACKEND_VERSION;
}

/**
 * @brief Print backend information
 */
inline void print_backend_info() {
    std::cout << "=== Triton JIT Backend Info ===" << std::endl;
    std::cout << "Backend: " << get_backend_name() << std::endl;
    std::cout << "Version: " << get_backend_version() << std::endl;
    std::cout << "===============================" << std::endl;
}

} // namespace triton_jit
