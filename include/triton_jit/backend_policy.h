/**
 * @file backend_policy.h
 * @brief Backend Policy Concept definition for Policy-Based Design
 *
 * This file defines the BackendPolicy concept that all backend implementations
 * must satisfy. Using C++20 Concepts provides compile-time verification and
 * clear error messages.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include <concepts>
#include <string>
#include <type_traits>

namespace triton_jit {

/**
 * @brief Backend Policy Concept
 *
 * A BackendPolicy must provide:
 * 1. Type definitions: StreamType, ContextType, KernelHandle
 * 2. Static methods: launch_kernel, ensure_context, get_device_index, load_kernel
 *
 * This concept uses C++20 features for clear, compile-time verification.
 *
 * Example usage:
 * @code
 * struct MyBackend {
 *     using StreamType = CUstream;
 *     using ContextType = CUcontext;
 *     using KernelHandle = CUfunction;
 *
 *     static void launch_kernel(...) { ... }
 *     static void ensure_context() { ... }
 *     static int get_device_index() { ... }
 *     static KernelHandle load_kernel(...) { ... }
 * };
 *
 * static_assert(BackendPolicy<MyBackend>);
 * @endcode
 */
template<typename T>
concept BackendPolicy = requires {
    // ========== Required Type Definitions ==========

    /// Stream type used by the backend (e.g., CUstream for CUDA, aclrtStream for NPU)
    typename T::StreamType;

    /// Context type used by the backend (e.g., CUcontext for CUDA, aclrtContext for NPU)
    typename T::ContextType;

    /// Kernel handle type (e.g., CUfunction for CUDA, void* for NPU)
    typename T::KernelHandle;

} && requires(
    typename T::StreamType stream,
    typename T::KernelHandle kernel,
    unsigned grid_x, unsigned grid_y, unsigned grid_z,
    unsigned block_x, unsigned block_y, unsigned block_z,
    void** args
) {
    // ========== Required Static Methods ==========

    /**
     * Launch a kernel on the device
     *
     * @param stream Device stream to launch on
     * @param kernel Kernel handle
     * @param grid_x, grid_y, grid_z Grid dimensions
     * @param block_x, block_y, block_z Block dimensions
     * @param args Kernel arguments (array of pointers)
     */
    { T::launch_kernel(stream, kernel,
                       grid_x, grid_y, grid_z,
                       block_x, block_y, block_z,
                       args) } -> std::same_as<void>;

    /**
     * Ensure device context is properly initialized
     * This may be a no-op if the framework (e.g., PyTorch) manages context
     */
    { T::ensure_context() } -> std::same_as<void>;

    /**
     * Get current device index
     * @return Device index (e.g., 0, 1, 2...)
     */
    { T::get_device_index() } -> std::same_as<int>;

} && requires(const std::string& dir, const std::string& name) {
    /**
     * Load a kernel from compiled binary
     *
     * @param dir Directory containing kernel binary
     * @param kernel_name Name of the kernel function
     * @return Kernel handle
     */
    { T::load_kernel(dir, name) } -> std::same_as<typename T::KernelHandle>;

    /**
     * Get shared memory size required by kernel
     *
     * @param dir Directory containing kernel binary
     * @param kernel_name Name of the kernel function
     * @return Shared memory size in bytes (0 if not applicable)
     */
    { T::get_shared_memory(dir, name) } -> std::same_as<unsigned int>;
};

/**
 * @brief Helper concept: Check if a type has StreamType
 */
template<typename T>
concept HasStreamType = requires {
    typename T::StreamType;
};

/**
 * @brief Helper concept: Check if a type has ContextType
 */
template<typename T>
concept HasContextType = requires {
    typename T::ContextType;
};

/**
 * @brief Helper concept: Check if a type has KernelHandle
 */
template<typename T>
concept HasKernelHandle = requires {
    typename T::KernelHandle;
};

} // namespace triton_jit
