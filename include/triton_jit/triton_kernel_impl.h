/**
 * @file triton_kernel_impl.h
 * @brief Template implementation of TritonKernel with Backend Policy
 *
 * This file contains the template implementation of TritonKernelImpl,
 * which is parameterized by a BackendPolicy.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {

// Forward declaration
template<BackendPolicy Backend>
class TritonJITFunctionImpl;

/**
 * @brief Template TritonKernel class parameterized by Backend
 *
 * This class manages a single Triton kernel with lazy loading.
 * The backend determines how kernels are loaded and launched.
 *
 * @tparam Backend Backend policy (must satisfy BackendPolicy concept)
 */
template<BackendPolicy Backend>
class TritonKernelImpl {
private:
    /// Directory containing kernel files (IRs, metadata, binary)
    std::string dir_;

    /// Name of the kernel function
    std::string kernel_name_;

    /// Whether the kernel has been loaded
    mutable bool loaded_ = false;

    /// Backend-specific kernel handle
    mutable typename Backend::KernelHandle kernel_handle_;

public:
    // ========== Constructors ==========

    TritonKernelImpl() = default;

    TritonKernelImpl(std::string_view dir, std::string_view kernel_name)
        : dir_(std::string(dir))
        , kernel_name_(std::string(kernel_name))
        , loaded_(false)
    {
    }

    // Delete copy constructor and assignment
    TritonKernelImpl(const TritonKernelImpl&) = delete;
    TritonKernelImpl& operator=(const TritonKernelImpl&) = delete;

    // Default move constructor and assignment
    TritonKernelImpl(TritonKernelImpl&&) = default;
    TritonKernelImpl& operator=(TritonKernelImpl&&) = default;

    // ========== Public Methods ==========

    /**
     * @brief Launch the kernel
     *
     * This method lazily loads the kernel on first launch, then invokes
     * the backend's launch_kernel method.
     *
     * @param grid_x, grid_y, grid_z Grid dimensions
     * @param num_warps Number of warps per block
     * @param stream Backend-specific stream
     * @param args Kernel arguments (array of pointers)
     *
     * @note The block dimensions are calculated as (num_warps * 32, 1, 1)
     */
    void launch(
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        int num_warps,
        typename Backend::StreamType stream,
        void** args
    ) const {
        // Lazy initialization
        lazy_init_handle();

        // Calculate block dimensions
        unsigned int block_x = num_warps * 32;  // Each warp has 32 threads
        unsigned int block_y = 1;
        unsigned int block_z = 1;

        // Get shared memory size (if backend supports it)
        unsigned int shared_memory = 0;
        if constexpr (std::is_same_v<Backend, CudaBackend>) {
            shared_memory = Backend::get_shared_memory(dir_, kernel_name_);
        }

        // Launch kernel using backend policy
        Backend::launch_kernel(
            stream,
            kernel_handle_,
            grid_x, grid_y, grid_z,
            block_x, block_y, block_z,
            args,
            shared_memory
        );
    }

    /**
     * @brief Get kernel directory
     */
    const std::string& get_dir() const {
        return dir_;
    }

    /**
     * @brief Get kernel name
     */
    const std::string& get_kernel_name() const {
        return kernel_name_;
    }

    /**
     * @brief Check if kernel is loaded
     */
    bool is_loaded() const {
        return loaded_;
    }

private:
    /**
     * @brief Lazy load kernel handle
     *
     * This method is called on the first launch() invocation.
     * It delegates to the backend's load_kernel() method.
     *
     * Thread-safe: multiple threads can call this safely.
     */
    void lazy_init_handle() const {
        if (loaded_) {
            return;
        }

        // Note: For thread safety, the backend's load_kernel should be thread-safe
        kernel_handle_ = Backend::load_kernel(dir_, kernel_name_);
        loaded_ = true;
    }

    // Friend declaration for TritonJITFunctionImpl
    friend class TritonJITFunctionImpl<Backend>;
};

// Verify that TritonKernelImpl is move constructible
template<BackendPolicy Backend>
static inline constexpr bool is_triton_kernel_move_constructible_v =
    std::is_move_constructible_v<TritonKernelImpl<Backend>>;

} // namespace triton_jit
