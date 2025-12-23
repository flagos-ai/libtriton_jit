#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {

// Forward declaration
template<BackendPolicy Backend>
class TritonJITFunctionImpl;

// Forward declaration for NpuBackend detection
struct NpuBackend;

/**
 * @brief Type trait to check if Backend is NpuBackend
 */
template<typename T>
struct is_npu_backend : std::false_type {};

template<>
struct is_npu_backend<NpuBackend> : std::true_type {};

template<typename T>
inline constexpr bool is_npu_backend_v = is_npu_backend<T>::value;

template<BackendPolicy Backend>
class TritonKernelImpl {
private:
    std::string dir_;
    std::string kernel_name_;
    mutable bool loaded_ = false;
    mutable typename Backend::KernelHandle kernel_handle_;

public:
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

    /**
     * @brief Launch kernel (original interface for CUDA/IX compatibility)
     */
    void launch(
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        int num_warps,
        typename Backend::StreamType stream,
        void** args
    ) const {
        // For NPU backend, use the signature-based launch with empty signature
        // This will trigger signature parsing if arg_layout is available
        launch_with_signature(grid_x, grid_y, grid_z, num_warps, stream, args, "");
    }

    /**
     * @brief Launch kernel with signature for NPU backend
     *
     * @param signature Full signature string (e.g., "*fp32:16,*fp32,i64,1024")
     */
    void launch_with_signature(
        unsigned int grid_x,
        unsigned int grid_y,
        unsigned int grid_z,
        int num_warps,
        typename Backend::StreamType stream,
        void** args,
        const std::string& signature
    ) const {
        // Lazy initialization
        lazy_init_handle();

        // Calculate block dimensions using backend-specific warp size
        unsigned int block_x = num_warps * Backend::WARP_SIZE;
        unsigned int block_y = 1;
        unsigned int block_z = 1;

        // Get shared memory size from backend
        unsigned int shared_memory = Backend::get_shared_memory(dir_, kernel_name_);

        // Launch kernel using backend policy
        if constexpr (is_npu_backend_v<Backend>) {
            // NPU backend: pass signature and try to get arg_layout from metadata
            const auto* metadata = Backend::get_kernel_metadata(dir_, kernel_name_);
            const auto* arg_layout = (metadata && metadata->has_arg_layout())
                                   ? &(metadata->arg_layout)
                                   : nullptr;

            Backend::launch_kernel(
                stream,
                kernel_handle_,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                args,
                shared_memory,
                signature,
                arg_layout
            );
        } else {
            // CUDA/IX backend: use standard launch
            Backend::launch_kernel(
                stream,
                kernel_handle_,
                grid_x, grid_y, grid_z,
                block_x, block_y, block_z,
                args,
                shared_memory
            );
        }
    }

    const std::string& get_dir() const {
        return dir_;
    }

    const std::string& get_kernel_name() const {
        return kernel_name_;
    }

    bool is_loaded() const {
        return loaded_;
    }

private:
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
