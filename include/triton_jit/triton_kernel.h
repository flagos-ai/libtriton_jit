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
template <BackendPolicy Backend>
class TritonJITFunctionImpl;

template <BackendPolicy Backend>
class TritonKernelImpl {
 private:
  std::string dir_;
  std::string kernel_name_;
  mutable bool loaded_ = false;
  mutable typename Backend::KernelHandle kernel_handle_;

 public:
  TritonKernelImpl() = default;

  TritonKernelImpl(std::string_view dir, std::string_view kernel_name)
      : dir_(std::string(dir)), kernel_name_(std::string(kernel_name)), loaded_(false) {
  }

  // Delete copy constructor and assignment
  TritonKernelImpl(const TritonKernelImpl&) = delete;
  TritonKernelImpl& operator=(const TritonKernelImpl&) = delete;

  // Default move constructor and assignment
  TritonKernelImpl(TritonKernelImpl&&) = default;
  TritonKernelImpl& operator=(TritonKernelImpl&&) = default;

  /**
   * @brief Launch kernel (convenience wrapper with empty signature)
   */
  void launch(unsigned int grid_x,
              unsigned int grid_y,
              unsigned int grid_z,
              int num_warps,
              typename Backend::StreamType stream,
              void** args) const {
    launch_with_signature(grid_x, grid_y, grid_z, num_warps, stream, args, "");
  }

  /**
   * @brief Launch kernel with signature
   *
   * @param signature Full signature string (e.g., "*fp32:16,*fp32,i64,1024")
   */
  void launch_with_signature(unsigned int grid_x,
                             unsigned int grid_y,
                             unsigned int grid_z,
                             int num_warps,
                             typename Backend::StreamType stream,
                             void** args,
                             const std::string& signature,
                             size_t num_args = 0) const {
    // Lazy initialization
    lazy_init_handle();

    // Calculate block dimensions using backend-specific warp size
    unsigned int block_x = num_warps * Backend::WARP_SIZE;
    unsigned int block_y = 1;
    unsigned int block_z = 1;

    // Get shared memory size from backend
    unsigned int shared_memory = Backend::get_shared_memory(dir_, kernel_name_);

    // Prepare backend-specific launch options (no branching)
    auto opts = Backend::prepare_launch(dir_, kernel_name_, shared_memory, signature, num_args);

    // Launch kernel using backend policy (unified interface)
    Backend::launch_kernel(stream,
                           kernel_handle_,
                           grid_x,
                           grid_y,
                           grid_z,
                           block_x,
                           block_y,
                           block_z,
                           args,
                           opts);
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

  // Friend declaration for TritonJITFunction
  friend class TritonJITFunctionImpl<Backend>;
};

// Verify that TritonKernelImpl is move constructible
template <BackendPolicy Backend>
static inline constexpr bool is_triton_kernel_move_constructible_v =
    std::is_move_constructible_v<TritonKernelImpl<Backend>>;

}  // namespace triton_jit
