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

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {

// Metadata available at the C++ runtime's kernel dispatch point. Hooks run
// synchronously, but the strings are owned so callers may safely retain a copy.
struct LaunchMetadata {
  std::string kernel_name;
  unsigned int grid_x = 0;
  unsigned int grid_y = 0;
  unsigned int grid_z = 0;
  int num_warps = 0;
  unsigned int shared_memory = 0;
  std::string signature;
  // Null when Backend::StreamType is not pointer-representable.
  void* stream = nullptr;
};

// Hooks execute synchronously on the launching thread and may be invoked
// concurrently by multiple launch threads. Callbacks are responsible for
// synchronizing their own state.
//
// Exceptions propagate to the caller. If enter throws, the kernel is not
// submitted. Exit runs only after Backend::launch_kernel returns successfully;
// if exit throws, the kernel has already been submitted.
using LaunchHook = std::function<void(const LaunchMetadata&)>;

// Hook updates affect subsequent launches. An in-flight launch retains the
// immutable enter/exit snapshot acquired before invoking enter, so a hook may
// safely update or clear the process-wide hooks.
void set_launch_enter_hook(LaunchHook hook);
void set_launch_exit_hook(LaunchHook hook);
void clear_launch_hooks();

namespace detail {

struct LaunchHooksState {
  LaunchHook enter;
  LaunchHook exit;
};

using LaunchHooksSnapshot = std::shared_ptr<const LaunchHooksState>;

LaunchHooksSnapshot get_launch_hooks_snapshot();

}  // namespace detail

// Forward declaration
template <BackendPolicy Backend>
class TritonJITFunctionImpl;

template <BackendPolicy Backend>
class TritonKernelImpl {
 private:
  std::string dir_;
  std::string kernel_name_;
  mutable bool loaded_ = false;
  mutable unsigned int cached_shared_memory_ = 0;
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

    // Prepare backend-specific launch options (no branching)
    auto opts = Backend::prepare_launch(dir_, kernel_name_, cached_shared_memory_, signature, num_args);

    // Take one immutable snapshot for the whole launch. Updating or clearing the
    // process-wide hooks from another thread (or from a hook itself) only affects
    // subsequent launches.
    detail::LaunchHooksSnapshot hooks = detail::get_launch_hooks_snapshot();
    LaunchMetadata metadata;
    if (hooks) {
      metadata.kernel_name = kernel_name_;
      metadata.grid_x = grid_x;
      metadata.grid_y = grid_y;
      metadata.grid_z = grid_z;
      metadata.num_warps = num_warps;
      metadata.shared_memory = cached_shared_memory_;
      metadata.signature = signature;
      if constexpr (std::is_pointer_v<typename Backend::StreamType>) {
        metadata.stream = reinterpret_cast<void*>(stream);
      }
      if (hooks->enter) {
        hooks->enter(metadata);
      }
    }

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

    if (hooks && hooks->exit) {
      hooks->exit(metadata);
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
    cached_shared_memory_ = Backend::get_shared_memory(dir_, kernel_name_);
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
