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

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "c10/util/Logging.h"
#include "tops_runtime_api.h"
#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

struct GcuBackend {
  using StreamType = topsStream_t;
  using ContextType = void*;
  using KernelHandle = topsFunction_t;

  // GCU blockDimX = num_warps directly, so WARP_SIZE = 1
  static constexpr unsigned int WARP_SIZE = 1;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
  };

  static inline std::unordered_map<std::string, std::pair<topsModule_t, topsFunction_t>> module_cache_;
  static inline std::mutex cache_mutex_;

  static LaunchOptions prepare_launch(const std::string& dir,
                                      const std::string& name,
                                      unsigned int shared_mem,
                                      const std::string& sig,
                                      size_t num_args) {
    return {.shared_memory = shared_mem};
  }

  static void launch_kernel(topsStream_t stream,
                            topsFunction_t kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    LOG(INFO) << fmt::format("topsModuleLaunchKernelEx: grid=({},{},{}), num_warps={}, shmem={}",
                             grid_x, grid_y, grid_z, block_x, opts.shared_memory);

    topsLaunchConfig_t config;
    memset(&config, 0, sizeof(config));
    config.gridDim = dim3(grid_x, grid_y, grid_z);
    config.blockDim = dim3(1, 1, 1);
    config.dynamicSmemBytes = opts.shared_memory;
    config.stream = stream;

    topsLaunchAttribute_t att;
    att.id = topsLaunchAttributeThreadDimension;
    att.val.ThreadDim.x = block_x;
    att.val.ThreadDim.y = 1;
    att.val.ThreadDim.z = 1;
    config.attrs = &att;
    config.numAttrs = 1;

    topsError_t result = topsModuleLaunchKernelEx(&config, kernel, args, nullptr);

    if (result != topsSuccess) {
      throw std::runtime_error(fmt::format("GCU kernel launch failed: {} (error code {})",
                                           topsGetErrorString(result), static_cast<int>(result)));
    }
  }

  static void ensure_context() {
    // GCU uses tops runtime which manages context implicitly via topsSetDevice
  }

  static int get_device_index() {
    int device_id = 0;
    topsError_t result = topsGetDevice(&device_id);

    if (result != topsSuccess) {
      throw std::runtime_error(fmt::format("Failed to get GCU device index: {} (error code {})",
                                           topsGetErrorString(result), static_cast<int>(result)));
    }
    return device_id;
  }

  static topsFunction_t load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.second;
    }

    LOG(INFO) << fmt::format("Loading GCU kernel {}", kernel_name);

    std::string fatbin_path = fmt::format("{}/{}.fatbin", dir, kernel_name);
    LOG(INFO) << fmt::format("Loading fatbin from {}", fatbin_path);

    topsModule_t module;
    topsError_t err = topsModuleLoad(&module, fatbin_path.c_str());
    if (err != topsSuccess) {
      throw std::runtime_error(fmt::format("Failed to load GCU module from: {} ({}, error {})",
                                           fatbin_path, topsGetErrorString(err), static_cast<int>(err)));
    }

    topsFunction_t function;
    err = topsModuleGetFunction(&function, module, kernel_name.c_str());
    if (err != topsSuccess) {
      topsModuleUnload(module);
      throw std::runtime_error(fmt::format("Failed to get function '{}' from module ({}, error {})",
                                           kernel_name, topsGetErrorString(err), static_cast<int>(err)));
    }

    module_cache_[key] = {module, function};
    return function;
  }

  static unsigned int get_shared_memory(const std::string& dir, const std::string& kernel_name) {
    return load_shared_memory(dir, kernel_name);
  }
};

static_assert(BackendPolicy<GcuBackend>, "GcuBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
