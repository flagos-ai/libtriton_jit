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

#include <mcr/mc_runtime.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

struct MacaKernelMetadata {
  unsigned int shared;
};

struct MacaBackend {
  using StreamType = mcStream_t;
  using ContextType = mcCtx_t;
  using KernelHandle = mcFunction_t;

  static constexpr unsigned int WARP_SIZE = 64;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
  };

  struct ModuleData {
    mcModule_t module;
    mcFunction_t function;
    MacaKernelMetadata metadata;
  };

  static inline std::unordered_map<std::string, ModuleData> module_cache_;
  static inline std::mutex cache_mutex_;

  static LaunchOptions prepare_launch(const std::string& /*dir*/,
                                      const std::string& /*name*/,
                                      unsigned int shared_mem,
                                      const std::string& /*sig*/,
                                      size_t /*num_args*/) {
    return {.shared_memory = shared_mem};
  }

  static void launch_kernel(mcStream_t stream,
                            mcFunction_t kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    if (grid_x * grid_y * grid_z == 0) {
      return;
    }

    checkMacaErrors(mcModuleLaunchKernel(kernel,
                                         grid_x,
                                         grid_y,
                                         grid_z,
                                         block_x,
                                         block_y,
                                         block_z,
                                         opts.shared_memory,
                                         stream,
                                         args,
                                         nullptr));
  }

  static void ensure_context() {
    mcCtx_t ctx = nullptr;
    checkMacaErrors(mcCtxGetCurrent(&ctx));
    if (!ctx) {
      int device = 0;
      checkMacaErrors(mcGetDevice(&device));
      checkMacaErrors(mcDevicePrimaryCtxRetain(&ctx, device));
      checkMacaErrors(mcCtxSetCurrent(ctx));
    }
  }

  static int get_device_index() {
    int device = 0;
    checkMacaErrors(mcGetDevice(&device));
    return device;
  }

  static mcFunction_t load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.function;
    }

    GpuKernelMeta gpu_meta = load_gpu_metadata(dir, kernel_name);
    MacaKernelMetadata metadata {.shared = gpu_meta.shared};

    std::string mcfatbin_path = fmt::format("{}/{}.mcfatbin", dir, kernel_name);
    if (!std::filesystem::exists(mcfatbin_path)) {
      throw std::runtime_error(fmt::format("No MACA mcfatbin found for kernel {} in {}", kernel_name, dir));
    }

    std::ifstream file(mcfatbin_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error(fmt::format("Failed to open mcfatbin file: {}", mcfatbin_path));
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
      throw std::runtime_error(fmt::format("Failed to read mcfatbin file: {}", mcfatbin_path));
    }

    mcModule_t module = nullptr;
    mcFunction_t function = nullptr;
    checkMacaErrors(mcModuleLoadData(&module, data.data()));
    checkMacaErrors(mcModuleGetFunction(&function, module, kernel_name.c_str()));

    module_cache_[key] = ModuleData {module, function, metadata};
    return function;
  }

  static unsigned int get_shared_memory(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.metadata.shared;
    }

    return load_shared_memory(dir, kernel_name);
  }
};

static_assert(BackendPolicy<MacaBackend>, "MacaBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
