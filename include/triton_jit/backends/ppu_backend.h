#pragma once

#include <hggc.h>

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

struct PpuKernelMetadata {
  unsigned int shared;
  // PPU may have its own architecture identifier, but we'll keep this for compatibility
  unsigned int arch;
};

// PPU (T-Head XuanTie GPU, e.g. 810E) backend on the HGGC runtime (libhggc.so).
//
// The Triton PPU backend (FlagTree third_party/ppu) compiles kernels to
// .hgbin binaries; this backend loads them through the HGGC driver API and
// launches them with hgLaunchKernel. PPU follows the CUDA conventions:
// warp size 32, block dims of num_warps * 32, and trailing global/profile
// scratch pointer arguments.
struct PpuBackend {
  using StreamType = HGstream;
  using ContextType = HGcontext;
  using KernelHandle = HGfunction;

  // PPU warp size: the FlagTree ppu driver launches with blockDimX = 32 * num_warps
  static constexpr unsigned int WARP_SIZE = 32;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
  };

  struct ModuleData {
    HGmodule module;
    HGfunction function;
    PpuKernelMetadata metadata;
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

  static void launch_kernel(HGstream stream,
                            HGfunction kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    HGresult result = hgLaunchKernel(kernel,
                                     grid_x,
                                     grid_y,
                                     grid_z,  // Grid dimensions
                                     block_x,
                                     block_y,
                                     block_z,             // Block dimensions
                                     opts.shared_memory,  // Shared memory
                                     stream,              // Stream
                                     args,                // Arguments
                                     nullptr              // Extra
    );

    if (result != HGGC_SUCCESS) {
      const char* error_string;
      hgGetErrorString(result, &error_string);
      throw std::runtime_error(fmt::format("PPU kernel launch failed: {}", error_string));
    }
  }

  static void ensure_context() {
    // When using PyTorch with a PPU backend, the context is typically already
    // initialized (PPU masquerades as a CUDA device in PyTorch).
    HGcontext ctx;
    HGresult result = hgCtxGetCurrent(&ctx);

    if (result != HGGC_SUCCESS || ctx == nullptr) {
      LOG(WARNING) << "No PPU context found. Creating default context.";
      HGdevice device;
      checkHggcErrors(hgDeviceGet(&device, 0));
      checkHggcErrors(hgDevicePrimaryCtxRetain(&ctx, device));
      checkHggcErrors(hgCtxSetCurrent(ctx));
    }
  }

  static int get_device_index() {
    HGdevice device;
    HGresult result = hgCtxGetDevice(&device);

    if (result != HGGC_SUCCESS) {
      const char* error_string;
      hgGetErrorString(result, &error_string);
      throw std::runtime_error(fmt::format("Failed to get PPU device: {}", error_string));
    }
    return static_cast<int>(device);
  }

  static HGfunction load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.function;
    }

    // Load metadata from .json file
    GpuKernelMeta gpu_meta = load_gpu_metadata(dir, kernel_name);
    PpuKernelMetadata metadata;
    metadata.shared = gpu_meta.shared;
    metadata.arch = gpu_meta.arch;

    // Load the pre-compiled .hgbin produced by the Triton PPU backend
    std::string hgbin_path = fmt::format("{}/{}.hgbin", dir, kernel_name);
    if (!std::filesystem::exists(hgbin_path)) {
      throw std::runtime_error(
          fmt::format("No binary (.hgbin) found for kernel {} in {}", kernel_name, dir));
    }

    // Read hgbin file as binary
    std::ifstream hgbin_file(hgbin_path, std::ios::binary | std::ios::ate);
    if (!hgbin_file.is_open()) {
      throw std::runtime_error(fmt::format("Failed to open hgbin file: {}", hgbin_path));
    }

    std::streamsize size = hgbin_file.tellg();
    hgbin_file.seekg(0, std::ios::beg);

    std::vector<char> hgbin_data(size);
    if (!hgbin_file.read(hgbin_data.data(), size)) {
      throw std::runtime_error(fmt::format("Failed to read hgbin file: {}", hgbin_path));
    }

    // Use hgModuleLoadData to load the compiled binary
    HGmodule module = nullptr;
    checkHggcErrors(hgModuleLoadData(&module, hgbin_data.data()));

    // Get function handle
    HGfunction kernel;
    checkHggcErrors(hgModuleGetFunction(&kernel, module, kernel_name.c_str()));

    // Configure shared memory if needed
    configure_shared_memory(kernel, metadata.shared);

    // Cache the loaded module and metadata
    module_cache_[key] = ModuleData {module, kernel, metadata};

    return kernel;
  }

  static unsigned int get_shared_memory(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.metadata.shared;
    }

    // If not in cache, load from metadata file directly
    return load_shared_memory(dir, kernel_name);
  }

 private:
  static void configure_shared_memory(HGfunction kernel, unsigned int required_shared) {
    // Check shared memory limits
    HGdevice device;
    checkHggcErrors(hgDeviceGet(&device, 0));

    int shared_optin = 0;
    if (hgDeviceGetAttribute(&shared_optin,
                             HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                             device) != HGGC_SUCCESS ||
        shared_optin <= 0) {
      // Fall back to the default per-block shared memory limit
      checkHggcErrors(
          hgDeviceGetAttribute(&shared_optin, HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device));
    }

    if (required_shared > static_cast<unsigned int>(shared_optin)) {
      throw std::runtime_error(
          fmt::format("OutOfResources: Requested shared memory ({} bytes) "
                      "exceeds GPU's maximum ({} bytes)",
                      required_shared,
                      shared_optin));
    }

    // Configure for large shared memory (> 48KB)
    if (required_shared > 49152 && shared_optin > 49152) {
      LOG(INFO) << fmt::format("Configuring large shared memory: required={}, max={}",
                               required_shared,
                               shared_optin);

      checkHggcErrors(hgFuncSetCacheConfig(kernel, HG_FUNC_CACHE_PREFER_SHARED));

      int shared_total = 0;
      int shared_static = 0;
      checkHggcErrors(hgDeviceGetAttribute(&shared_total,
                                           HG_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                                           device));
      checkHggcErrors(hgFuncGetAttribute(&shared_static, HG_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));

      LOG(INFO) << fmt::format("Shared memory - total: {}, static: {}", shared_total, shared_static);

      checkHggcErrors(hgFuncSetAttribute(kernel,
                                         HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         shared_optin - shared_static));

      LOG(INFO) << fmt::format("Set dynamic shared memory to {}", shared_optin - shared_static);
    }
  }
};

static_assert(BackendPolicy<PpuBackend>, "PpuBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
