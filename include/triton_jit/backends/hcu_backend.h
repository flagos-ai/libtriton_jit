#pragma once

#include <hip/hip_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

struct HcuBackend {
  using StreamType = hipStream_t;
  using ContextType = hipCtx_t;
  using KernelHandle = hipFunction_t;

  // HCU warp size is always 64 threads (same as AMD GCN/CDNA)
  static constexpr unsigned int WARP_SIZE = 64;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
  };

  struct ModuleData {
    hipModule_t module;
    hipFunction_t function;
    HcuKernelMetadata metadata;
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

  static void launch_kernel(hipStream_t stream,
                            hipFunction_t kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    LOG(INFO) << "hipModuleLaunchKernel (HCU Backend)";

    hipError_t result = hipModuleLaunchKernel(kernel,
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

    if (result != hipSuccess) {
      const char* error_string = hipGetErrorString(result);
      throw std::runtime_error(fmt::format("HCU kernel launch failed: {}", error_string));
    }
  }

  static void ensure_context() {
    // When using PyTorch with HIP, the HIP context is typically already initialized
    // This function primarily serves as a validation step
    hipDevice_t device;
    hipError_t result = hipGetDevice(&device);

    if (result != hipSuccess) {
      LOG(WARNING) << "No HCU device context found. Initializing default device.";
      checkHcuErrors(hipSetDevice(0));
    }
  }

  static int get_device_index() {
    hipDevice_t device;
    hipError_t result = hipGetDevice(&device);

    if (result != hipSuccess) {
      const char* error_string = hipGetErrorString(result);
      throw std::runtime_error(fmt::format("Failed to get HCU device: {}", error_string));
    }

    return static_cast<int>(device);
  }

  static hipFunction_t load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);

    std::lock_guard<std::mutex> lock(cache_mutex_);

    // Check cache first
    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.function;
    }

    // Load metadata
    HcuKernelMetadata metadata = load_hcu_metadata(dir, kernel_name);
    if (metadata.arch.empty()) {
      throw std::runtime_error(fmt::format("Failed to load metadata for kernel: {}", kernel_name));
    }

    LOG(INFO) << fmt::format("Loading HCU kernel {} with arch={}, shared={}",
                             kernel_name,
                             metadata.arch,
                             metadata.shared);

    // Check architecture compatibility
    hipDeviceProp_t props;
    int device_id;
    checkHcuErrors(hipGetDevice(&device_id));
    checkHcuErrors(hipGetDeviceProperties(&props, device_id));

    std::string device_arch(props.gcnArchName);
    // gcnArchName may contain extra info like ":sramecc+:xnack-", strip it
    auto colon_pos = device_arch.find(':');
    if (colon_pos != std::string::npos) {
      device_arch = device_arch.substr(0, colon_pos);
    }

    if (device_arch != metadata.arch) {
      throw std::runtime_error(
          fmt::format("Compute architecture mismatch! Device has {}, kernel requires {}",
                      device_arch,
                      metadata.arch));
    }

    // Load module from .hsaco binary
    std::string hsaco_path = fmt::format("{}/{}.hsaco", dir, kernel_name);
    LOG(INFO) << fmt::format("Loading hsaco from {}", hsaco_path);

    hipModule_t module;
    checkHcuErrors(hipModuleLoad(&module, hsaco_path.c_str()));

    // Get function
    hipFunction_t kernel;
    checkHcuErrors(hipModuleGetFunction(&kernel, module, kernel_name.c_str()));

    // Configure shared memory if needed
    configure_shared_memory(kernel, device_id, metadata.shared);

    // Cache the module and function
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

    // If not in cache, load metadata
    return load_shared_memory(dir, kernel_name);
  }

 private:
  static void configure_shared_memory(hipFunction_t kernel, int device_id, unsigned int required_shared) {
    // Query device shared memory limits
    hipDeviceProp_t props;
    checkHcuErrors(hipGetDeviceProperties(&props, device_id));

    unsigned int max_shared = static_cast<unsigned int>(props.sharedMemPerBlock);

    if (required_shared > max_shared) {
      throw std::runtime_error(
          fmt::format("OutOfResources: Requested shared memory ({} bytes) "
                      "exceeds GPU's maximum ({} bytes)",
                      required_shared,
                      max_shared));
    }

    // Configure for large shared memory (> 48KB) via HIP function attribute
    if (required_shared > 49152) {
      LOG(INFO) << fmt::format("Configuring large shared memory: required={}, max={}",
                               required_shared,
                               max_shared);

      int shared_static = 0;
      checkHcuErrors(hipFuncGetAttribute(&shared_static, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel));

      LOG(INFO) << fmt::format("Shared memory - max: {}, static: {}", max_shared, shared_static);

      checkHcuErrors(hipFuncSetAttribute(kernel,
                                         hipFuncAttributeMaxDynamicSharedMemorySize,
                                         max_shared - shared_static));

      LOG(INFO) << fmt::format("Set dynamic shared memory to {}", max_shared - shared_static);
    }
  }
};

static_assert(BackendPolicy<HcuBackend>, "HcuBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
