#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "c10/util/Logging.h"
#include "cn_api.h"
#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

// Check grid size.
static inline void check_grid(int grid, int max_grid) {
  if (grid > max_grid) {
    throw std::runtime_error(
        fmt::format("MLU kernel OutOfResources: grid size, Required: {}, Hardware limit: {}",
                    grid,
                    max_grid));
  }
}

struct MluBackend {
  using StreamType = CNqueue;
  using ContextType = CNcontext;
  using KernelHandle = void*;

  // MLU does not use this parameter, but keep this for compatibility.
  static constexpr unsigned int WARP_SIZE = 1;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
    int num_warps;
    int promote_shared;
  };

  struct ModuleData {
    CNmodule module;
    CNkernel function;
    MluKernelMetadata metadata;
  };

  static inline std::unordered_map<std::string, ModuleData> module_cache_;
  static inline std::mutex cache_mutex_;

  static LaunchOptions prepare_launch(const std::string& dir,
                                      const std::string& name,
                                      unsigned int shared_mem,
                                      const std::string& sig,
                                      size_t num_args) {
    MluKernelMetadata metadata;
    std::string key = fmt::format("{}::{}", dir, name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      metadata = it->second.metadata;
    } else {
      throw std::runtime_error(
          fmt::format("Launch kernel failed, cannot find kernel metadata from path: {}/{}", dir, name));
    }
    return {.shared_memory = shared_mem,
            .num_warps = metadata.num_warps,
            .promote_shared = metadata.promote_shared};
  }

  static void launch_kernel(CNqueue stream,
                            void* kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    grid_x *= opts.num_warps;
    auto gridMultipication = grid_x * grid_y * grid_z;

    MluDeviceInfo info;
    get_device_properties(info);

    check_grid(grid_x, info.max_grid_x);
    check_grid(grid_y, info.max_grid_y);
    check_grid(grid_z, info.max_grid_z);

    uint64_t func_type =
        ((opts.num_warps == 1) && (grid_x % info.core_num_per_cluster == 0) &&
         (opts.promote_shared == 1 || gridMultipication <= info.cluster_num * info.core_num_per_cluster))
            ? info.core_num_per_cluster
            : opts.num_warps;

    CNresult result = cnInvokeKernel((CNkernel)kernel,
                                     grid_x,
                                     grid_y,
                                     grid_z,
                                     (KernelClass)func_type,
                                     0,
                                     stream,
                                     args,
                                     NULL);

    if (result != CN_SUCCESS) {
      const char* error_string;
      cnGetErrorString(result, &error_string);
      throw std::runtime_error(fmt::format("MLU kernel launch failed: {}", error_string));
    }
  }

  static void ensure_context() {
    CNcontext ctx;
    CNresult result = cnCtxGetCurrent(&ctx);

    if (result != CN_SUCCESS || ctx == nullptr) {
      LOG(WARNING) << "No MLU context found. Creating default context.";
      CNdev device;
      checkMluErrors(cnDeviceGet(&device, 0));
      checkMluErrors(cnCtxCreate(&ctx, 0, device));
      checkMluErrors(cnCtxSetCurrent(ctx));
    }
  }

  static int get_device_index() {
    CNdev device;
    CNresult result = cnCtxGetDevice(&device);

    if (result != CN_SUCCESS) {
      const char* error_string;
      cnGetErrorString(result, &error_string);
      throw std::runtime_error(fmt::format("Failed to get MLU device: {}", error_string));
    }
    return static_cast<int>(device);
  }

  static void* load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.function;
    }

    // Load metadata.
    MluKernelMetadata metadata = load_mlu_metadata(dir, kernel_name);

    if (metadata.arch == 0) {
      throw std::runtime_error(fmt::format("Failed to load metadata for kernel: {}", kernel_name));
    }

    LOG(INFO) << fmt::format("Loading kernel {}", kernel_name);

    // Check architecture compatibility.
    CNdev device;
    checkMluErrors(cnCtxGetDevice(&device));

    int device_arch = 0;
    checkMluErrors(cnDeviceGetAttribute(&device_arch, CN_DEVICE_ATTRIBUTE_MLU_ISA_VERSION, device));

    if (device_arch != metadata.arch) {
      throw std::runtime_error("Compute architecture mismatch!");
    }

    // Load module.
    std::string cnbin_path = fmt::format("{}/{}.cnbin", dir, kernel_name);
    LOG(INFO) << fmt::format("Loading cnbin from {}", cnbin_path);

    CNmodule module;
    checkMluErrors(cnModuleLoad(cnbin_path.c_str(), &module));

    // Get function.
    CNkernel kernel;
    checkMluErrors(cnModuleGetKernel(module, kernel_name.c_str(), &kernel));

    // Cache the module and function.
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

    // If not in cache, load metadata.
    return load_shared_memory(dir, kernel_name);
  }

 private:
  struct MluDeviceInfo {
    int max_grid_x;
    int max_grid_y;
    int max_grid_z;
    int core_num_per_cluster;
    int cluster_num;
  };

  // Get neccessary device properties for kernel launch.
  static void get_device_properties(MluDeviceInfo& info) {
    int device_id = get_device_index();
    CNdev device;
    cnDeviceGet(&device, device_id);
    cnDeviceGetAttribute(&info.max_grid_x, CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_X, device);
    cnDeviceGetAttribute(&info.max_grid_y, CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_Y, device);
    cnDeviceGetAttribute(&info.max_grid_z, CN_DEVICE_ATTRIBUTE_MAX_BLOCK_TASK_DIM_Z, device);
    cnDeviceGetAttribute(&info.core_num_per_cluster, CN_DEVICE_ATTRIBUTE_MAX_CORE_COUNT_PER_CLUSTER, device);
    cnDeviceGetAttribute(&info.cluster_num, CN_DEVICE_ATTRIBUTE_MAX_CLUSTER_COUNT, device);
  };
};

static_assert(BackendPolicy<MluBackend>, "MluBackend must satisfy BackendPolicy concept");
}  // namespace triton_jit
