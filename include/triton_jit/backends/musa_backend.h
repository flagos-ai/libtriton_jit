#pragma once

#include <musa.h>
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

struct MusaKernelMetadata {
  unsigned int shared;
  // MUSA may have its own architecture identifier, but we'll keep this for compatibility
  unsigned int arch;
};

struct MusaBackend {
  using StreamType = MUstream;
  using ContextType = MUcontext;
  using KernelHandle = MUfunction;

  // MUSA warp size: Triton MUSA backend uses 32, matching CUDA convention
  // Note: Actual MUSA hardware may have different warp size, but Triton compiles with 32
  static constexpr unsigned int WARP_SIZE = 32;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
  };

  struct ModuleData {
    MUmodule module;
    MUfunction function;
    MusaKernelMetadata metadata;
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

  static void launch_kernel(MUstream stream,
                            MUfunction kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned block_x,
                            unsigned block_y,
                            unsigned block_z,
                            void** args,
                            const LaunchOptions& opts) {
    MUresult result = muLaunchKernel(kernel,
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

    if (result != MUSA_SUCCESS) {
      const char* error_string;
      muGetErrorString(result, &error_string);
      throw std::runtime_error(fmt::format("MUSA kernel launch failed: {}", error_string));
    }
  }

  static void ensure_context() {
    // When using PyTorch with a MUSA backend, the context is typically already initialized.
    MUcontext ctx;
    MUresult result = muCtxGetCurrent(&ctx);

    if (result != MUSA_SUCCESS || ctx == nullptr) {
      LOG(WARNING) << "No MUSA context found. Creating default context.";
      MUdevice device;
      checkMusaErrors(muDeviceGet(&device, 0));
      checkMusaErrors(muCtxCreate(&ctx, 0, device));
    }
  }

  static int get_device_index() {
    MUdevice device;
    MUresult result = muCtxGetDevice(&device);

    if (result != MUSA_SUCCESS) {
      const char* error_string;
      muGetErrorString(result, &error_string);
      throw std::runtime_error(fmt::format("Failed to get MUSA device: {}", error_string));
    }
    return static_cast<int>(device);
  }

  static MUfunction load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.function;
    }

    // Load metadata from .json file
    GpuKernelMeta gpu_meta = load_gpu_metadata(dir, kernel_name);
    MusaKernelMetadata metadata;
    metadata.shared = gpu_meta.shared;

    // Try to load pre-compiled binaries in priority order: .mubin, .o (ELF), .so, .llir
    MUmodule module = nullptr;
    std::string mubin_path = fmt::format("{}/{}.mubin", dir, kernel_name);
    std::string obj_path = fmt::format("{}/{}.o", dir, kernel_name);
    std::string so_path = fmt::format("{}/{}.so", dir, kernel_name);
    std::string llir_path = fmt::format("{}/{}.llir", dir, kernel_name);

    if (std::filesystem::exists(mubin_path)) {
      // Read mubin file as binary
      std::ifstream mubin_file(mubin_path, std::ios::binary | std::ios::ate);
      if (!mubin_file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open mubin file: {}", mubin_path));
      }

      std::streamsize size = mubin_file.tellg();
      mubin_file.seekg(0, std::ios::beg);

      std::vector<char> mubin_data(size);
      if (!mubin_file.read(mubin_data.data(), size)) {
        throw std::runtime_error(fmt::format("Failed to read mubin file: {}", mubin_path));
      }

      // Use muModuleLoadData to load the compiled binary
      checkMusaErrors(muModuleLoadData(&module, mubin_data.data()));

    } else if (std::filesystem::exists(obj_path)) {
      // Use muModuleLoad to load the ELF object file by path
      checkMusaErrors(muModuleLoad(&module, obj_path.c_str()));

    } else if (std::filesystem::exists(so_path)) {
      checkMusaErrors(muModuleLoad(&module, so_path.c_str()));

    } else if (std::filesystem::exists(llir_path)) {
      // Read LLIR file
      std::ifstream llir_file(llir_path, std::ios::binary);
      if (!llir_file.is_open()) {
        throw std::runtime_error(fmt::format("Failed to open LLIR file: {}", llir_path));
      }

      std::string llir_code((std::istreambuf_iterator<char>(llir_file)), std::istreambuf_iterator<char>());

      // Use muModuleLoadData for runtime JIT compilation
      checkMusaErrors(muModuleLoadData(&module, llir_code.c_str()));

    } else {
      throw std::runtime_error(
          fmt::format("No binary (.mubin, .o, .so) or LLIR found for kernel {} in {}", kernel_name, dir));
    }

    // Get function handle
    MUfunction kernel;
    checkMusaErrors(muModuleGetFunction(&kernel, module, kernel_name.c_str()));

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
};

static_assert(BackendPolicy<MusaBackend>, "MusaBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
