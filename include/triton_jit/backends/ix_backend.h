#pragma once

#include <cuda.h>
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


struct IxKernelMetadata {
    unsigned int shared;
    unsigned int arch;
};


struct IxBackend {
    using StreamType = CUstream;
    using ContextType = CUcontext;
    using KernelHandle = CUfunction;

    // IX (Tianshu) warp size is 64 threads
    static constexpr unsigned int WARP_SIZE = 64;

    struct ModuleData {
        CUmodule module;
        CUfunction function;
        IxKernelMetadata metadata;
    };

    static inline std::unordered_map<std::string, ModuleData> module_cache_;
    static inline std::mutex cache_mutex_;

    static void launch_kernel(
        CUstream stream,
        CUfunction kernel,
        unsigned grid_x, unsigned grid_y, unsigned grid_z,
        unsigned block_x, unsigned block_y, unsigned block_z,
        void** args,
        unsigned int shared_memory = 0
    ) {
        LOG(INFO) << "cuLaunchKernel (IX Backend)";

        CUresult result = cuLaunchKernel(
            kernel,
            grid_x, grid_y, grid_z,        // Grid dimensions
            block_x, block_y, block_z,     // Block dimensions
            shared_memory,                 // Shared memory
            stream,                        // Stream
            args,                          // Arguments
            nullptr                        // Extra
        );

        if (result != CUDA_SUCCESS) {
            const char* error_string;
            cuGetErrorString(result, &error_string);
            throw std::runtime_error(
                fmt::format("IX kernel launch failed: {}", error_string)
            );
        }
    }

    static void ensure_context() {
        // When using PyTorch, the CUDA context is typically already initialized
        // This function primarily serves as a validation step
        CUcontext ctx;
        CUresult result = cuCtxGetCurrent(&ctx);

        if (result != CUDA_SUCCESS || ctx == nullptr) {
            // If no context exists, we could create one, but this is unusual
            // in PyTorch workflows. Log a warning.
            LOG(WARNING) << "No IX context found. Creating default context.";

            CUdevice device;
            checkCudaErrors(cuDeviceGet(&device, 0));
            checkCudaErrors(cuCtxCreate(&ctx, 0, device));
        }
    }

    static int get_device_index() {
        CUdevice device;
        CUresult result = cuCtxGetDevice(&device);

        if (result != CUDA_SUCCESS) {
            const char* error_string;
            cuGetErrorString(result, &error_string);
            throw std::runtime_error(
                fmt::format("Failed to get IX device: {}", error_string)
            );
        }

        return static_cast<int>(device);
    }

    static CUfunction load_kernel(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check cache first
        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            return it->second.function;
        }

        // Load metadata
        GpuKernelMeta gpu_meta = load_gpu_metadata(dir, kernel_name);
        IxKernelMetadata metadata;
        metadata.shared = gpu_meta.shared;
        metadata.arch = gpu_meta.arch;

        if (metadata.arch == 0) {
            throw std::runtime_error(
                fmt::format("Failed to load metadata for kernel: {}", kernel_name));
        }

        LOG(INFO) << fmt::format(
            "Loading IX kernel {} with arch={}, shared={}",
            kernel_name, metadata.arch, metadata.shared);

        // Note: For IX platform, architecture validation may differ from CUDA
        // Skip strict architecture check for now, as IX may report different values
        // TODO: Add IX-specific architecture validation if needed

        // Load module
        std::string cubin_path = fmt::format("{}/{}.cubin", dir, kernel_name);
        LOG(INFO) << fmt::format("Loading cubin from {}", cubin_path);

        CUmodule module;
        checkCudaErrors(cuModuleLoad(&module, cubin_path.c_str()));

        // Get function
        CUfunction kernel;
        checkCudaErrors(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

        // Configure shared memory if needed
        configure_shared_memory(kernel, metadata.shared);

        // Cache the module and function
        module_cache_[key] = ModuleData{module, kernel, metadata};

        return kernel;
    }

    static unsigned int get_shared_memory(
        const std::string& dir,
        const std::string& kernel_name
    ) {
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
    static void configure_shared_memory(
        CUfunction kernel,
        unsigned int required_shared
    ) {
        CUdevice device;
        checkCudaErrors(cuCtxGetDevice(&device));

        // Check shared memory limits
        int shared_optin;
        cuDeviceGetAttribute(
            &shared_optin,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device);

        if (required_shared > static_cast<unsigned int>(shared_optin)) {
            throw std::runtime_error(fmt::format(
                "OutOfResources: Requested shared memory ({} bytes) "
                "exceeds GPU's maximum ({} bytes)",
                required_shared, shared_optin));
        }

        // Configure for large shared memory (> 48KB)
        if (required_shared > 49152 && shared_optin > 49152) {
            LOG(INFO) << fmt::format(
                "Configuring large shared memory: required={}, max={}",
                required_shared, shared_optin);

            checkCudaErrors(cuFuncSetCacheConfig(
                kernel, CU_FUNC_CACHE_PREFER_SHARED));

            int shared_total, shared_static;
            checkCudaErrors(cuDeviceGetAttribute(
                &shared_total,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                device));

            checkCudaErrors(cuFuncGetAttribute(
                &shared_static,
                CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                kernel));

            LOG(INFO) << fmt::format(
                "Shared memory - total: {}, static: {}",
                shared_total, shared_static);

            checkCudaErrors(cuFuncSetAttribute(
                kernel,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_optin - shared_static));

            LOG(INFO) << fmt::format(
                "Set dynamic shared memory to {}",
                shared_optin - shared_static);
        }
    }
};

static_assert(BackendPolicy<IxBackend>, "IxBackend must satisfy BackendPolicy concept");

} // namespace triton_jit
