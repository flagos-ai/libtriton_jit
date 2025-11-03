/**
 * @file cuda_backend.h
 * @brief CUDA Backend Policy implementation
 *
 * This file implements the Backend Policy for CUDA using the CUDA Driver API.
 * It provides kernel launching, context management, and module loading
 * functionalities specific to CUDA.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include <cuda.h>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {

/**
 * @brief Kernel metadata structure
 *
 * Stores metadata loaded from the kernel's JSON file,
 * including shared memory requirements and target architecture.
 */
struct CudaKernelMetadata {
    unsigned int shared;  ///< Required shared memory in bytes
    unsigned int arch;    ///< Target CUDA architecture (e.g., 80 for sm_80)
};

/**
 * @brief CUDA Backend Policy
 *
 * Implements the BackendPolicy concept for CUDA devices.
 * Uses CUDA Driver API for low-level kernel management.
 *
 * Features:
 * - Lazy kernel loading with module caching
 * - Shared memory configuration
 * - Architecture compatibility checking
 * - Thread-safe module management
 */
struct CudaBackend {
    // ========== Type Definitions ==========

    using StreamType = CUstream;
    using ContextType = CUcontext;
    using KernelHandle = CUfunction;

    /**
     * @brief Module handle with metadata
     *
     * Stores both the CUDA module and associated metadata for a kernel.
     */
    struct ModuleData {
        CUmodule module;
        CudaKernelMetadata metadata;
    };

    // ========== Static Members ==========

    /// Cache for loaded modules (thread-safe)
    static inline std::unordered_map<std::string, ModuleData> module_cache_;
    static inline std::mutex cache_mutex_;

    // ========== Core Backend Methods ==========

    /**
     * @brief Launch a CUDA kernel
     *
     * @param stream CUDA stream to launch on
     * @param kernel Kernel function handle
     * @param grid_x, grid_y, grid_z Grid dimensions
     * @param block_x, block_y, block_z Block dimensions
     * @param args Kernel arguments (array of pointers)
     * @param shared_memory Shared memory size in bytes (optional)
     *
     * @throws std::runtime_error if kernel launch fails
     */
    static void launch_kernel(
        CUstream stream,
        CUfunction kernel,
        unsigned grid_x, unsigned grid_y, unsigned grid_z,
        unsigned block_x, unsigned block_y, unsigned block_z,
        void** args,
        unsigned int shared_memory = 0
    ) {
        LOG(INFO) << "cuLaunchKernel";

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
                fmt::format("CUDA kernel launch failed: {}", error_string)
            );
        }
    }

    /**
     * @brief Ensure CUDA context is properly initialized
     *
     * For PyTorch integration, the context is usually managed by PyTorch.
     * This function verifies the context exists and is valid.
     *
     * @throws std::runtime_error if context initialization fails
     */
    static void ensure_context() {
        // When using PyTorch, the CUDA context is typically already initialized
        // This function primarily serves as a validation step
        CUcontext ctx;
        CUresult result = cuCtxGetCurrent(&ctx);

        if (result != CUDA_SUCCESS || ctx == nullptr) {
            // If no context exists, we could create one, but this is unusual
            // in PyTorch workflows. Log a warning.
            LOG(WARNING) << "No CUDA context found. Creating default context.";

            CUdevice device;
            checkCudaErrors(cuDeviceGet(&device, 0));
            checkCudaErrors(cuCtxCreate(&ctx, 0, device));
        }
    }

    /**
     * @brief Get current CUDA device index
     *
     * @return Device index (0, 1, 2, ...)
     * @throws std::runtime_error if device query fails
     */
    static int get_device_index() {
        CUdevice device;
        CUresult result = cuCtxGetDevice(&device);

        if (result != CUDA_SUCCESS) {
            const char* error_string;
            cuGetErrorString(result, &error_string);
            throw std::runtime_error(
                fmt::format("Failed to get CUDA device: {}", error_string)
            );
        }

        return static_cast<int>(device);
    }

    /**
     * @brief Load a kernel from compiled binary
     *
     * This function:
     * 1. Loads kernel metadata from JSON
     * 2. Verifies architecture compatibility
     * 3. Loads CUBIN module
     * 4. Configures shared memory if needed
     * 5. Returns kernel function handle
     *
     * Results are cached for subsequent calls.
     *
     * @param dir Directory containing kernel files (.cubin, .json)
     * @param kernel_name Name of the kernel function
     * @return Kernel function handle
     * @throws std::runtime_error if loading fails or architecture mismatches
     */
    static CUfunction load_kernel(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check cache first
        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            // Return cached kernel
            CUfunction kernel;
            checkCudaErrors(cuModuleGetFunction(
                &kernel, it->second.module, kernel_name.c_str()));
            return kernel;
        }

        // Load metadata
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);
        if (!f.is_open()) {
            throw std::runtime_error(
                fmt::format("Failed to open metadata file: {}", metadata_path));
        }

        nlohmann::json meta_data = nlohmann::json::parse(f);
        CudaKernelMetadata metadata;
        metadata.shared = meta_data["shared"];
        metadata.arch = meta_data["target"]["arch"];

        LOG(INFO) << fmt::format(
            "Loading kernel {} with arch={}, shared={}",
            kernel_name, metadata.arch, metadata.shared);

        // Check architecture compatibility
        CUdevice device;
        checkCudaErrors(cuCtxGetDevice(&device));

        int major = 0, minor = 0;
        checkCudaErrors(cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        checkCudaErrors(cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

        unsigned int device_arch = major * 10 + minor;
        if (device_arch != metadata.arch) {
            throw std::runtime_error(fmt::format(
                "Compute architecture mismatch! Device has sm_{}, kernel requires sm_{}",
                device_arch, metadata.arch));
        }

        // Load module
        std::string cubin_path = fmt::format("{}/{}.cubin", dir, kernel_name);
        LOG(INFO) << fmt::format("Loading cubin from {}", cubin_path);

        CUmodule module;
        checkCudaErrors(cuModuleLoad(&module, cubin_path.c_str()));

        // Get function
        CUfunction kernel;
        checkCudaErrors(cuModuleGetFunction(&kernel, module, kernel_name.c_str()));

        // Configure shared memory if needed
        configure_shared_memory(kernel, device, metadata.shared);

        // Cache the module
        module_cache_[key] = ModuleData{module, metadata};

        return kernel;
    }

    /**
     * @brief Get shared memory size for a loaded kernel
     *
     * @param dir Directory containing kernel
     * @param kernel_name Kernel name
     * @return Shared memory size in bytes
     */
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
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);
        if (!f.is_open()) {
            return 0;
        }

        nlohmann::json meta_data = nlohmann::json::parse(f);
        return meta_data["shared"];
    }

private:
    /**
     * @brief Configure shared memory for a kernel
     *
     * Handles shared memory configuration including:
     * - Validation against device limits
     * - Cache configuration for large shared memory
     * - Dynamic shared memory attribute setting
     *
     * @param kernel Kernel function handle
     * @param device CUDA device
     * @param required_shared Required shared memory in bytes
     * @throws std::runtime_error if shared memory exceeds device limits
     */
    static void configure_shared_memory(
        CUfunction kernel,
        CUdevice device,
        unsigned int required_shared
    ) {
        // Check shared memory limits
        int shared_optin;
        cuDeviceGetAttribute(
            &shared_optin,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device);

        if (required_shared > shared_optin) {
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

// Compile-time verification that CudaBackend satisfies BackendPolicy concept
static_assert(BackendPolicy<CudaBackend>,
              "CudaBackend must satisfy BackendPolicy concept");

} // namespace triton_jit
