#pragma once

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "runtime/runtime/rt.h"
#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"
#include "triton_jit/backends/npu_types.h"
#include "triton_jit/backends/npu_arg_buffer.h"

namespace triton_jit {

struct NpuBackend {
    using StreamType = aclrtStream;
    using ContextType = aclrtContext;
    using KernelHandle = void*;

    // NPU does not use warp concept, but we need a non-zero value for block size calculation
    static constexpr unsigned int WARP_SIZE = 1;

    struct ModuleData {
        void* bin_handle;
        void* fn_handle;
        NpuKernelMetadata metadata;
    };

    static inline std::unordered_map<std::string, ModuleData> module_cache_;
    static inline std::mutex cache_mutex_;
    // Static storage for function stubs
    static inline std::unordered_map<std::string, size_t> registered_names_;
    static inline std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs_;

    static void launch_kernel(
        aclrtStream stream,
        void* kernel,
        unsigned grid_x, unsigned grid_y, unsigned grid_z,
        unsigned block_x, unsigned block_y, unsigned block_z,
        void** args,
        unsigned int shared_memory = 0,
        const std::string& signature = "",
        const std::vector<NpuArgInfo>* arg_layout = nullptr,
        size_t num_args = 0,
        size_t workspace_size = 0
    ) {
        uint32_t blockNum = grid_x * grid_y * grid_z;

        // Get system control address
        rtError_t ret;
        void* ffts_addr = nullptr;
        uint32_t ffts_len;
        ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
        if (ret != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtGetC2cCtrlAddr failed: {}",
                                                static_cast<int>(ret)));
        }

        // Determine argument layout
        std::vector<NpuArgInfo> layout;
        if (!signature.empty()) {
            layout = parse_signature(signature);
            LOG(INFO) << fmt::format("Parsed signature '{}' -> {} runtime args",
                                    signature, layout.size());
        } else if (arg_layout != nullptr && !arg_layout->empty()) {
            layout = *arg_layout;
            LOG(INFO) << fmt::format("Using metadata arg_layout with {} args", layout.size());
        } else {
            throw std::runtime_error("launch_kernel: no signature or arg_layout provided");
        }

        if (num_args != 0 && num_args != layout.size()) {
            throw std::runtime_error(fmt::format(
                "launch_kernel: arg count mismatch (layout={}, args={})",
                layout.size(), num_args));
        }

        // Build argument buffer dynamically
        NpuArgBuffer arg_buffer(layout.size() * 8 + 16);

        // 1. Allocate workspace if needed
        void* workspace_addr = nullptr;
        if (workspace_size > 0) {
            size_t total_workspace = workspace_size * blockNum;
            rtError_t ws_ret = rtMalloc(&workspace_addr, total_workspace, RT_MEMORY_HBM, 0);
            if (ws_ret != RT_ERROR_NONE) {
                throw std::runtime_error(fmt::format(
                    "rtMalloc workspace failed: {}, requested {} bytes",
                    static_cast<int>(ws_ret), total_workspace));
            }
            LOG(INFO) << fmt::format(
                "NPU workspace allocated: {} bytes ({} per block x {} blocks)",
                total_workspace, workspace_size, blockNum);
        }

        // 2. Set system arguments (ffts, sync_lock, workspace)
        arg_buffer.set_system_args(ffts_addr, nullptr, workspace_addr);

        // 3. Add user arguments based on layout
        if (args != nullptr) {
            LOG(INFO) << fmt::format("NPU args debug: num_args={}", layout.size());
            for (size_t i = 0; i < std::min(layout.size(), size_t(6)); ++i) {
                if (args[i] != nullptr) {
                    if (layout[i].type == NpuArgType::POINTER) {
                        void* ptr_val = *reinterpret_cast<void**>(args[i]);
                        LOG(INFO) << fmt::format("  arg[{}]: POINTER = {}", i, ptr_val);
                    } else if (layout[i].type == NpuArgType::I64) {
                        int64_t val = *reinterpret_cast<int64_t*>(args[i]);
                        LOG(INFO) << fmt::format("  arg[{}]: I64 = {}", i, val);
                    }
                }
            }
            arg_buffer.push_args_from_layout(args, layout);
        } else {
            LOG(WARNING) << "launch_kernel: args is nullptr!";
        }

        // 4. Add grid dimensions
        arg_buffer.set_grid(
            static_cast<int32_t>(grid_x),
            static_cast<int32_t>(grid_y),
            static_cast<int32_t>(grid_z)
        );

        LOG(INFO) << fmt::format(
            "NPU launch_kernel: blockNum={}, arg_buffer_size={}, grid=({},{},{}), workspace={}",
            blockNum, arg_buffer.size(), grid_x, grid_y, grid_z,
            workspace_addr ? fmt::format("{}B", workspace_size * blockNum) : "none");

        // Launch kernel
        rtError_t rt_err = rtKernelLaunch(kernel,
                                          blockNum,
                                          arg_buffer.data(),
                                          static_cast<uint32_t>(arg_buffer.size()),
                                          nullptr,
                                          stream);

        if (rt_err != RT_ERROR_NONE) {
            if (workspace_addr) {
                rtFree(workspace_addr);
            }
            throw std::runtime_error(fmt::format("rtKernelLaunch failed: {}",
                                                static_cast<int>(rt_err)));
        }

        // Free workspace after kernel completes via stream callback
        if (workspace_addr) {
            aclError cb_err = aclrtLaunchCallback(
                [](void* userData) { rtFree(userData); },
                workspace_addr,
                ACL_CALLBACK_BLOCK,
                stream
            );
            if (cb_err != ACL_ERROR_NONE) {
                LOG(WARNING) << fmt::format(
                    "Failed to register workspace cleanup callback (err={}), "
                    "workspace may leak", static_cast<int>(cb_err));
            }
        }
    }

    static void ensure_context() {
        aclrtContext ctx;
        aclError ret = aclrtGetCurrentContext(&ctx);

        if (ret != ACL_ERROR_NONE || ctx == nullptr) {
            LOG(WARNING) << "No ACL context found. Creating default context.";
            int deviceId = 0;
            aclError err = aclrtSetDevice(deviceId);
            if (err != ACL_ERROR_NONE) {
                throw std::runtime_error(fmt::format("aclrtSetDevice failed: {}",
                                                    static_cast<int>(err)));
            }
            err = aclrtCreateContext(&ctx, deviceId);
            if (err != ACL_ERROR_NONE) {
                throw std::runtime_error(fmt::format("aclrtCreateContext failed: {}",
                                                    static_cast<int>(err)));
            }
            err = aclrtSetCurrentContext(ctx);
            if (err != ACL_ERROR_NONE) {
                throw std::runtime_error(fmt::format("aclrtSetCurrentContext failed: {}",
                                                    static_cast<int>(err)));
            }
        }
    }

    static int get_device_index() {
        int device_id = -1;
        aclError err = aclrtGetDevice(&device_id);

        if (err != ACL_ERROR_NONE) {
            throw std::runtime_error(fmt::format("Failed to get NPU device: {}",
                                                static_cast<int>(err)));
        }

        return device_id;
    }

    static void* load_kernel(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);

        std::lock_guard<std::mutex> lock(cache_mutex_);

        // Check cache first
        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            return it->second.fn_handle;
        }

        // Load metadata via centralized loader (no JSON dependency in header)
        NpuKernelMetadata metadata = load_npu_metadata(dir, kernel_name);

        LOG(INFO) << fmt::format(
            "Loading NPU kernel {} with mix_mode={}, shared={}",
            kernel_name, metadata.mix_mode, metadata.shared);

        // Find kernel binary file (try .npubin, .o, .ttadapter, .bin)
        std::string rt_bin_path = fmt::format("{}/{}.npubin", dir, kernel_name);
        std::ifstream bin_file(rt_bin_path, std::ios::binary | std::ios::ate);

        if (!bin_file.good()) {
            std::vector<std::string> fallback_exts = {".o", ".ttadapter", ".bin"};
            bool file_found = false;

            for (const auto& ext : fallback_exts) {
                rt_bin_path = fmt::format("{}/{}{}", dir, kernel_name, ext);
                bin_file.open(rt_bin_path, std::ios::binary | std::ios::ate);
                if (bin_file.good()) {
                    file_found = true;
                    break;
                }
                bin_file.close();
                bin_file.clear();
            }

            if (!file_found) {
                throw std::runtime_error(fmt::format("Kernel binary not found: {}/{}",
                                                    dir, kernel_name));
            }
        }

        // Read binary file
        std::streamsize size = bin_file.tellg();
        if (size <= 0) {
            throw std::runtime_error(fmt::format("Invalid binary size: {}", rt_bin_path));
        }

        bin_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));

        if (!bin_file.read(buffer.data(), size)) {
            throw std::runtime_error(fmt::format("Failed to read binary: {}", rt_bin_path));
        }
        bin_file.close();

        LOG(INFO) << fmt::format("Loading NPU binary from {}, size={}", rt_bin_path, size);

        // Get current device ID
        int device_id = -1;
        aclError err = aclrtGetDevice(&device_id);
        if (err != ACL_SUCCESS) {
            device_id = 0;  // fallback
        }

        // Set device
        rtError_t rt_err = rtSetDevice(device_id);
        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtSetDevice failed for device {}, error: {}",
                                                device_id, static_cast<int>(rt_err)));
        }

        // Register binary with RT API
        rtDevBinary_t binary;
        binary.data = buffer.data();
        binary.length = static_cast<uint32_t>(size);

        binary.magic = (metadata.mix_mode == "aiv") ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
        binary.version = 0;

        void* rt_bin_handle = nullptr;
        rt_err = rtDevBinaryRegister(&binary, &rt_bin_handle);
        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtDevBinaryRegister failed: {}",
                                                static_cast<int>(rt_err)));
        }

        // Create function stub with unique name
        std::string stubName = kernel_name;
        stubName += "_" + std::to_string(registered_names_[kernel_name]);
        registered_names_[kernel_name]++;

        auto registered = func_stubs_.emplace(stubName, std::make_unique<size_t>(0));
        void* func_stub_handle = registered.first->second.get();

        // Register function
        rt_err = rtFunctionRegister(rt_bin_handle,
                                   func_stub_handle,
                                   stubName.c_str(),
                                   (void*)kernel_name.c_str(),
                                   0);
        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtFunctionRegister failed: {}",
                                                static_cast<int>(rt_err)));
        }

        // Cache the module
        module_cache_[key] = ModuleData{rt_bin_handle, func_stub_handle, metadata};

        return func_stub_handle;
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

        return load_shared_memory(dir, kernel_name);
    }

    /**
     * @brief Get kernel metadata including arg_layout
     */
    static const NpuKernelMetadata* get_kernel_metadata(
        const std::string& dir,
        const std::string& kernel_name
    ) {
        std::string key = fmt::format("{}::{}", dir, kernel_name);
        std::lock_guard<std::mutex> lock(cache_mutex_);

        auto it = module_cache_.find(key);
        if (it != module_cache_.end()) {
            return &(it->second.metadata);
        }

        return nullptr;
    }
};

static_assert(BackendPolicy<NpuBackend>, "NpuBackend must satisfy BackendPolicy concept");

} // namespace triton_jit
