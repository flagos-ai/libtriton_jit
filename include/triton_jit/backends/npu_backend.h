#pragma once

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cstring>
#include <sstream>
#include <cctype>

#include "acl/acl.h"
#include "experiment/runtime/runtime/rt.h"
#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"  // for checkAclErrors

namespace triton_jit {

enum class NpuArgType : uint8_t {
    POINTER = 0,
    I32 = 1,
    I64 = 2,
    F32 = 3,
    F64 = 4,
};

struct NpuArgInfo {
    NpuArgType type;
    size_t size;

    static size_t get_size(NpuArgType t) {
        switch (t) {
            case NpuArgType::POINTER: return sizeof(void*);
            case NpuArgType::I32:     return sizeof(int32_t);
            case NpuArgType::I64:     return sizeof(int64_t);
            case NpuArgType::F32:     return sizeof(float);
            case NpuArgType::F64:     return sizeof(double);
            default:                  return 8;
        }
    }

    static size_t get_align(NpuArgType t) {
        switch (t) {
            case NpuArgType::POINTER: return alignof(void*);
            case NpuArgType::I32:     return alignof(int32_t);
            case NpuArgType::I64:     return alignof(int64_t);
            case NpuArgType::F32:     return alignof(float);
            case NpuArgType::F64:     return alignof(double);
            default:                  return 8;
        }
    }
};

struct NpuKernelMetadata {
    unsigned int shared;
    std::string mix_mode;
    std::vector<NpuArgInfo> arg_layout;  // Dynamic argument layout

    bool has_arg_layout() const {
        return !arg_layout.empty();
    }
};

/**
 * @brief Dynamic argument buffer for NPU kernel launch
 *
 * NPU kernel arguments must be packed into a contiguous memory block:
 * [0-7]   ffts_addr (8B)      - System parameter
 * [8-15]  syncBlockLock (8B)  - System parameter
 * [16-23] workspace_addr (8B) - System parameter
 * [24...] User arguments      - Dynamic, based on kernel signature
 * [...]   gridX, gridY, gridZ - Grid dimensions (4B each)
 */
class NpuArgBuffer {
public:
    static constexpr size_t SYSTEM_ARGS_SIZE = 3 * sizeof(void*);  // 24 bytes
    static constexpr size_t USER_ARGS_OFFSET = SYSTEM_ARGS_SIZE;

    explicit NpuArgBuffer(size_t estimated_user_args = 64) {
        buffer_.resize(SYSTEM_ARGS_SIZE + estimated_user_args + 16);
        cursor_ = USER_ARGS_OFFSET;
    }

    void set_system_args(void* ffts, void* sync_lock, void* workspace) {
        std::memcpy(buffer_.data() + 0, &ffts, sizeof(void*));
        std::memcpy(buffer_.data() + 8, &sync_lock, sizeof(void*));
        std::memcpy(buffer_.data() + 16, &workspace, sizeof(void*));
    }

    template<typename T>
    void push_arg(const T& value) {
        size_t align = alignof(T);
        cursor_ = align_to(cursor_, align);

        ensure_capacity(cursor_ + sizeof(T));
        std::memcpy(buffer_.data() + cursor_, &value, sizeof(T));
        cursor_ += sizeof(T);
    }

    void push_arg_by_type(void* arg_ptr, NpuArgType type) {
        if (arg_ptr == nullptr) {
            LOG(WARNING) << "push_arg_by_type: arg_ptr is nullptr";
            return;
        }

        switch (type) {
            case NpuArgType::POINTER:
                push_arg(*reinterpret_cast<void**>(arg_ptr));
                break;
            case NpuArgType::I32:
                push_arg(*reinterpret_cast<int32_t*>(arg_ptr));
                break;
            case NpuArgType::I64:
                push_arg(*reinterpret_cast<int64_t*>(arg_ptr));
                break;
            case NpuArgType::F32:
                push_arg(*reinterpret_cast<float*>(arg_ptr));
                break;
            case NpuArgType::F64:
                push_arg(*reinterpret_cast<double*>(arg_ptr));
                break;
        }
    }

    void push_args_from_layout(void** args, const std::vector<NpuArgInfo>& layout) {
        for (size_t i = 0; i < layout.size(); ++i) {
            if (args[i] != nullptr) {
                push_arg_by_type(args[i], layout[i].type);
            }
        }
    }

    void set_grid(int32_t gx, int32_t gy, int32_t gz) {
        cursor_ = align_to(cursor_, alignof(int32_t));
        ensure_capacity(cursor_ + 3 * sizeof(int32_t));

        std::memcpy(buffer_.data() + cursor_, &gx, sizeof(int32_t));
        cursor_ += sizeof(int32_t);
        std::memcpy(buffer_.data() + cursor_, &gy, sizeof(int32_t));
        cursor_ += sizeof(int32_t);
        std::memcpy(buffer_.data() + cursor_, &gz, sizeof(int32_t));
        cursor_ += sizeof(int32_t);
    }

    void* data() { return buffer_.data(); }
    size_t size() const { return cursor_; }

private:
    static size_t align_to(size_t pos, size_t alignment) {
        return (pos + alignment - 1) & ~(alignment - 1);
    }

    void ensure_capacity(size_t required) {
        if (required > buffer_.size()) {
            buffer_.resize(required + 32);
        }
    }

    std::vector<std::byte> buffer_;
    size_t cursor_;
};

/**
 * @brief Parse signature string to extract argument types
 *
 * Signature format: "*fp32:16,*fp32,i64,1024,nullopt"
 * - "*..." indicates pointer type
 * - "i32", "i64", "u32", "u64" for integers
 * - "fp32", "fp64" for floats
 * - Pure numbers are constexpr (skipped)
 * - "nullopt" is skipped
 */
inline std::vector<NpuArgInfo> parse_signature(const std::string& sig) {
    std::vector<NpuArgInfo> layout;

    std::stringstream ss(sig);
    std::string token;

    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start, end - start + 1);

        // Skip empty tokens
        if (token.empty()) continue;

        // Skip "nullopt"
        if (token == "nullopt") continue;

        // Skip pure numbers (constexpr values like "1024", "128")
        bool is_number = !token.empty() && (std::isdigit(token[0]) ||
                         (token[0] == '-' && token.size() > 1 && std::isdigit(token[1])));
        if (is_number) continue;

        // Remove specialization suffix (:16, :1, etc.)
        size_t colon_pos = token.find(':');
        if (colon_pos != std::string::npos) {
            token = token.substr(0, colon_pos);
        }

        NpuArgInfo info;

        if (token[0] == '*') {
            // Pointer type: *fp32, *fp16, *i32, etc.
            info.type = NpuArgType::POINTER;
            info.size = sizeof(void*);
        } else if (token.substr(0, 3) == "i64" || token.substr(0, 3) == "u64") {
            info.type = NpuArgType::I64;
            info.size = sizeof(int64_t);
        } else if (token.substr(0, 3) == "i32" || token.substr(0, 3) == "u32") {
            info.type = NpuArgType::I32;
            info.size = sizeof(int32_t);
        } else if (token.substr(0, 4) == "fp64" || token.substr(0, 3) == "f64") {
            info.type = NpuArgType::F64;
            info.size = sizeof(double);
        } else if (token.substr(0, 4) == "fp32" || token.substr(0, 3) == "f32") {
            info.type = NpuArgType::F32;
            info.size = sizeof(float);
        } else if (token.substr(0, 4) == "fp16" || token.substr(0, 3) == "f16" ||
                   token.substr(0, 4) == "bf16") {
            // fp16/bf16 scalars are promoted to fp32 in most cases
            info.type = NpuArgType::F32;
            info.size = sizeof(float);
        } else {
            // Default to i64 for unknown types
            LOG(WARNING) << "Unknown type in signature: " << token << ", defaulting to i64";
            info.type = NpuArgType::I64;
            info.size = sizeof(int64_t);
        }

        layout.push_back(info);
    }

    return layout;
}

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
        const std::vector<NpuArgInfo>* arg_layout = nullptr
    ) {
        // Calculate block count
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
        if (arg_layout != nullptr && !arg_layout->empty()) {
            // Use provided layout from metadata
            layout = *arg_layout;
            LOG(INFO) << fmt::format("Using metadata arg_layout with {} args", layout.size());
        } else if (!signature.empty()) {
            // Parse from signature string
            layout = parse_signature(signature);
            LOG(INFO) << fmt::format("Parsed signature '{}' -> {} runtime args",
                                    signature, layout.size());
        } else {
            throw std::runtime_error("launch_kernel: no signature or arg_layout provided");
        }

        // Build argument buffer dynamically
        NpuArgBuffer arg_buffer(layout.size() * 8 + 16);

        // 1. Set system arguments
        arg_buffer.set_system_args(ffts_addr, nullptr, nullptr);

        // 2. Add user arguments based on layout
        if (args != nullptr) {
            arg_buffer.push_args_from_layout(args, layout);
        } else {
            LOG(WARNING) << "launch_kernel: args is nullptr!";
        }

        // 3. Add grid dimensions
        arg_buffer.set_grid(
            static_cast<int32_t>(grid_x),
            static_cast<int32_t>(grid_y),
            static_cast<int32_t>(grid_z)
        );

        LOG(INFO) << fmt::format(
            "NPU launch_kernel: blockNum={}, arg_buffer_size={}, grid=({},{},{})",
            blockNum, arg_buffer.size(), grid_x, grid_y, grid_z);

        // Launch kernel
        rtError_t rt_err = rtKernelLaunch(kernel,
                                          blockNum,
                                          arg_buffer.data(),
                                          static_cast<uint32_t>(arg_buffer.size()),
                                          nullptr,
                                          stream);

        if (rt_err != RT_ERROR_NONE) {
            throw std::runtime_error(fmt::format("rtKernelLaunch failed: {}",
                                                static_cast<int>(rt_err)));
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

        // Load metadata
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);

        NpuKernelMetadata metadata;
        metadata.shared = 0;
        metadata.mix_mode = "mix";

        if (f.is_open()) {
            nlohmann::json meta_data = nlohmann::json::parse(f);
            metadata.shared = meta_data.contains("shared") ? meta_data["shared"].get<int>() : 0;
            metadata.mix_mode = meta_data.contains("mix_mode") ?
                               meta_data["mix_mode"].get<std::string>() : "mix";

            // Parse arg_layout if present
            if (meta_data.contains("arg_layout") && meta_data["arg_layout"].is_array()) {
                for (const auto& arg : meta_data["arg_layout"]) {
                    if (!arg.contains("type")) continue;

                    std::string type_str = arg["type"].get<std::string>();
                    NpuArgInfo info;

                    // Skip constexpr arguments
                    if (type_str == "constexpr") continue;

                    if (type_str == "ptr" || type_str == "pointer") {
                        info.type = NpuArgType::POINTER;
                        info.size = sizeof(void*);
                    } else if (type_str == "i64" || type_str == "u64") {
                        info.type = NpuArgType::I64;
                        info.size = sizeof(int64_t);
                    } else if (type_str == "i32" || type_str == "u32") {
                        info.type = NpuArgType::I32;
                        info.size = sizeof(int32_t);
                    } else if (type_str == "fp64" || type_str == "f64") {
                        info.type = NpuArgType::F64;
                        info.size = sizeof(double);
                    } else if (type_str == "fp32" || type_str == "f32") {
                        info.type = NpuArgType::F32;
                        info.size = sizeof(float);
                    } else {
                        // Default to i64 for unknown types
                        LOG(WARNING) << "Unknown arg type in metadata: " << type_str;
                        info.type = NpuArgType::I64;
                        info.size = sizeof(int64_t);
                    }

                    metadata.arg_layout.push_back(info);
                }
                LOG(INFO) << fmt::format("Loaded arg_layout from JSON with {} args",
                                        metadata.arg_layout.size());
            }
        }

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

        // Set magic value based on mix_mode
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

        // If not in cache, load metadata
        std::string metadata_path = fmt::format("{}/{}.json", dir, kernel_name);
        std::ifstream f(metadata_path);
        if (!f.is_open()) {
            return 0;
        }

        nlohmann::json meta_data = nlohmann::json::parse(f);
        return meta_data.contains("shared") ? meta_data["shared"].get<unsigned int>() : 0;
    }

    /**
     * @brief Get kernel metadata including arg_layout
     *
     * @param dir Directory containing kernel files
     * @param kernel_name Name of the kernel
     * @return Pointer to cached metadata, or nullptr if not found
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
