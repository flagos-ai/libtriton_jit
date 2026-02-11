#pragma once

#include <cctype>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "c10/util/Logging.h"
#include "triton_jit/backends/npu_types.h"

namespace triton_jit {

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
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start, end - start + 1);

        if (token.empty()) continue;
        if (token == "nullopt") continue;

        bool is_number = !token.empty() && (std::isdigit(token[0]) ||
                         (token[0] == '-' && token.size() > 1 && std::isdigit(token[1])));
        if (is_number) continue;

        size_t colon_pos = token.find(':');
        if (colon_pos != std::string::npos) {
            token = token.substr(0, colon_pos);
        }

        NpuArgInfo info;

        if (token[0] == '*') {
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
            info.type = NpuArgType::F32;
            info.size = sizeof(float);
        } else {
            LOG(WARNING) << "Unknown type in signature: " << token << ", defaulting to i64";
            info.type = NpuArgType::I64;
            info.size = sizeof(int64_t);
        }

        layout.push_back(info);
    }

    return layout;
}

} // namespace triton_jit
