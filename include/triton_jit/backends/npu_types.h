#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
    std::vector<NpuArgInfo> arg_layout;
    size_t workspace_size = 0;

    bool has_arg_layout() const {
        return !arg_layout.empty();
    }
};

} // namespace triton_jit
