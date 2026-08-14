// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

  static size_t get_size(NpuArgType t) {
    switch (t) {
      case NpuArgType::POINTER:
        return sizeof(void*);
      case NpuArgType::I32:
        return sizeof(int32_t);
      case NpuArgType::I64:
        return sizeof(int64_t);
      case NpuArgType::F32:
        return sizeof(float);
      case NpuArgType::F64:
        return sizeof(double);
      default:
        return 8;
    }
  }

  static size_t get_align(NpuArgType t) {
    switch (t) {
      case NpuArgType::POINTER:
        return alignof(void*);
      case NpuArgType::I32:
        return alignof(int32_t);
      case NpuArgType::I64:
        return alignof(int64_t);
      case NpuArgType::F32:
        return alignof(float);
      case NpuArgType::F64:
        return alignof(double);
      default:
        return 8;
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

}  // namespace triton_jit
