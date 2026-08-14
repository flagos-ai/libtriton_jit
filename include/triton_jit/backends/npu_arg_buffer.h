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

#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
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

  template <typename T>
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

  void* data() {
    return buffer_.data();
  }
  size_t size() const {
    return cursor_;
  }

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
 * - "true"/"false" are boolean constexpr (skipped)
 */
inline std::string trim_signature_token(std::string_view token) {
  const size_t start = token.find_first_not_of(" \t");
  if (start == std::string_view::npos) {
    return "";
  }
  const size_t end = token.find_last_not_of(" \t");
  return std::string(token.substr(start, end - start + 1));
}

inline std::vector<std::string> split_signature_tokens(std::string_view signature) {
  std::vector<std::string> tokens;
  size_t token_start = 0;
  int depth = 0;

  for (size_t i = 0; i < signature.size(); ++i) {
    const char ch = signature[i];
    if (ch == '(') {
      ++depth;
    } else if (ch == ')') {
      if (depth == 0) {
        throw std::invalid_argument("Unmatched ')' in signature");
      }
      --depth;
    } else if (ch == ',' && depth == 0) {
      std::string token = trim_signature_token(signature.substr(token_start, i - token_start));
      if (token.empty()) {
        throw std::invalid_argument("Empty token in signature");
      }
      tokens.push_back(std::move(token));
      token_start = i + 1;
    }
  }

  if (depth != 0) {
    throw std::invalid_argument("Unmatched '(' in signature");
  }

  std::string final_token = trim_signature_token(signature.substr(token_start));
  if (!final_token.empty()) {
    tokens.push_back(std::move(final_token));
  } else if (!tokens.empty()) {
    throw std::invalid_argument("Empty token in signature");
  }
  return tokens;
}

inline void append_signature_token(std::string token, std::vector<NpuArgInfo>& layout) {
  if (token.starts_with('(') && token.ends_with(')')) {
    std::string_view inner(token.data() + 1, token.size() - 2);
    if (trim_signature_token(inner).empty()) {
      throw std::invalid_argument("Runtime tuple signature must not be empty");
    }
    for (std::string element : split_signature_tokens(inner)) {
      if (element.find_first_of("()") != std::string::npos) {
        throw std::invalid_argument("Nested runtime tuple signatures are not supported");
      }
      append_signature_token(std::move(element), layout);
    }
    return;
  }

  if (token.starts_with("@jit:")) {
    return;
  }

  if (token == "nullopt" || token == "true" || token == "false") {
    return;
  }

  const bool is_number =
      std::isdigit(static_cast<unsigned char>(token[0])) ||
      (token[0] == '-' && token.size() > 1 &&
       std::isdigit(static_cast<unsigned char>(token[1])));
  if (is_number) {
    return;
  }

  const size_t colon_pos = token.find(':');
  if (colon_pos != std::string::npos) {
    token = token.substr(0, colon_pos);
  }

  NpuArgInfo info;
  if (token[0] == '*') {
    info.type = NpuArgType::POINTER;
  } else if (token.substr(0, 3) == "i64" || token.substr(0, 3) == "u64") {
    info.type = NpuArgType::I64;
  } else if (token.substr(0, 3) == "i32" || token.substr(0, 3) == "u32") {
    info.type = NpuArgType::I32;
  } else if (token.substr(0, 4) == "fp64" || token.substr(0, 3) == "f64") {
    info.type = NpuArgType::F64;
  } else if (token.substr(0, 4) == "fp32" || token.substr(0, 3) == "f32") {
    info.type = NpuArgType::F32;
  } else if (token.substr(0, 4) == "fp16" || token.substr(0, 3) == "f16" ||
             token.substr(0, 4) == "bf16") {
    info.type = NpuArgType::F32;
  } else {
    LOG(WARNING) << "Unknown type in signature: " << token << ", defaulting to i64";
    info.type = NpuArgType::I64;
  }
  layout.push_back(info);
}

inline std::vector<NpuArgInfo> parse_signature(const std::string& sig) {
  std::vector<NpuArgInfo> layout;
  for (std::string token : split_signature_tokens(sig)) {
    append_signature_token(std::move(token), layout);
  }
  return layout;
}

}  // namespace triton_jit
