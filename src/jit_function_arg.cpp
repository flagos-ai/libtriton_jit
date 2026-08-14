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

#include "triton_jit/jit_function_arg.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include "fmt/core.h"

namespace triton_jit {
namespace {

uint64_t fnv1a64(std::string_view bytes) {
  uint64_t hash = 14695981039346656037ULL;
  for (const unsigned char byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string hex_encode(std::string_view bytes) {
  static constexpr char kHexDigits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(bytes.size() * 2);
  for (const unsigned char byte : bytes) {
    encoded.push_back(kHexDigits[byte >> 4]);
    encoded.push_back(kHexDigits[byte & 0x0f]);
  }
  return encoded;
}

std::string hex_u64(uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << value;
  return output.str();
}

std::string read_module_source(const std::filesystem::path& path) {
  std::ifstream input{path, std::ios::binary};
  if (!input) {
    throw std::runtime_error(fmt::format("Failed to open JITFunction module: {}", path.string()));
  }
  return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

}  // namespace

JitFunctionArg::JitFunctionArg(std::string module_path, std::string function_name)
    : function_name_(std::move(function_name)) {
  if (module_path.empty()) {
    throw std::invalid_argument("JITFunction module path must not be empty");
  }
  if (function_name_.empty()) {
    throw std::invalid_argument("JITFunction name must not be empty");
  }

  std::error_code error;
  auto path = std::filesystem::absolute(std::filesystem::path{module_path}, error);
  if (error) {
    throw std::invalid_argument(
        fmt::format("Invalid JITFunction module path '{}': {}", module_path, error.message()));
  }
  path = path.lexically_normal();
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw std::invalid_argument(
        fmt::format("JITFunction module is not a regular file: {}", path.string()));
  }

  module_path_ = path.string();
  source_fingerprint_ = hex_u64(fnv1a64(read_module_source(path)));
}

const std::string& JitFunctionArg::module_path() const noexcept {
  return module_path_;
}

const std::string& JitFunctionArg::function_name() const noexcept {
  return function_name_;
}

const std::string& JitFunctionArg::source_fingerprint() const noexcept {
  return source_fingerprint_;
}

std::string JitFunctionArg::signature_token() const {
  return fmt::format("@jit:{}:{}:{}", hex_encode(module_path_), hex_encode(function_name_),
                     source_fingerprint_);
}

}  // namespace triton_jit
