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

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include "triton_jit/backends/npu_arg_buffer.h"
#include "triton_jit/jit_function_arg.h"
#include "triton_jit/triton_jit_function.h"

namespace {

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

bool is_lower_hex(std::string_view value) {
  return !value.empty() &&
         std::all_of(value.begin(), value.end(),
                     [](char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); });
}

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    const auto base = std::filesystem::temp_directory_path();
    std::random_device random;
    for (int attempt = 0; attempt < 128; ++attempt) {
      auto candidate =
          base / ("libtriton_jit_function_arg_test-" + std::to_string(random()) + "-" +
                  std::to_string(random()));
      std::error_code error;
      if (std::filesystem::create_directory(candidate, error)) {
        path_ = std::move(candidate);
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::filesystem::filesystem_error(
            "Failed to create temporary test directory", candidate, error);
      }
    }
    throw std::runtime_error("Failed to create a unique temporary test directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

template <typename Exception, typename Fn>
bool expect_throws(Fn&& fn, const char* message) {
  try {
    fn();
  } catch (const Exception&) {
    return true;
  } catch (...) {
  }
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

}  // namespace

int main() {
  using namespace triton_jit;

  const auto fixture =
      std::filesystem::path{TRITON_JIT_TEST_SOURCE_DIR} / "fixtures/jit_function_module.py";
  JitFunctionArg function{fixture.string(), "mul_func"};
  JitFunctionArg same_function{fixture.string(), "mul_func"};

  StaticSignature static_signature{1, {ArgType::NON_CONSTEXPR}};
  ParameterBuffer buffer;
  c10::SmallVector<std::string> signature;
  ArgHandle handler{static_signature, buffer, signature, 0};
  handler.handle_arg(function);

  bool ok = true;
  ok &= expect(handler.idx == 1, "JITFunction must consume one static-signature slot");
  ok &= expect(buffer.size() == 0, "JITFunction must not push an ABI value");
  ok &= expect(signature.size() == 1, "JITFunction must emit one signature token");
  if (signature.size() == 1) {
    const auto& token = signature[0];
    ok &= expect(token.starts_with("@jit:"), "JITFunction token must use @jit prefix");
    ok &= expect(token.find(',') == std::string::npos,
                 "JITFunction token must be comma-safe");
    const auto path_end = token.find(':', 5);
    const auto name_end =
        path_end == std::string::npos ? std::string::npos : token.find(':', path_end + 1);
    ok &= expect(path_end != std::string::npos && name_end != std::string::npos &&
                     token.find(':', name_end + 1) == std::string::npos,
                 "JITFunction token must have exactly four colon-separated fields");
    if (path_end != std::string::npos && name_end != std::string::npos) {
      ok &= expect(is_lower_hex(std::string_view{token}.substr(5, path_end - 5)),
                   "module path must be lowercase hex");
      ok &= expect(is_lower_hex(
                       std::string_view{token}.substr(path_end + 1, name_end - path_end - 1)),
                   "function name must be lowercase hex");
      const auto fingerprint = std::string_view{token}.substr(name_end + 1);
      ok &= expect(fingerprint.size() == 16 && is_lower_hex(fingerprint),
                   "source fingerprint must be 16 lowercase hex characters");
    }
  }
  ok &= expect(function.signature_token() == same_function.signature_token(),
               "unchanged source must produce a deterministic token");
  const auto npu_layout = parse_signature(function.signature_token() + ",*fp32");
  ok &= expect(npu_layout.size() == 1,
               "NPU layout must not allocate an ABI entry for JITFunction");
  if (npu_layout.size() == 1) {
    ok &= expect(npu_layout[0].type == NpuArgType::POINTER,
                 "NPU layout must preserve the following runtime argument");
  }

  TemporaryDirectory temporary_directory;
  const auto temporary = temporary_directory.path() / "module.py";
  {
    std::ofstream output{temporary, std::ios::binary | std::ios::trunc};
    output << "value = 1\n";
  }
  JitFunctionArg first_snapshot{temporary.string(), "mul_func"};
  ok &= expect(first_snapshot.source_fingerprint() == "e37bb3a25227347c",
               "FNV-1a fingerprint must match the known value");
  {
    std::ofstream output{temporary, std::ios::binary | std::ios::trunc};
    output << "value = 2\n";
  }
  JitFunctionArg second_snapshot{temporary.string(), "mul_func"};
  ok &= expect(first_snapshot.signature_token() != second_snapshot.signature_token(),
               "changed source must produce a different token");

  ok &= expect_throws<std::invalid_argument>(
      [&] { (void)JitFunctionArg{fixture.string(), ""}; },
      "an empty function name must be rejected");
  ok &= expect_throws<std::invalid_argument>(
      [&] { (void)JitFunctionArg{"/definitely/missing/jit_function.py", "mul_func"}; },
      "a missing module must be rejected");

  return ok ? 0 : 1;
}
