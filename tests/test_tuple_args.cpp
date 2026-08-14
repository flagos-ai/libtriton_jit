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

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

#include "triton_jit/backends/npu_arg_buffer.h"
#include "triton_jit/triton_jit_function.h"

namespace {

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  using namespace triton_jit;

  static_assert(is_runtime_tuple_element_v<float>);
  static_assert(is_runtime_tuple_element_v<int32_t>);
  static_assert(!is_runtime_tuple_element_v<bool>);
  static_assert(!is_runtime_tuple_element_v<int16_t>);
  static_assert(!is_runtime_tuple_element_v<at::Tensor>);

  StaticSignature static_signature{1, {ArgType::NON_CONSTEXPR}};
  ParameterBuffer buffer;
  c10::SmallVector<std::string> signature;
  ArgHandle handler{static_signature, buffer, signature, 0};

  handler.handle_arg(std::tuple<float, int32_t>{3.5F, 7});

  bool ok = true;
  ok &= expect(handler.idx == 1, "tuple must consume one static-signature slot");
  ok &= expect(buffer.size() == 2, "tuple must push one ABI value per element");
  ok &= expect(signature.size() == 1, "tuple must emit one grouped signature token");
  if (signature.size() == 1) {
    ok &= expect(signature[0] == "(fp32,i32)", "grouped token must preserve element types");
  }

  auto pointers = buffer.get_ptrs();
  if (pointers.size() == 2) {
    ok &= expect(*reinterpret_cast<float*>(pointers[0]) == 3.5F,
                 "first ABI value must be preserved");
    ok &= expect(*reinterpret_cast<int32_t*>(pointers[1]) == 7,
                 "second ABI value must be preserved");
  }

  const auto layout = parse_signature("*fp32:16,(fp32,i32),64");
  ok &= expect(layout.size() == 3, "NPU layout must flatten the tuple into two ABI entries");
  if (layout.size() == 3) {
    ok &= expect(layout[0].type == NpuArgType::POINTER, "NPU layout entry 0 must be a pointer");
    ok &= expect(layout[1].type == NpuArgType::F32, "NPU layout entry 1 must be fp32");
    ok &= expect(layout[2].type == NpuArgType::I32, "NPU layout entry 2 must be i32");
  }

  StaticSignature constexpr_signature{1, {ArgType::CONSTEXPR}};
  ParameterBuffer constexpr_buffer;
  c10::SmallVector<std::string> constexpr_tokens;
  ArgHandle constexpr_handler{constexpr_signature, constexpr_buffer, constexpr_tokens, 0};
  bool constexpr_rejected = false;
  try {
    constexpr_handler.handle_arg(std::tuple<float, int32_t>{1.0F, 2});
  } catch (const std::exception&) {
    constexpr_rejected = true;
  }
  ok &= expect(constexpr_rejected, "runtime tuple must be rejected for a constexpr slot");
  ok &= expect(constexpr_buffer.size() == 0, "rejected constexpr tuple must not push ABI values");

  bool nested_signature_rejected = false;
  try {
    (void)parse_signature("((fp32,i32),i64)");
  } catch (const std::invalid_argument&) {
    nested_signature_rejected = true;
  }
  ok &= expect(nested_signature_rejected, "nested raw tuple signatures must be rejected");

  bool malformed_signature_rejected = false;
  try {
    (void)parse_signature("(fp32(i32)i64)");
  } catch (const std::invalid_argument&) {
    malformed_signature_rejected = true;
  }
  ok &= expect(malformed_signature_rejected,
               "tuple signatures with embedded parentheses must be rejected");

  return ok ? 0 : 1;
}
