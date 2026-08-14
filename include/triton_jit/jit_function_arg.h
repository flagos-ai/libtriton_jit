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

#include <string>

namespace triton_jit {

// A compile-time reference to a Triton JITFunction defined in a Python module.
//
// Construction snapshots the module path and exact source contents. Recreate
// this object after intentionally changing the source file.
class JitFunctionArg {
 public:
  JitFunctionArg(std::string module_path, std::string function_name);

  const std::string& module_path() const noexcept;
  const std::string& function_name() const noexcept;
  const std::string& source_fingerprint() const noexcept;
  std::string signature_token() const;

 private:
  std::string module_path_;
  std::string function_name_;
  std::string source_fingerprint_;
};

}  // namespace triton_jit
