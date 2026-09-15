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

// Host-side tests for freeze.h: guard nesting, the refusal helper, and the
// no-capture answer for a null stream. The device behaviour (a compile miss
// during graph capture throws before anything is recorded) is exercised by the
// operator tests that run on a device.

#include "triton_jit/freeze.h"

#include <iostream>
#include <string>
#include <thread>

namespace {

int failures = 0;

#define CHECK(cond)                                                                        \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      ++failures;                                                                          \
      std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << std::endl; \
    }                                                                                      \
  } while (0)

bool refused(void* stream) {
  try {
    triton_jit::detail::refuse_if_frozen(stream, "kernel.py:k [i32]");
    return false;
  } catch (const triton_jit::FrozenMissError& error) {
    return std::string(error.what()).find("kernel.py:k [i32]") != std::string::npos;
  }
}

}  // namespace

int main() {
  CHECK(!triton_jit::is_frozen());
  CHECK(!triton_jit::stream_is_capturing(nullptr));
  CHECK(!refused(nullptr));
  {
    triton_jit::ScopedFreeze outer;
    CHECK(triton_jit::is_frozen());
    CHECK(refused(nullptr));
    {
      triton_jit::ScopedFreeze inner;
      CHECK(triton_jit::is_frozen());
    }
    CHECK(triton_jit::is_frozen());  // still held by the outer guard
    // process-wide: another thread sees the freeze
    bool seen_in_thread = false;
    std::thread([&] { seen_in_thread = triton_jit::is_frozen() && refused(nullptr); }).join();
    CHECK(seen_in_thread);
  }
  CHECK(!triton_jit::is_frozen());
  CHECK(!refused(nullptr));
  // FrozenMissError is a runtime_error so existing catch sites keep working
  try {
    triton_jit::ScopedFreeze guard;
    triton_jit::detail::refuse_if_frozen(nullptr, "x");
    CHECK(false);
  } catch (const std::runtime_error&) {
  }
  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "freeze: all checks passed" << std::endl;
  return 0;
}
