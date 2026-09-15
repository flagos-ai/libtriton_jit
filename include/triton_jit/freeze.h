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

// Process-wide, nestable cold-work policy. Ready kernels continue to run.
// CUDA graph capture records GPU operations; it does not record Python calls.
// Resolving or compiling on a miss may synchronize or start nested capture,
// so those cold paths refuse before entering their Python callbacks.

#include <stdexcept>
#include <string>

namespace triton_jit {

enum class ColdWork { kConfig, kProgram, kFunction };
enum class ColdRestriction { kExplicitFreeze, kCapture };

class FrozenMissError : public std::runtime_error {
 public:
  explicit FrozenMissError(const std::string& what,
                           ColdWork work = ColdWork::kProgram,
                           ColdRestriction restriction = ColdRestriction::kExplicitFreeze)
      : std::runtime_error(what), work_(work), restriction_(restriction) {
  }
  ColdWork work() const noexcept {
    return work_;
  }
  ColdRestriction restriction() const noexcept {
    return restriction_;
  }

 private:
  ColdWork work_;
  ColdRestriction restriction_;
};

// True while at least one ScopedFreeze is alive anywhere in the process.
bool is_frozen() noexcept;

// True when `stream` (the backend's stream handle) is currently being
// captured into a graph. Backends without capture support return false.
bool stream_is_capturing(void* stream) noexcept;

// Nestable, process-wide. Any compile-cache miss (and any tuned-config
// resolve that would enter Python) throws FrozenMissError until the last
// guard is destroyed.
class ScopedFreeze {
 public:
  ScopedFreeze();
  ~ScopedFreeze();
  ScopedFreeze(const ScopedFreeze&) = delete;
  ScopedFreeze& operator=(const ScopedFreeze&) = delete;
};

namespace detail {
  // Throws FrozenMissError when the runtime is frozen or `stream` is capturing.
  // `what` describes the operation that would have run (kernel name and
  // signature); it is only evaluated on the slow path.
  void refuse_if_frozen(void* stream, const std::string& what, ColdWork work = ColdWork::kProgram);
}  // namespace detail

}  // namespace triton_jit
