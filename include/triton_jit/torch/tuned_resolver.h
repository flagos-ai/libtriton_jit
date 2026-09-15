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

// Optional Python tuner bridge (LibTuner and plain @triton.autotune). Link
// TritonJIT::triton_jit_torch_resolver. Pass ResolveArgs as the context; see
// docs/autotune.md for setup.

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "triton_jit/tuned_config.h"

namespace triton_jit {
namespace torch_resolver {

  // A triton.language dtype passed as a constexpr argument (DTYPE=tl.float32).
  // `name` is spelled as in triton.language: "float32", "bfloat16", "int8";
  // a leading "tl." is accepted. The Python side materialises the object.
  struct TritonDtype {
    std::string name;
  };

  // One launch argument as the Python kernel would see it.
  using ArgValue = std::variant<at::Tensor, int64_t, double, bool, std::string, TritonDtype>;

  // What resolve() passes through as `context`. The bridge also hands Python
  // the device index of the resolve() call and `launch_stream`.
  struct ResolveArgs {
    std::vector<ArgValue> args;                            // positional arguments
    std::vector<std::pair<std::string, ArgValue>> kwargs;  // extra constexprs / meta by name
    // The kernel file this launch comes from (what get_instance() was given).
    // Kernels of the same name exist in several files (per-arch overrides);
    // this picks the right tuner.
    std::string source_path;
    // Optional portable, versioned source identity. Must match the exported
    // cache_namespace. Empty uses source_path and assumes immutable source.
    std::string cache_namespace;
    // Filled from resolve(..., stream) by the built-in adapter.
    void* launch_stream = nullptr;
    // Needed only when the key has never been tuned and Python must benchmark
    // now: the launch grid as a Python expression over the kernel's argument
    // names and META (the candidate's kwargs), e.g.
    //   "(triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)"
    // Without it an untuned key is reported as unsupported (a miss), never
    // benchmarked. Trial launches run on clones of the tensor arguments.
    std::string grid;
  };

  struct Options {
    std::string package;                    // "flag_gems", "flag_blas", or the kernels' module
    std::string module = "tuned_resolver";  // Python module providing resolve()
    std::string extra_sys_path;             // prepended to sys.path (tests)
  };

  // Build a TunedTable::Resolver. The returned callable expects `context` to
  // point at a ResolveArgs; a null context is a miss.
  TunedTable::Resolver make_resolver(Options options);

}  // namespace torch_resolver
}  // namespace triton_jit
