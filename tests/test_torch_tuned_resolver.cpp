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

// Tests the torch-side resolver bridge against a stand-in Python module
// (tests/fixtures/fake_tuned_resolver.py): argument conversion in both
// directions (including TritonDtype markers and the device index), the
// unsupported answer, and a Python exception. CPU tensors only; no device
// needed.

#include "triton_jit/torch/tuned_resolver.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>
#include <string>

#include "triton_jit/freeze.h"

namespace {

int failures = 0;

#define EXPECT(cond)                                                                       \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      ++failures;                                                                          \
      std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << std::endl; \
    }                                                                                      \
  } while (0)

using triton_jit::KeyStrategy;
using triton_jit::TunedTable;
using triton_jit::TuneKeyView;
namespace torch_resolver = triton_jit::torch_resolver;

}  // namespace

int main() {
  auto& table = TunedTable::instance();
  table.clear();
  torch_resolver::Options options;
  options.package = "fake_package";
  options.module = "fake_tuned_resolver";
  options.extra_sys_path = std::string(TRITON_JIT_TEST_SOURCE_DIR) + "/fixtures";
  table.set_resolver(torch_resolver::make_resolver(options));

  const at::Tensor a = torch::ones({4, 8}, torch::kFloat16);
  const at::Tensor b = torch::ones({8, 2}, torch::kBFloat16);
  torch_resolver::ResolveArgs ctx;
  ctx.args = {a, b, int64_t {4}, double {0.5}, true, std::string("fast")};
  ctx.kwargs = {
      {"SPLIT_K",                             int64_t {2}},
      {  "DTYPE", torch_resolver::TritonDtype {"float32"}}
  };
  ctx.source_path = "/tmp/fixtures/mm.py";
  ctx.grid = "(triton.cdiv(M, META['BLOCK_M']),)";

  const int64_t dims[] = {1000, 4000};
  const char* dtypes[] = {"torch.float16", "torch.bfloat16"};
  // the fake module echoes what it received into the config it returns
  const auto* cfg = table.resolve("mm_kernel", 0, TuneKeyView {dims, 2, dtypes, 2}, &ctx);
  EXPECT(cfg != nullptr);
  if (cfg != nullptr) {
    EXPECT(cfg->num_warps == 8 && cfg->num_stages == 2);
    EXPECT(cfg->get_i64("BLOCK_M", -1) == 64);
    EXPECT(cfg->get_bool("EVEN_K", false) == true);
    EXPECT(cfg->get_f64("SCALE", 0.0) == 0.25);
    EXPECT(cfg->get_str("MODE") != nullptr && *cfg->get_str("MODE") == "fast");
    EXPECT(cfg->extra.at("maxnreg") == "255");
    // echoed argument facts: tensor dtypes/shape, scalar types, kwargs
    EXPECT(cfg->get_str("ARG0") != nullptr && *cfg->get_str("ARG0") == "tensor:torch.float16:[4, 8]");
    EXPECT(cfg->get_str("ARG1") != nullptr && *cfg->get_str("ARG1") == "tensor:torch.bfloat16:[8, 2]");
    EXPECT(cfg->get_str("ARG2") != nullptr && *cfg->get_str("ARG2") == "int:4");
    EXPECT(cfg->get_str("ARG3") != nullptr && *cfg->get_str("ARG3") == "float:0.5");
    EXPECT(cfg->get_str("ARG4") != nullptr && *cfg->get_str("ARG4") == "bool:True");
    EXPECT(cfg->get_str("ARG5") != nullptr && *cfg->get_str("ARG5") == "str:fast");
    // a TritonDtype reaches Python as the resolver module's marker, everything else as itself
    EXPECT(cfg->get_str("KW") != nullptr && *cfg->get_str("KW") == "SPLIT_K=2,DTYPE=dtype:float32");
    // the device the entry is resolved for is handed over as well
    EXPECT(cfg->get_str("DEVICE") != nullptr && *cfg->get_str("DEVICE") == "0");
    EXPECT(cfg->get_str("PACKAGE") != nullptr && *cfg->get_str("PACKAGE") == "fake_package");
    EXPECT(cfg->get_str("SRC") != nullptr && *cfg->get_str("SRC") == "/tmp/fixtures/mm.py");
    EXPECT(cfg->get_str("GRID") != nullptr && *cfg->get_str("GRID") == "(triton.cdiv(M, META['BLOCK_M']),)");
  }
  // stored under the normalised key with the strategies Python reported (log, log)
  const auto* handle = table.kernel(triton_jit::scoped_kernel_id("mm_kernel", ctx.source_path), 0);
  EXPECT(handle != nullptr && handle->info().key_columns.size() == 2 &&
         handle->info().key_columns[0].second == KeyStrategy::kLog);
  const int64_t same_bucket[] = {1024, 4096};
  EXPECT(table.find_for_context("mm_kernel", 0, TuneKeyView {same_bucket, 2, dtypes, 2}, &ctx) != nullptr);
  // a second resolve for the same key does not go back to Python (the fake counts calls)
  const auto* again = table.resolve("mm_kernel", 0, TuneKeyView {dims, 2, dtypes, 2}, &ctx);
  EXPECT(again != nullptr && again->get_i64("CALLS", -1) == 1);
  // unsupported -> miss, not an error
  EXPECT(table.resolve("heuristic_kernel", 0, TuneKeyView {dims, 2, dtypes, 2}, &ctx) == nullptr);
  EXPECT(table.kernel("heuristic_kernel", 0) == nullptr);
  // a Python exception surfaces as std::runtime_error carrying the Python message
  bool threw = false;
  try {
    table.resolve("boom_kernel", 0, TuneKeyView {dims, 2, dtypes, 2}, &ctx);
  } catch (const std::runtime_error& error) {
    threw = std::string(error.what()).find("deliberate failure") != std::string::npos;
  }
  EXPECT(threw);
  // null context is a miss
  EXPECT(table.resolve("mm_kernel", 1, TuneKeyView {dims, 2, dtypes, 2}, nullptr) == nullptr);
  // frozen: refused before Python is entered
  {
    triton_jit::ScopedFreeze guard;
    bool refused = false;
    try {
      table.resolve("mm_kernel", 2, TuneKeyView {dims, 2, dtypes, 2}, &ctx);
    } catch (const triton_jit::FrozenMissError&) {
      refused = true;
    }
    EXPECT(refused);
  }
  table.set_resolver(nullptr);
  table.clear();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "torch_tuned_resolver: all checks passed" << std::endl;
  return 0;
}
