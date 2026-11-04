#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>
#include <memory>
#include <string>
#include <vector>
#include "triton_jit/freeze.h"
#include "triton_jit/torch/tuned_resolver.h"
#include "triton_jit/triton_jit_function.h"

namespace py = pybind11;
using namespace triton_jit;

namespace {
CompileOptions compile_options(int auto_blockify) {
  CompileOptions options;
  options.num_warps = 4;
  options.num_stages = 1;
  if (auto_blockify >= 0) {
    options.extra["enable_auto_blockify"] = auto_blockify ? "true" : "false";
  }
  return options;
}
}  // namespace

PYBIND11_MODULE(npu_runtime_probe, m) {
  m.def("info", [] {
    uint32_t cores = 0;
    const auto status = rtGetAiCoreCount(&cores);
    auto fingerprint = detect_backend_fingerprint(0);
    py::dict result;
    result["ai_core_count"] = cores;
    result["ai_core_status"] = static_cast<int>(status);
    result["device_index"] = NpuBackend::get_device_index();
    result["backend"] = fingerprint.backend;
    result["device_name"] = fingerprint.device_name;
    return result;
  });
  m.def(
      "launch",
      [](const std::string& path,
         const std::string& name,
         at::Tensor a,
         at::Tensor x,
         at::Tensor y,
         int64_t rows,
         int64_t cols,
         int64_t block_m,
         int64_t block_n,
         uintptr_t stream,
         bool frozen,
         int auto_blockify) {
        std::unique_ptr<ScopedFreeze> guard;
        if (frozen) guard = std::make_unique<ScopedFreeze>();
        auto& function = TritonJITFunction::get_instance(path, name);
        function(reinterpret_cast<aclrtStream>(stream),
                 static_cast<unsigned>((rows + block_m - 1) / block_m),
                 1,
                 1,
                 compile_options(auto_blockify),
                 a,
                 x,
                 y,
                 rows,
                 cols,
                 block_m,
                 block_n);
      },
      py::arg("path"),
      py::arg("name"),
      py::arg("a"),
      py::arg("x"),
      py::arg("y"),
      py::arg("rows"),
      py::arg("cols"),
      py::arg("block_m"),
      py::arg("block_n"),
      py::arg("stream"),
      py::arg("frozen"),
      py::arg("auto_blockify") = -1);
  m.def("prepare",
        [](const std::string& path,
           const std::string& name,
           at::Tensor a,
           at::Tensor x,
           at::Tensor y,
           int64_t rows,
           int64_t cols,
           int64_t block_m,
           int64_t block_n,
           uintptr_t stream) {
          auto& function = TritonJITFunction::get_instance(path, name);
          ParameterBuffer buffer;
          c10::SmallVector<std::string> signature;
          ArgHandle handler = {function.get_static_sig(), buffer, signature, 0};
          handler.handle_args(a, x, y, rows, cols, block_m, block_n);
          CompileOptions options;
          options.num_warps = 4;
          options.num_stages = 1;
          function.prepare(join_sig(signature),
                           options,
                           NpuBackend::get_device_index(),
                           reinterpret_cast<void*>(stream));
          return function.is_prepared(join_sig(signature), options, NpuBackend::get_device_index());
        });
  m.def(
      "count_programs",
      [](const std::string& path,
         at::Tensor counts,
         int64_t blocks,
         uintptr_t stream,
         int auto_blockify,
         bool frozen) {
        std::unique_ptr<ScopedFreeze> guard;
        if (frozen) guard = std::make_unique<ScopedFreeze>();
        auto& function = TritonJITFunction::get_instance(path, "counter_kernel");
        function(reinterpret_cast<aclrtStream>(stream),
                 static_cast<unsigned>(blocks),
                 1,
                 1,
                 compile_options(auto_blockify),
                 counts,
                 blocks);
      },
      py::arg("path"),
      py::arg("counts"),
      py::arg("blocks"),
      py::arg("stream"),
      py::arg("auto_blockify") = -1,
      py::arg("frozen") = false);
  m.def("install_resolver", [](const std::string& kernel_directory) {
    auto& table = TunedTable::instance();
    table.clear();
    table.set_resolver(torch_resolver::make_resolver({"npu_kernels", "tuned_resolver", kernel_directory}));
  });
  m.def("resolve",
        [](const std::string& path,
           at::Tensor a,
           at::Tensor x,
           at::Tensor y,
           int64_t rows,
           int64_t cols,
           const std::vector<std::string>& dtype_names,
           uintptr_t stream,
           bool frozen,
           bool find_only) {
          torch_resolver::ResolveArgs context;
          context.source_path = path;
          context.args = {a, x, y, rows, cols};
          context.grid = "(triton.cdiv(M, META['BLOCK_M']),)";
          const int64_t dims[] = {rows, cols};
          std::vector<const char*> dtypes;
          for (const auto& dtype : dtype_names) dtypes.push_back(dtype.c_str());
          const TuneKeyView key {dims, 2, dtypes.data(), dtypes.size()};
          std::unique_ptr<ScopedFreeze> guard;
          if (frozen) guard = std::make_unique<ScopedFreeze>();
          auto& table = TunedTable::instance();
          const auto* config =
              find_only ? table.find(scoped_kernel_id("native_gemv", path), 0, key)
                        : table.resolve("native_gemv", 0, key, &context, reinterpret_cast<void*>(stream));
          py::dict result;
          if (config) {
            result["BLOCK_M"] = config->get_i64("BLOCK_M", -1);
            result["BLOCK_N"] = config->get_i64("BLOCK_N", -1);
            result["num_warps"] = config->num_warps;
            result["num_stages"] = config->num_stages;
          }
          return result;
        });
  m.def("load_table", [](const std::string& path) {
    TunedTable::instance().clear();
    TunedTable::instance().load(path, 0);
  });
}
