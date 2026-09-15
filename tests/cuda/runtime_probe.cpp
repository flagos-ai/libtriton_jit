#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>  // at::Tensor <-> Python
#include <memory>
#include <vector>
#include "triton_jit/freeze.h"
#include "triton_jit/torch/tuned_resolver.h"
#include "triton_jit/triton_jit_function.h"
namespace py = pybind11;
using namespace triton_jit;

PYBIND11_MODULE(tuned_runtime_probe, m) {
  m.def("install", [](const std::string& path) {
    auto& table = TunedTable::instance();
    table.clear();
    table.set_resolver(torch_resolver::make_resolver({"fake_package", "fake_tuned_resolver", path}));
  });
  m.def("resolve", [](const std::string& name, const std::string& source, bool frozen, uintptr_t stream) {
    torch_resolver::ResolveArgs ctx;
    ctx.source_path = source;
    const int64_t dims[] = {1000, 4000};
    const char* dtypes[] = {"torch.float16", "torch.bfloat16"};
    std::unique_ptr<ScopedFreeze> guard;
    if (frozen) guard = std::make_unique<ScopedFreeze>();
    const auto* cfg =
        TunedTable::instance().resolve(name, 0, {dims, 2, dtypes, 2}, &ctx, reinterpret_cast<void*>(stream));
    py::dict result;
    if (cfg) {
      result["source"] = *cfg->get_str("SRC");
      result["calls"] = cfg->get_i64("CALLS", -1);
    }
    return result;
  });
  m.def(
      "launch",
      [](const std::string& path, uintptr_t x, uintptr_t y, int n, int block, uintptr_t stream, bool frozen) {
        std::unique_ptr<ScopedFreeze> guard;
        if (frozen) guard = std::make_unique<ScopedFreeze>();
        auto& f = TritonJITFunction::get_instance(path, "vector_kernel");
        f(reinterpret_cast<CUstream>(stream),
          (n + block - 1) / block,
          1,
          1,
          4,
          1,
          device_ptr<float>(reinterpret_cast<float*>(x)),
          device_ptr<float>(reinterpret_cast<float*>(y)),
          n,
          block);
      });
  // ---- plain @triton.autotune kernels through the real resolver script ----
  m.def("install_native", [](const std::string& package, const std::string& extra_sys_path) {
    auto& table = TunedTable::instance();
    table.clear();
    table.set_resolver(torch_resolver::make_resolver({package, "tuned_resolver", extra_sys_path}));
  });
  // resolve() for a kernel (x, y, n[, DTYPE]) with real tensors, an optional
  // TritonDtype constexpr and the launch grid; {} on a miss.
  m.def("resolve_native",
        [](const std::string& name,
           const std::string& source,
           at::Tensor x,
           at::Tensor y,
           int64_t n,
           std::vector<std::string> dtype_names,
           const std::string& dtype_constexpr,
           const std::string& grid,
           bool frozen,
           uintptr_t stream) {
          torch_resolver::ResolveArgs ctx;
          ctx.source_path = source;
          ctx.args = {x, y, n};
          if (!dtype_constexpr.empty())
            ctx.kwargs = {
                {"DTYPE", torch_resolver::TritonDtype {dtype_constexpr}}
            };
          ctx.grid = grid;
          const int64_t dims[] = {n};
          std::vector<const char*> dtypes;
          for (const auto& dtype : dtype_names) dtypes.push_back(dtype.c_str());
          std::unique_ptr<ScopedFreeze> guard;
          if (frozen) guard = std::make_unique<ScopedFreeze>();
          const auto* cfg = TunedTable::instance().resolve(name,
                                                           0,
                                                           {dims, 1, dtypes.data(), dtypes.size()},
                                                           &ctx,
                                                           reinterpret_cast<void*>(stream));
          py::dict result;
          if (cfg) {
            result["BLOCK"] = cfg->get_i64("BLOCK", -1);
            result["num_warps"] = cfg->num_warps;
            result["num_stages"] = cfg->num_stages;
          }
          return result;
        });
  m.def("load_table", [](const std::string& path) {
    TunedTable::instance().clear();
    TunedTable::instance().load(path, 0);
  });
  m.def("find_native",
        [](const std::string& name,
           const std::string& source,
           int64_t n,
           std::vector<std::string> dtype_names) {
          const int64_t dims[] = {n};
          std::vector<const char*> dtypes;
          for (const auto& dtype : dtype_names) dtypes.push_back(dtype.c_str());
          const auto* cfg = TunedTable::instance().find(scoped_kernel_id(name, source),
                                                        0,
                                                        {dims, 1, dtypes.data(), dtypes.size()});
          py::dict result;
          if (cfg) {
            result["BLOCK"] = cfg->get_i64("BLOCK", -1);
            result["num_warps"] = cfg->num_warps;
            result["num_stages"] = cfg->num_stages;
          }
          return result;
        });
  m.def("prepare", [](const std::string& path, uintptr_t x, uintptr_t y, int n, int block) {
    auto& f = TritonJITFunction::get_instance(path, "vector_kernel");
    ParameterBuffer buffer;
    c10::SmallVector<std::string> signature;
    ArgHandle handler = {f.get_static_sig(), buffer, signature, 0};
    handler.handle_args(device_ptr<float>(reinterpret_cast<float*>(x)),
                        device_ptr<float>(reinterpret_cast<float*>(y)),
                        n,
                        block);
    CompileOptions opts;
    opts.num_warps = 4;
    opts.num_stages = 1;
    f.prepare(join_sig(signature), opts, 0);
    return f.is_prepared(join_sig(signature), opts, 0);
  });
}
