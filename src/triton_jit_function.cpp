#include "triton_jit/triton_jit_function.h"

#include <string>
#include <vector>

#include "c10/util/Logging.h"  // 仅用于 c10::initLogging()，如果不需要也可去掉
#include "fmt/core.h"
#include "pybind11/embed.h"

namespace triton_jit {
std::unordered_map<std::string, TritonJITFunction> TritonJITFunction::functions_;

void ensure_initialized() {
  // When using libtriton_jit with a python C-extension, it is already initialized
  c10::initLogging();
  if (!Py_IsInitialized()) {
    Py_InitializeEx(false);
  }
}

TritonJITFunction::TritonJITFunction(std::string_view path, std::string_view name)
    : file_path_(std::string(path)), function_name_(std::string(name)) {
  namespace py = pybind11;
  ensure_initialized();
  py::gil_scoped_acquire gil;

  std::filesystem::path script_dir = get_script_dir();
  py::module_ sys = py::module_::import("sys");
  sys.attr("path").attr("insert")(0, script_dir.c_str());
  py::module_ mod = py::module_::import("gen_ssig");
  py::object fn = mod.attr("extract_static_signature");
  py::object ans = fn(this->file_path_, this->function_name_);
  py::list arg_types_raw = ans.cast<py::list>();

  int num_args = arg_types_raw.size();
  std::vector<ArgType> arg_types;
  arg_types.reserve(num_args);
  for (auto item : arg_types_raw) {
    try {
      arg_types.push_back(ArgType(item.cast<int>()));
    } catch (const py::cast_error&) {
      // Invalid argument type, skip
    }
  }
  this->static_sig_ = StaticSignature {num_args, arg_types};
}

const TritonKernel& TritonJITFunction::get_kernel(std::string_view _signature,
                                                  int num_warps,
                                                  int num_stages,
                                                  int device_index) const {
  std::string signature(_signature);
  std::string key = fmt::format("{};{}", signature, device_index);
  auto pos = this->overloads_.find(key);
  if (pos == this->overloads_.end()) {
    namespace py = pybind11;
    ensure_initialized();
    py::gil_scoped_acquire gil;

    std::filesystem::path script_dir = get_script_dir();
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, script_dir.c_str());
    py::module_ mod = py::module_::import("standalone_compile");
    py::object fn = mod.attr("compile_a_kernel");
    py::object ans;
    try {
      ans = fn(this->file_path_, this->function_name_, signature, num_warps, num_stages, device_index);
    } catch (const py::error_already_set&) {
      throw;
    }
    std::string cache_dir = ans.cast<std::string>();
    TritonKernel k(cache_dir, this->function_name_);
    auto result = this->overloads_.emplace(std::move(key), std::move(k));
    if (result.second) {
      pos = result.first;
    } else {
      throw std::runtime_error("Failed to cache compiled kernel");
    }
  }
  return pos->second;
}

TritonJITFunction& TritonJITFunction::get_instance(std::string_view path, std::string_view name) {
  std::string function_id = fmt::format("{}:{}", path, name);
  auto pos = TritonJITFunction::functions_.find(function_id);

  if (pos == TritonJITFunction::functions_.end()) {
    TritonJITFunction f(path, name);
    auto result = TritonJITFunction::functions_.emplace(std::move(function_id), std::move(f));
    if (result.second) {
      pos = result.first;
    } else {
      throw std::runtime_error("Failed to cache JIT function");
    }
  }
  return pos->second;
}

void TritonJITFunction::launch_with_raw_args(aclrtStream stream,
                                             unsigned int grid_x,
                                             unsigned int grid_y,
                                             unsigned int grid_z,
                                             unsigned int num_warps,
                                             unsigned int num_stages,
                                             std::string full_signature,
                                             void** args,
                                             size_t num_args) const {
  aclrtContext ctx = nullptr;
  aclrtGetCurrentContext(&ctx);

  int device_index = -1;
  aclError err = aclrtGetDevice(&device_index);
  if (err != ACL_SUCCESS) {
    device_index = 0;
  }

  const TritonKernel& kernel = this->get_kernel(full_signature, num_warps, num_stages, device_index);
  kernel.launch(grid_x, grid_y, grid_z, num_warps, stream, args);
}
}  // namespace triton_jit
