#include "triton_jit/triton_jit_function_impl.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "pybind11/embed.h"

namespace triton_jit {

static void ensure_initialized() {
  // Use std::call_once to ensure initialization happens only once
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    c10::initLogging();
    if (!Py_IsInitialized()) {
      Py_InitializeEx(false);
    }
    // Set Python os.environ directly via pybind11
    namespace py = pybind11;
    py::gil_scoped_acquire gil;
    py::module_::import("os").attr("environ")["TRITON_JIT_BACKEND"] = BACKEND_NAME;

    // Import backend-specific modules for device registration
    std::string backend_name(BACKEND_NAME);
    if (backend_name == "mtgpu") {
      try {
        // Import torch_musa to register MUSA as PrivateUse1 backend
        py::module_::import("torch_musa");
      } catch (const py::error_already_set& e) {
        std::cerr << "Warning: Failed to import torch_musa: " << e.what() << std::endl;
      }
    }
  });
}

template <BackendPolicy Backend>
TritonJITFunctionImpl<Backend>::TritonJITFunctionImpl(std::string_view path, std::string_view name)
    : file_path_(std::string(path)), function_name_(std::string(name)) {
  // Embed Python to extract static signature
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
    } catch (const py::cast_error& e) {
      std::cerr << "Type error: " << e.what() << std::endl;
    }
  }
  this->static_sig_ = StaticSignature {num_args, arg_types};
}

template <BackendPolicy Backend>
const TritonKernelImpl<Backend>& TritonJITFunctionImpl<Backend>::get_kernel(std::string_view _signature,
                                                                            int num_warps,
                                                                            int num_stages,
                                                                            int device_index) const {
  std::string signature(_signature);
  std::string key = fmt::format("{};{}", signature, device_index);

  auto pos = this->overloads_.find(key);
  if (pos == this->overloads_.end()) {
    // Compile kernel via Python
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
    } catch (const py::error_already_set& e) {
      std::cerr << "Python exception: " << e.what() << std::endl;
      throw;
    }

    std::string cache_dir = ans.cast<std::string>();
    TritonKernelImpl<Backend> k(cache_dir, this->function_name_);

    auto result = this->overloads_.emplace(std::move(key), std::move(k));
    if (result.second) {
      pos = result.first;
    } else {
      throw std::runtime_error("Unable to emplace the kernel into TritonJITFunction's cache");
    }
  }
  return pos->second;
}

}  // namespace triton_jit

#ifdef BACKEND_NPU
#include "triton_jit/backends/npu_backend.h"
template class triton_jit::TritonJITFunctionImpl<triton_jit::NpuBackend>;
#endif

#ifdef BACKEND_CUDA
#include "triton_jit/backends/cuda_backend.h"
template class triton_jit::TritonJITFunctionImpl<triton_jit::CudaBackend>;
#endif

#ifdef BACKEND_IX
#include "triton_jit/backends/ix_backend.h"
template class triton_jit::TritonJITFunctionImpl<triton_jit::IxBackend>;
#endif

#ifdef BACKEND_MUSA
#include "triton_jit/backends/musa_backend.h"
template class triton_jit::TritonJITFunctionImpl<triton_jit::MusaBackend>;
#endif

#ifdef BACKEND_MLU
#include "triton_jit/backends/mlu_backend.h"
template class triton_jit::TritonJITFunctionImpl<triton_jit::MluBackend>;
#endif

#ifdef BACKEND_HCU
#include "triton_jit/backends/hcu_backend.h"
template class triton_jit::TritonJITFunctionImpl<triton_jit::HcuBackend>;
#endif
