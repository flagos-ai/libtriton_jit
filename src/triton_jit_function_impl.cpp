/**
 * @file triton_jit_function_impl.cpp
 * @brief Template specialization implementations for TritonJITFunctionImpl
 *
 * This file contains the non-template member function implementations
 * (constructor and get_kernel) for TritonJITFunctionImpl.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#include "triton_jit/triton_jit_function_impl.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "pybind11/embed.h"

namespace triton_jit {

/**
 * @brief Ensure Python interpreter is initialized
 */
static void ensure_initialized() {
    // When using libtriton_jit with a python C-extension, it is already initialized
    c10::initLogging();
    if (!Py_IsInitialized()) {
        Py_InitializeEx(false);
    }
}

// ========== TritonJITFunctionImpl Constructor ==========

template<BackendPolicy Backend>
TritonJITFunctionImpl<Backend>::TritonJITFunctionImpl(
    std::string_view path,
    std::string_view name
)
    : file_path_(std::string(path))
    , function_name_(std::string(name))
{
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
    this->static_sig_ = StaticSignature{num_args, arg_types};
}

// ========== TritonJITFunctionImpl::get_kernel ==========

template<BackendPolicy Backend>
const TritonKernelImpl<Backend>& TritonJITFunctionImpl<Backend>::get_kernel(
    std::string_view _signature,
    int num_warps,
    int num_stages,
    int device_index
) const {
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
            ans = fn(this->file_path_,
                    this->function_name_,
                    signature,
                    num_warps,
                    num_stages,
                    device_index);
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
            throw std::runtime_error(
                "Unable to emplace the kernel into TritonJITFunction's cache");
        }
    }
    return pos->second;
}

// ========== Explicit Template Instantiations ==========

// Instantiate for CudaBackend
#include "triton_jit/backends/cuda_backend.h"
template class TritonJITFunctionImpl<CudaBackend>;

// Future backends can be added here:
// #include "triton_jit/backends/npu_backend.h"
// template class TritonJITFunctionImpl<NpuBackend>;

} // namespace triton_jit
