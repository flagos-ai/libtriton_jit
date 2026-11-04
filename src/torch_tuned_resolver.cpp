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

#include "triton_jit/torch/tuned_resolver.h"

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>  // pybind11 caster for at::Tensor (libtorch_python)

#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#if defined(BACKEND_CUDA)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

#include "triton_jit/jit_utils.h"

namespace triton_jit {
namespace torch_resolver {

  namespace py = pybind11;

  namespace {

    // `resolver` is the Python module: a TritonDtype becomes its marker object,
    // which resolve() turns into the triton.language dtype.
    py::object to_python(const ArgValue& value, const py::module_& resolver) {
      return std::visit(
          [&resolver](const auto& inner) -> py::object {
            using T = std::decay_t<decltype(inner)>;
            if constexpr (std::is_same_v<T, at::Tensor>) {
              return py::cast(inner);
            } else if constexpr (std::is_same_v<T, bool>) {
              return py::bool_(inner);
            } else if constexpr (std::is_same_v<T, int64_t>) {
              return py::int_(inner);
            } else if constexpr (std::is_same_v<T, double>) {
              return py::float_(inner);
            } else if constexpr (std::is_same_v<T, TritonDtype>) {
              return resolver.attr("TritonDtype")(py::str(inner.name));
            } else {
              return py::str(inner);
            }
          },
          value);
    }

    TunedValue from_python(const py::handle& value, const std::string& what) {
      if (py::isinstance<py::bool_>(value)) {  // before int_: bool is an int in Python
        return value.cast<bool>();
      }
      if (py::isinstance<py::int_>(value)) {
        return value.cast<int64_t>();
      }
      if (py::isinstance<py::float_>(value)) {
        return value.cast<double>();
      }
      if (py::isinstance<py::str>(value)) {
        return value.cast<std::string>();
      }
      throw std::runtime_error("tuned resolver: " + what + " has an unsupported Python type " +
                               std::string(py::str(py::type::of(value))));
    }

    std::optional<TunedTable::ResolvedEntry> parse_answer(const py::dict& answer,
                                                          std::string_view kernel_id,
                                                          TuneKeyView key,
                                                          std::set<std::string>& reported) {
      if (!answer.contains("kernel_id") || answer["kernel_id"].cast<std::string>() != kernel_id)
        throw std::runtime_error("tuned resolver returned a different kernel identity");
      if (answer.contains("unsupported")) {
        const std::string reason = py::str(answer["unsupported"]);
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        if (reported.insert(std::string(kernel_id)).second) {
          std::cerr << "[triton_jit::tuned] resolver cannot serve '" << kernel_id << "': " << reason
                    << " (falling back)\n";
        }
        return std::nullopt;
      }
      TunedTable::ResolvedEntry entry;
      for (const py::handle& column : answer["key_columns"].cast<py::list>()) {
        const py::dict item = column.cast<py::dict>();
        const std::string name = py::str(item["name"]);
        const std::string strategy_name =
            item.contains("strategy") ? std::string(py::str(item["strategy"])) : "default";
        const auto strategy = parse_key_strategy(strategy_name);
        if (!strategy) {
          throw std::runtime_error("tuned resolver: unknown key strategy '" + strategy_name +
                                   "' for column '" + name + "' of " + std::string(kernel_id));
        }
        entry.key_columns.emplace_back(name, *strategy);
      }
      if (entry.key_columns.size() != key.ndims || answer["dtype_keys"].cast<size_t>() != key.ndtypes)
        throw std::runtime_error("tuned resolver returned a different key schema");
      auto normalized = answer["key"].cast<py::list>();
      if (normalized.size() != key.ndims + key.ndtypes)
        throw std::runtime_error("tuned resolver returned a different key width");
      for (size_t i = 0; i < key.ndims; ++i)
        if (normalized[i].cast<int64_t>() != normalize_dim(entry.key_columns[i].second, key.dims[i]))
          throw std::runtime_error("tuned resolver returned a different normalized key");
      for (size_t i = 0; i < key.ndtypes; ++i)
        if (normalized[key.ndims + i].cast<std::string>() != key.dtypes[i])
          throw std::runtime_error("tuned resolver returned different dtypes");
      if (answer.contains("num_warps")) {
        entry.config.num_warps = answer["num_warps"].cast<int>();
      }
      if (answer.contains("num_stages")) {
        entry.config.num_stages = answer["num_stages"].cast<int>();
      }
      if (answer.contains("extra") && !answer["extra"].is_none()) {
        for (const auto& item : answer["extra"].cast<py::dict>()) {
          entry.config.extra[std::string(py::str(item.first))] = std::string(py::str(item.second));
        }
      }
      if (answer.contains("kwargs") && !answer["kwargs"].is_none()) {
        for (const py::handle& pair : answer["kwargs"].cast<py::list>()) {
          const py::sequence kv = pair.cast<py::sequence>();
          const std::string name = py::str(kv[0]);
          entry.config.kwargs.emplace_back(name, from_python(kv[1], "kwarg '" + name + "'"));
        }
      }
      return entry;
    }

  }  // namespace

  TunedTable::Resolver make_resolver(Options options) {
    auto reported = std::make_shared<std::set<std::string>>();
    auto opts = std::make_shared<const Options>(std::move(options));
    TunedTable::Resolver resolver = [reported, opts](
                                        std::string_view kernel_id,
                                        int device_index,
                                        TuneKeyView key,
                                        const void* context) -> std::optional<TunedTable::ResolvedEntry> {
      const auto* resolve_args = static_cast<const ResolveArgs*>(context);
      if (resolve_args == nullptr) {
        return std::nullopt;
      }
      ensure_initialized();
      py::gil_scoped_acquire gil;
      try {
#if defined(BACKEND_CUDA)
        std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
        auto check_device = [&](const ArgValue& value) {
          if (const auto* tensor = std::get_if<at::Tensor>(&value); tensor && tensor->is_cuda()) {
            if (tensor->get_device() != device_index)
              throw std::runtime_error("tuned resolver: tensor device differs from requested device");
            if (!stream_guard)
              stream_guard.emplace(
                  c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(resolve_args->launch_stream),
                                                   device_index));
          }
        };
        for (const auto& value : resolve_args->args) check_device(value);
        for (const auto& [name, value] : resolve_args->kwargs) check_device(value);
#endif
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path");
        if (!opts->extra_sys_path.empty()) {
          path.insert(0, opts->extra_sys_path);
        }
        path.insert(0, get_script_dir().string());
        py::module_ mod = py::module_::import(opts->module.c_str());
        py::list args;
        for (const auto& value : resolve_args->args) {
          args.append(to_python(value, mod));
        }
        py::dict kwargs;
        for (const auto& [name, value] : resolve_args->kwargs) {
          kwargs[py::str(name)] = to_python(value, mod);
        }
        py::object source_path =
            resolve_args->source_path.empty() ? py::none() : py::object(py::cast(resolve_args->source_path));
        py::object grid = resolve_args->grid.empty() ? py::none() : py::object(py::cast(resolve_args->grid));
        // The device the table entry is for, and the launch stream, reach
        // Python too: the resolver makes the device current for backends this
        // bridge cannot guard here.
        py::object answer = mod.attr("resolve")(
            opts->package,
            std::string(kernel_id),
            args,
            kwargs,
            py::arg("source_path") = source_path,
            py::arg("grid") = grid,
            py::arg("device_index") = py::int_(device_index),
            py::arg("stream") = py::int_(reinterpret_cast<uintptr_t>(resolve_args->launch_stream)));
        auto entry = parse_answer(answer.cast<py::dict>(), kernel_id, key, *reported);
        if (entry)
          entry->cache_namespace = resolve_args->cache_namespace.empty() ? resolve_args->source_path
                                                                         : resolve_args->cache_namespace;
        return entry;
      } catch (const py::error_already_set& error) {
        throw std::runtime_error("tuned resolver: Python raised while resolving '" + std::string(kernel_id) +
                                 "': " + error.what());
      }
    };
    resolver.identity = [](std::string_view kernel, const void* context) {
      const auto* args = static_cast<const ResolveArgs*>(context);
      return scoped_kernel_id(
          kernel,
          args ? (args->cache_namespace.empty() ? args->source_path : args->cache_namespace)
               : std::string_view {});
    };
    resolver.wait = [](std::shared_future<const TunedConfig*>& future) {
      if (Py_IsInitialized() && PyGILState_Check()) {
        py::gil_scoped_release release;
        return future.get();
      }
      return future.get();
    };
    auto callback = static_cast<TunedTable::Resolver::Function>(resolver);
    resolver.with_stream =
        [callback](std::string_view kernel, int device, TuneKeyView key, const void* context, void* stream) {
          if (!context) return std::optional<TunedTable::ResolvedEntry> {};
          auto local = *static_cast<const ResolveArgs*>(context);
          local.launch_stream = stream;
          return callback(kernel, device, key, &local);
        };
    return resolver;
  }

}  // namespace torch_resolver
}  // namespace triton_jit
