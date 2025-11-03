/**
 * @file triton_jit_function.h
 * @author your name (you@domain.com)
 * @brief TritonJITFunction is a class that wraps triton jit functions so as to be called
 * in c++ project.
 * @version 0.1
 * @date 2025-07-04
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "cuda.h"

#include "fmt/core.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/triton_kernel.h"

namespace triton_jit {

/**
 * @brief An enum to describe how an argument is handled by the runtime.
 *
 */
enum struct ArgType : int8_t {
  NON_CONSTEXPR = 0,  // non-constexpr argument that is not specialized
  SPECIALIZED = 1,    // non-constexpr argument that is specialized
  CONSTEXPR = 2,      // constexpr argument(argument to the compiler instead of the kernel)
};

/**
 * @brief Description of a triton jit function on how it handles its arguments
 *
 * StaticSignature is dependent only on the function definition (and the triton.jit
 * decorator) itself without passing actual arguments. This is what the 'static' here
 * means
 */
struct StaticSignature {
  int num_args;
  std::vector<ArgType> arg_type;

  const ArgType &at(size_t i) const {
    return arg_type.at(i);
  }
};

/**
 * @brief An class to wrap triton jit function for it to be called in c++.
 *
 * Wrap a triton jit function given the path in which it is defined and the function
 * name, then you can call it in c++ in almost the same way as in python.
 */
class TritonJITFunction {
 private:
  std::string file_path_;
  std::string function_name_;
  StaticSignature static_sig_;
  // the cached compiled TritonKernel of this TritonJITFunction
  mutable std::unordered_map<std::string, TritonKernel> overloads_;

  // a registry to hold all TritonJITFunctions
  static std::unordered_map<std::string, TritonJITFunction> functions_;

 public:
  static TritonJITFunction &get_instance(std::string_view path, std::string_view name);
  TritonJITFunction(const TritonJITFunction &) = delete;
  TritonJITFunction &operator=(const TritonJITFunction &) = delete;
  TritonJITFunction(TritonJITFunction &&) = default;
  TritonJITFunction &operator=(TritonJITFunction &&) = default;

  template <typename... Args>
  void operator()(CUstream stream,
                  unsigned int grid_x,
                  unsigned int grid_y,
                  unsigned int grid_z,
                  unsigned int num_warps,
                  unsigned int num_stages,
                  Args... args) const;

  /**
   * A Low level API to launch Triton Kernel directly with pointers to all kernel args. This is
   * a thin wrapper around cuLaunchKernel. It is experimental and subject to change. It is
   * designed to be used manual argument processing. An argument-buffer-like design is working in
   * progress now to support more flexible argument processing.
   */
  void launch_with_raw_args(CUstream stream,
                            unsigned int grid_x,
                            unsigned int grid_y,
                            unsigned int grid_z,
                            unsigned int num_warps,
                            unsigned int num_stages,
                            std::string full_signature,
                            void **args) const;

 private:
  TritonJITFunction(std::string_view path, std::string_view name);

  /**
   * Get or Add a TritonKernel corresponding to the signature, compile options and device index.
   * It may trigger triton.compile via the embedded python interpreter.
   */
  const TritonKernel &get_kernel(std::string_view signature,
                                 int num_warps,
                                 int num_stages,
                                 CUdevice device_index) const;
};

struct ArgHandle {
  const StaticSignature &ssig;
  /* data pointer of Tensors;
  It is not that straigt extract data pointer from a tensor, since it is encapsulated
  by Storage. We gather data pointers here for them to live out of the loop while iterating
  over arguments.*/
  c10::SmallVector<void *> &data_pointers;
  c10::SmallVector<void *> &kernel_args;
  c10::SmallVector<std::string> &signature;
  int idx;

  /***
   * Iterate over the args and populate data_pointers, kernel_args and signature according to
   * to rules of Triton's jit runtime.
   */
  template <typename... Args>
  void handle_args(Args... args) {
    (handle_arg(args), ...);
  }

  template <typename T>
  void handle_arg(const T &item) {
    if constexpr (is_optional<decltype(item)>::value) {
      handle_optional(item);
    } else if constexpr (is_same_ignore_cvref<c10::Scalar, T>::value) {
      handle_scalar(item);
    } else {
      handle_arg_plain(item);
    }
  }

  template <typename T>
  void handle_optional(const std::optional<T> &item) {
    if (item.has_value()) {
      const T &v = item.value();
      handle_arg(v);
    } else {
      handle_arg(std::nullopt);
    }
  }

  void handle_scalar(const c10::Scalar &item) {
    TORCH_CHECK(!item.isSymbolic());
    c10::ScalarType tp = item.type();
    const void *p = item.data_ptr();
    if (tp == c10::ScalarType::Bool) {
      handle_arg_plain(*reinterpret_cast<const bool *>(p));
    } else if (tp == c10::ScalarType::Long) {
      handle_arg_plain(*reinterpret_cast<const int64_t *>(p));
    } else if (tp == c10::ScalarType::UInt64) {
      handle_arg_plain(*reinterpret_cast<const uint64_t *>(p));
    } else if (tp == c10::ScalarType::Double) {
      handle_arg_plain(*reinterpret_cast<const double *>(p));
    } else {
      throw std::runtime_error("unsupported scalar type.");
    }
  }

  template <typename T>
  void handle_arg_plain(const T &item) {
    if constexpr (is_same_ignore_cvref<at::Tensor, T>::value) {
      handle_tensor(item);
    } else if constexpr (is_same_ignore_cvref<std::nullopt_t, T>::value) {
      // Assumption nullopt is alway treated as constexpr,
      // even if the parameter is not marked as constexpr
      signature.push_back("nullopt");
    } else {
      if (ssig.at(idx) == ArgType::CONSTEXPR) {  // constexpr
        handle_constexpr(item);
      } else if (ssig.at(idx) == ArgType::SPECIALIZED) {  // specialzied
        handle_specialized(item);
      } else {  // ArgType::NON-CONSTEXPR
        handle_non_constexpr(item);
      }
    }
    idx++;
  }

  void handle_tensor(const at::Tensor &item) {
    // Assumuption: Tensor is never constexpr
    TORCH_CHECK(this->ssig.at(idx) != ArgType::CONSTEXPR);
    void *p_item = item.data_ptr();
    data_pointers.push_back(p_item);
    kernel_args.push_back(&(data_pointers.back()));
    const char *dtype = to_triton_typename(item.scalar_type());

    const char *specialization = "";
    if (ssig.at(idx) == ArgType::SPECIALIZED) {
      specialization = spec(reinterpret_cast<std::uintptr_t>(data_pointers.back()));
    }
    std::string sig_for_idx = fmt::format("*{}{}", dtype, specialization);
    signature.push_back(sig_for_idx);
  }

  template <typename T>
  void handle_constexpr(const T &item) {
    signature.push_back(fmt::format("{}", item));
  }

  template <typename T>
  void handle_specialized(const T &item) {
    const char *dtype = triton_type<decltype(item)>::name;
    if constexpr (std::is_integral_v<std::remove_cv_t<std::remove_reference_t<decltype(item)>>>) {
      const char *specialization = spec(item);
      if (specialization != ":1") {
        const void *p_item = &item;
        // cuLaunchKernel requires `void*`, so if the argument is const,
        // we need to const_cast to remove the const qualifier to call it
        kernel_args.push_back(const_cast<void *>(p_item));
      }
      std::string sig_for_idx = fmt::format("{}{}", dtype, specialization);
      signature.push_back(sig_for_idx);
    } else {
      const void *p_item = &item;
      kernel_args.push_back(const_cast<void *>(p_item));
      std::string sig_for_idx = fmt::format("{}", dtype);
      signature.push_back(sig_for_idx);
    }
  }

  template <typename T>
  void handle_non_constexpr(const T &item) {
    const void *p_item = &item;
    kernel_args.push_back(const_cast<void *>(p_item));
    const char *dtype = triton_type<decltype(item)>::name;
    signature.push_back(dtype);
  }
};

/***
 * The mainly used method of TritonJITFunction. It can be used with different triton functions
 * with different arguments. The main work are signature extraction; kernel arg extraction and
 * kernel launch.
 *
 * Arguments consist of 2 parts:
 * fixed part: stream, grid, compile options;
 * variadic part: arguments to the triton function.ArgHandle
 *
 * TODO:
 * customization point: compile options for different backends may be different.
 */
template <typename... Args>
void TritonJITFunction::operator()(CUstream stream,
                                   unsigned int grid_x,
                                   unsigned int grid_y,
                                   unsigned int grid_z,
                                   unsigned int num_warps,
                                   unsigned int num_stages,
                                   Args... args) const {
  const int num_args = this->static_sig_.num_args;

  // since we need to take address of all the arguemnts to the kernel to launch a kernel
  // but data pointers are not the arguement of the function operator(), they are local variables
  // that are created in `arg_handle`, to take the addresses of them, we need to keep them alive
  // out of the function
  c10::SmallVector<void *> data_pointers;
  data_pointers.reserve(num_args);
  c10::SmallVector<void *> kernel_args;
  kernel_args.reserve(num_args);
  c10::SmallVector<std::string> signature;
  signature.reserve(num_args);

  ArgHandle handler = {this->static_sig_, data_pointers, kernel_args, signature, 0};
  (handler.handle_arg(args), ...);

  // global scratch: introduced in triton 3.3
  void *global_scratch = nullptr;
  data_pointers.push_back(global_scratch);
  kernel_args.push_back(&(data_pointers.back()));
  std::string full_signature;
  for (int i = 0; i < signature.size(); i++) {
    if (i == 0) {
      full_signature += signature[i];
    } else {
      full_signature += ",";
      full_signature += signature[i];
    }
  }
  // LOG(INFO) << fmt::format("full signature is {}", full_signature);
  // LOG(INFO) << "raw_args_list.size(): " << kernel_args.size() << std::endl;

  // TODO: use torch backend-agnostic device APIs
  ensure_cuda_context();
  CUdevice device_index;
  checkCudaErrors(cuCtxGetDevice(&device_index));
  const TritonKernel &kernel = this->get_kernel(full_signature, num_warps, num_stages, device_index);
  kernel.launch(grid_x, grid_y, grid_z, num_warps, stream, kernel_args.data());
  return;
}
static_assert(std::is_move_constructible_v<TritonJITFunction>);
}  // namespace triton_jit
