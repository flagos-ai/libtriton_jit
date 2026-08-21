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

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fmt/core.h"
#include "triton_jit/backend_config.h"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_function_arg.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/triton_kernel.h"

namespace triton_jit {

template <typename T>
T get_next_multiple_of(T pos, T step) {
  return ((pos + step - 1) / step) * step;
}

struct ParameterBuffer {
  c10::SmallVector<std::byte> buff_;
  size_t cursor_ = 0;
  c10::SmallVector<size_t> offsets_;

  void reserve(size_t new_cap) {
    const int ESTIMATED_BYTES_PER_ARG = 4;
    this->buff_.reserve(new_cap * ESTIMATED_BYTES_PER_ARG);
    this->offsets_.reserve(new_cap);
  }

  template <typename T>
  void push_arg(T&& v) {
    using U = std::decay_t<T>;
    static_assert(std::is_trivially_copyable_v<U>, "Non trivially copyable type");
    size_t align = alignof(U);
    size_t offset = get_next_multiple_of(this->cursor_, align);
    this->offsets_.push_back(offset);

    size_t size = sizeof(U);
    this->buff_.resize(offset + size);
    std::byte* ptr = this->buff_.data() + offset;
    std::memcpy(ptr, &v, size);

    this->cursor_ = offset + size;
  }

  c10::SmallVector<void*> get_ptrs() {
    c10::SmallVector<void*> ptrs;
    ptrs.reserve(this->offsets_.size());
    std::byte* start = this->buff_.data();
    for (const size_t off : this->offsets_) {
      ptrs.push_back(start + off);
    }
    return ptrs;
  }

  size_t size() const {
    return this->offsets_.size();
  }
};

enum struct ArgType : int8_t {
  NON_CONSTEXPR = 0,
  SPECIALIZED = 1,
  CONSTEXPR = 2,
  SPECIALIZED_NO_ALIGNMENT = 3,
};

struct StaticSignature {
  int num_args;
  std::vector<ArgType> arg_type;

  const ArgType& at(size_t i) const {
    return arg_type.at(i);
  }
};

template <typename T>
struct is_std_tuple : std::false_type {};

template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {};

template <typename T>
inline constexpr bool is_runtime_tuple_element_v =
    is_same_ignore_cvref<int, T>::value || is_same_ignore_cvref<unsigned int, T>::value ||
    is_same_ignore_cvref<int64_t, T>::value || is_same_ignore_cvref<uint64_t, T>::value ||
    is_same_ignore_cvref<float, T>::value || is_same_ignore_cvref<double, T>::value;

struct ArgHandle {
  const StaticSignature& ssig;
  /* data pointer of Tensors;
  It is not straightforward to extract data pointer from a tensor, since it is encapsulated
  by Storage. We gather data pointers here for them to live out of the loop while iterating
  over arguments.*/
  ParameterBuffer& buf;
  c10::SmallVector<std::string>& signature;
  int idx;

  template <typename... Args>
  void handle_args(Args... args) {
    (handle_arg(args), ...);
  }

  template <typename T>
  void handle_arg(const T& item) {
    if constexpr (is_optional<decltype(item)>::value) {
      handle_optional(item);
    } else if constexpr (is_same_ignore_cvref<c10::Scalar, T>::value) {
      handle_scalar(item);
    } else if constexpr (is_same_ignore_cvref<JitFunctionArg, T>::value) {
      handle_jitfunction(item);
    } else if constexpr (is_std_tuple<std::remove_cvref_t<T>>::value) {
      handle_tuple(item);
    } else {
      handle_arg_plain(item);
    }
  }

  void handle_jitfunction(const JitFunctionArg& item) {
    (void)this->ssig.at(idx);
    signature.push_back(item.signature_token());
    idx++;
  }

  template <typename... Ts>
  void handle_tuple(const std::tuple<Ts...>& item) {
    static_assert(sizeof...(Ts) > 0, "Runtime tuple arguments must not be empty");
    static_assert((is_runtime_tuple_element_v<Ts> && ...),
                  "Runtime tuple arguments contain an unsupported scalar type");
    TORCH_CHECK(this->ssig.at(idx) != ArgType::CONSTEXPR,
                "Runtime tuple arguments cannot be constexpr");

    std::string grouped_signature = "(";
    bool first = true;
    auto append_element = [&](const auto& element) {
      if (!first) {
        grouped_signature += ",";
      }
      first = false;
      this->buf.push_arg(element);
      grouped_signature += triton_type<std::remove_cvref_t<decltype(element)>>::name;
    };
    std::apply([&](const auto&... elements) { (append_element(elements), ...); }, item);
    grouped_signature += ")";

    signature.push_back(std::move(grouped_signature));
    idx++;
  }

  template <typename T>
  void handle_optional(const std::optional<T>& item) {
    if (item.has_value()) {
      const T& v = item.value();
      handle_arg(v);
    } else {
      handle_arg(std::nullopt);
    }
  }

  void handle_scalar(const c10::Scalar& item) {
    TORCH_CHECK(!item.isSymbolic());
    c10::ScalarType tp = item.type();
    const void* p = item.data_ptr();
    if (tp == c10::ScalarType::Bool) {
      handle_arg_plain(*reinterpret_cast<const bool*>(p));
    } else if (tp == c10::ScalarType::Long) {
      handle_arg_plain(*reinterpret_cast<const int64_t*>(p));
    } else if (tp == c10::ScalarType::UInt64) {
      handle_arg_plain(*reinterpret_cast<const uint64_t*>(p));
    } else if (tp == c10::ScalarType::Double) {
#if defined(BACKEND_GCU)
      float f = static_cast<float>(*reinterpret_cast<const double*>(p));
      handle_arg_plain(f);
#else
      handle_arg_plain(*reinterpret_cast<const double*>(p));
#endif
    } else {
      throw std::runtime_error("unsupported scalar type.");
    }
  }

  template <typename T>
  void handle_arg_plain(const T& item) {
    if constexpr (is_same_ignore_cvref<at::Tensor, T>::value) {
      handle_tensor(item);
    } else if constexpr (is_same_ignore_cvref<std::nullopt_t, T>::value) {
      // Assumption: nullopt is always treated as constexpr,
      // even if the parameter is not marked as constexpr
      signature.push_back("nullopt");
    } else if constexpr (std::is_same_v<std::decay_t<T>, const char*> ||
                         std::is_same_v<std::decay_t<T>, char*> ||
                         is_same_ignore_cvref<std::string, T>::value ||
                         is_same_ignore_cvref<std::string_view, T>::value) {
      // A string argument (e.g. a dtype spelled "tl.float32") can only ever be
      // a constexpr Triton parameter. Route it to handle_constexpr at compile
      // time so the specialized / non-constexpr paths -- which require
      // triton_type<T>::name and do not specialize for strings -- are never
      // instantiated for a string type.
      handle_constexpr(item);
    } else {
      if (ssig.at(idx) == ArgType::CONSTEXPR) {  // constexpr
        handle_constexpr(item);
      } else if (ssig.at(idx) == ArgType::SPECIALIZED) {  // specialized
        handle_specialized(item);
      } else if (ssig.at(idx) == ArgType::SPECIALIZED_NO_ALIGNMENT) {
        handle_specialized_no_alignment(item);
      } else {  // ArgType::NON_CONSTEXPR
        handle_non_constexpr(item);
      }
    }
    idx++;
  }

  void handle_tensor(const at::Tensor& item) {
    // Assumption: Tensor is never constexpr
    TORCH_CHECK(this->ssig.at(idx) != ArgType::CONSTEXPR);
    void* p_item = item.data_ptr();
    this->buf.push_arg(p_item);
    const char* dtype = to_triton_typename(item.scalar_type());

    const char* specialization = "";
    if (ssig.at(idx) == ArgType::SPECIALIZED) {
#if defined(BACKEND_NPU)
      // NPU: disable :1 specialization to keep arg list consistent
#else
      specialization = spec(reinterpret_cast<std::uintptr_t>(p_item));
#endif
    }
    std::string sig_for_idx = fmt::format("*{}{}", dtype, specialization);
    signature.push_back(sig_for_idx);
  }

  template <typename T>
  void handle_constexpr(const T& item) {
    signature.push_back(fmt::format("{}", item));
  }

  template <typename T>
  void handle_specialized(const T& item) {
    const char* dtype = narrow_type_name(item);
    if constexpr (std::is_integral_v<std::remove_cv_t<std::remove_reference_t<decltype(item)>>>) {
      const char* specialization = "";
#if defined(BACKEND_NPU)
      // NPU: disable :1 specialization so args are always passed
      this->buf.push_arg(item);
#else
      specialization = spec(item);
      if (specialization != ":1") {
        this->buf.push_arg(item);
      }
#endif
      std::string sig_for_idx = fmt::format("{}{}", dtype, specialization);
      signature.push_back(sig_for_idx);
    } else {
      this->buf.push_arg(item);
      std::string sig_for_idx = fmt::format("{}", dtype);
      signature.push_back(sig_for_idx);
    }
  }

  template <typename T>
  void handle_specialized_no_alignment(const T& item) {
    const char* dtype = narrow_type_name(item);
    if constexpr (std::is_integral_v<std::remove_cv_t<std::remove_reference_t<decltype(item)>>>) {
      const bool equal_to_1 = item == 1;
#if defined(BACKEND_NPU)
      this->buf.push_arg(item);
      signature.push_back(dtype);
#else
      if (!equal_to_1) {
        this->buf.push_arg(item);
      }
      std::string sig_for_idx = fmt::format("{}{}", dtype, equal_to_1 ? ":1" : "");
      signature.push_back(sig_for_idx);
#endif
    } else {
      handle_non_constexpr(item);
    }
  }

  template <typename T>
  void handle_non_constexpr(const T& item) {
    this->buf.push_arg(item);
    const char* dtype = narrow_type_name(item);
    signature.push_back(dtype);
  }

  void append_global_scratch() {
    void* global_scratch = nullptr;
    this->buf.push_arg(global_scratch);
  }
};

template <BackendPolicy Backend>
class TritonJITFunctionImpl {
 private:
  std::string file_path_;
  std::string function_name_;
  StaticSignature static_sig_;

  /// Cached compiled kernels (keyed by signature)
  mutable std::unordered_map<std::string, TritonKernelImpl<Backend>> overloads_;

  /// Global registry of all TritonJITFunctionImpl instances
  static std::unordered_map<std::string, std::unique_ptr<TritonJITFunctionImpl<Backend>>> functions_;

  /// Per-backend kernel file resolution: in KUNLUNXIN builds, prefer a
  /// device-tuned "<base>_kunlunxin.py" variant when it exists next to the
  /// generic kernel, leaving other backends on the shared source file.
  static std::string resolve_kernel_path(std::string_view path) {
#ifdef BACKEND_KUNLUNXIN
    std::string s(path);
    if (s.size() > 3 && s.compare(s.size() - 3, 3, ".py") == 0) {
      std::string alt = s.substr(0, s.size() - 3) + "_kunlunxin.py";
      if (std::filesystem::exists(alt)) {
        return alt;
      }
    }
#else
    (void)path;
#endif
    return std::string(path);
  }

 public:
  static TritonJITFunctionImpl& get_instance(std::string_view path, std::string_view name) {
    std::string key = fmt::format("{}:{}", path, name);

    auto it = functions_.find(key);
    if (it == functions_.end()) {
      // Use new instead of make_unique since constructor is private
      auto ptr = std::unique_ptr<TritonJITFunctionImpl>(
          new TritonJITFunctionImpl(resolve_kernel_path(path), name));
      functions_.emplace(key, std::move(ptr));
    }

    return *functions_.at(key);
  }

  // Delete copy constructor and assignment
  TritonJITFunctionImpl(const TritonJITFunctionImpl&) = delete;
  TritonJITFunctionImpl& operator=(const TritonJITFunctionImpl&) = delete;

  // Default move constructor and assignment
  TritonJITFunctionImpl(TritonJITFunctionImpl&&) = default;
  TritonJITFunctionImpl& operator=(TritonJITFunctionImpl&&) = default;

  const StaticSignature& get_static_sig() const {
    return this->static_sig_;
  }

  // Backward-compatible overload: plain (num_warps, num_stages) are wrapped into a
  // CompileOptions carrying no extra switches, then forwarded to the primary overload.
  // Kept so existing call sites (and operator dispatch code) compile unchanged.
  template <typename... Args>
  void operator()(typename Backend::StreamType stream,
                  unsigned int grid_x,
                  unsigned int grid_y,
                  unsigned int grid_z,
                  unsigned int num_warps,
                  unsigned int num_stages,
                  Args... args) const {
    CompileOptions copts;
    copts.num_warps = static_cast<int>(num_warps);
    copts.num_stages = static_cast<int>(num_stages);
    (*this)(stream, grid_x, grid_y, grid_z, copts, args...);
  }

  // Primary overload: CompileOptions carries num_warps/num_stages plus any extra
  // backend compiler switches (e.g. {"opt_level","O2"}). All of it feeds the cache key.
  template <typename... Args>
  void operator()(typename Backend::StreamType stream,
                  unsigned int grid_x,
                  unsigned int grid_y,
                  unsigned int grid_z,
                  const CompileOptions& copts,
                  Args... args) const {
    const int num_args = this->static_sig_.num_args;

    // Storage for argument processing using ParameterBuffer
    ParameterBuffer buffer;
    buffer.reserve(num_args);  // this is a coarse estimation of parameter size
    c10::SmallVector<std::string> signature;
    signature.reserve(num_args);

    // Process arguments
    ArgHandle handler = {this->static_sig_, buffer, signature, 0};
    (handler.handle_arg(args), ...);

#if !defined(BACKEND_NPU) && !defined(BACKEND_KUNLUNXIN)
    // global scratch: introduced in triton 3.3
    // NPU/Kunlunxin backends do not use global scratch
    handler.append_global_scratch();
    handler.append_global_scratch();
#endif
    std::string full_signature = join_sig(signature);

    // Backend-specific context setup
    Backend::ensure_context();
    int device_index = Backend::get_device_index();

    // Get or compile kernel
    const TritonKernelImpl<Backend>& kernel =
        this->get_kernel(full_signature, copts, device_index);

    // Launch kernel with signature (for NPU backend to parse argument types)
    c10::SmallVector<void*> ptrs = buffer.get_ptrs();
    kernel.launch_with_signature(grid_x,
                                 grid_y,
                                 grid_z,
                                 copts.num_warps,
                                 stream,
                                 ptrs.data(),
                                 full_signature,
                                 ptrs.size());
  }

  void launch_with_raw_args(typename Backend::StreamType stream,
                            unsigned int grid_x,
                            unsigned int grid_y,
                            unsigned int grid_z,
                            unsigned int num_warps,
                            unsigned int num_stages,
                            std::string full_signature,
                            void** args,
                            size_t num_args = 0) const {
    Backend::ensure_context();
    int device_index = Backend::get_device_index();

    CompileOptions copts;
    copts.num_warps = static_cast<int>(num_warps);
    copts.num_stages = static_cast<int>(num_stages);
    const TritonKernelImpl<Backend>& kernel =
        this->get_kernel(full_signature, copts, device_index);

    kernel.launch_with_signature(grid_x, grid_y, grid_z, num_warps, stream, args, full_signature, num_args);
  }

 private:
  TritonJITFunctionImpl(std::string_view path, std::string_view name);
  const TritonKernelImpl<Backend>& get_kernel(std::string_view signature,
                                              const CompileOptions& opts,
                                              int device_index) const;
};

// Initialize static member
template <BackendPolicy Backend>
std::unordered_map<std::string, std::unique_ptr<TritonJITFunctionImpl<Backend>>>
    TritonJITFunctionImpl<Backend>::functions_;

// Compile-time checks
template <BackendPolicy Backend>
static inline constexpr bool is_triton_jit_function_move_constructible_v =
    std::is_move_constructible_v<TritonJITFunctionImpl<Backend>>;

}  // namespace triton_jit
