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
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "c10/util/Logging.h"  // use torch's logging
#include "torch/torch.h"

#ifdef BACKEND_NPU
#include "acl/acl.h"
#elif defined(BACKEND_MUSA)
#include <musa.h>
#elif defined(BACKEND_MACA)
#include <mcr/mc_runtime.h>
#elif defined(BACKEND_MLU)
#include <cn_api.h>
#elif defined(BACKEND_GCU)
#elif defined(BACKEND_HCU)
#include <hip/hip_runtime.h>
#else
#include "cuda.h"
#endif

namespace triton_jit {

// Compile-time options threaded from the C++ launch call down to triton.compile.
// num_warps/num_stages are always part of the compiled artifact; `extra` carries
// backend-specific compiler switches (e.g. MLU opt_level) as string key/values that
// the Python bridge coerces to the right type. Every field here participates in the
// kernel cache key -- see get_kernel in triton_jit_function.cpp.
struct CompileOptions {
  int num_warps = 4;
  int num_stages = 3;
  std::map<std::string, std::string> extra;
};

namespace detail {

inline void validate_compile_options(const CompileOptions& opts) {
  for (const char* reserved : {"num_warps", "num_stages"}) {
    if (opts.extra.find(reserved) != opts.extra.end()) {
      throw std::invalid_argument(std::string("CompileOptions.extra must not override reserved option '") +
                                  reserved + "'");
    }
  }
}

inline void append_cache_key_field(std::string& key, std::string_view value) {
  key += std::to_string(value.size());
  key.push_back(':');
  key.append(value.begin(), value.end());
}

inline std::string make_kernel_cache_key(std::string_view signature,
                                         int device_index,
                                         const CompileOptions& opts) {
  validate_compile_options(opts);

  std::string key;
  key.reserve(signature.size() + opts.extra.size() * 24 + 64);
  key += "sig=";
  append_cache_key_field(key, signature);
  key += ";dev=";
  append_cache_key_field(key, std::to_string(device_index));
  key += ";nw=";
  append_cache_key_field(key, std::to_string(opts.num_warps));
  key += ";ns=";
  append_cache_key_field(key, std::to_string(opts.num_stages));
  key += ";extra=";
  append_cache_key_field(key, std::to_string(opts.extra.size()));
  for (const auto& [name, value] : opts.extra) {
    key += ";name=";
    append_cache_key_field(key, name);
    key += ";value=";
    append_cache_key_field(key, value);
  }
  return key;
}

}  // namespace detail

constexpr const char* to_triton_typename(c10::ScalarType t) {
  switch (t) {
    case c10::ScalarType::Float:
      return "fp32";
    case c10::ScalarType::Double:
      return "fp64";
    case c10::ScalarType::Half:
      return "fp16";
    case c10::ScalarType::BFloat16:
      return "bf16";
    case c10::ScalarType::Int:
      return "i32";
    case c10::ScalarType::Long:
      return "i64";
    case c10::ScalarType::Short:
      return "i16";
    case c10::ScalarType::UInt32:
      return "u32";
    case c10::ScalarType::UInt64:
      return "u64";
    case c10::ScalarType::UInt16:
      return "u16";
    case c10::ScalarType::Char:
      return "i8";
    case c10::ScalarType::Float8_e4m3fn:
      return "fp8e4nv";
    case c10::ScalarType::Byte:
      return "u8";
    case c10::ScalarType::Bool:
      return "i1";
    default:
      throw std::runtime_error("<unsupported_type>");
      return "<unsupported_type>";
  }
}

template <typename T>
constexpr const char* spec(T v) {
  return v % 16 == 0 ? ":16" : v == 1 ? ":1" : "";
}

template <typename T, typename = void>
struct has_data_ptr : std::false_type {};

template <typename T>
struct has_data_ptr<
    T,
    std::enable_if_t<std::conjunction_v<
        std::is_same<decltype(std::declval<std::remove_reference_t<T>>().data_ptr()), void*>,
        std::is_same<decltype(std::declval<std::remove_reference_t<T>>().scalar_type()), c10::ScalarType>>>>
    : std::true_type {};

template <typename T>
struct is_optional_helper : public std::false_type {};

template <typename T>
struct is_optional_helper<std::optional<T>> : public std::true_type {};

template <typename T>
struct is_optional : public is_optional_helper<std::remove_const_t<std::remove_reference_t<T>>> {};

template <typename T>
struct is_scalar_helper : public std::false_type {};

template <>
struct is_scalar_helper<c10::Scalar> : public std::true_type {};

template <typename T>
struct is_scalar : public is_scalar_helper<std::remove_const_t<std::remove_reference_t<T>>> {};

template <typename T>
struct triton_type_helper;

template <typename T, typename U>
struct is_same_ignore_cvref : public std::is_same<std::remove_reference_t<std::remove_cv_t<T>>,
                                                  std::remove_reference_t<std::remove_cv_t<U>>> {};

#define DEFINE_TRITON_TYPE(T, Name)           \
  template <>                                 \
  struct triton_type_helper<T> {              \
    static constexpr const char* name = Name; \
  }

DEFINE_TRITON_TYPE(bool, "i1");
DEFINE_TRITON_TYPE(int, "i32");
DEFINE_TRITON_TYPE(uint, "u32");
DEFINE_TRITON_TYPE(int64_t, "i64");
DEFINE_TRITON_TYPE(uint64_t, "u64");
DEFINE_TRITON_TYPE(float, "fp32");
DEFINE_TRITON_TYPE(double, "fp64");
DEFINE_TRITON_TYPE(std::nullptr_t, "*i8");
DEFINE_TRITON_TYPE(const char*, "constexpr");
DEFINE_TRITON_TYPE(std::string, "constexpr");

#undef DEFINE_TRITON_TYPE

template <typename T>
struct triton_type : triton_type_helper<std::remove_cv_t<std::remove_reference_t<T>>> {};

// Mimic Triton runtime's native_specialize_impl: narrow integer types by value range
template <typename T>
constexpr const char* narrow_type_name(const T& v) {
  if constexpr (std::is_integral_v<std::decay_t<T>>) {
    if constexpr (std::is_unsigned_v<std::decay_t<T>>) {
      return triton_type<T>::name;
    }
    if (v >= INT32_MIN && v <= INT32_MAX) {
      return "i32";
    }
    return "i64";
  }
  return triton_type<T>::name;
}

// path of python executable
std::filesystem::path get_script_dir();

#ifdef BACKEND_NPU
// ACL error checking function
inline void checkAclErrors(aclError code, const char* message = "") {
  if (code != ACL_ERROR_NONE) {
    const char* error_string = aclGetRecentErrMsg();
    if (!error_string) {
      error_string = "Unknown AscendCL error";
    }
    fprintf(stderr, "AscendCL API error = %04d. Detail: <%s>. Message: %s\n", code, error_string, message);
    throw std::runtime_error(std::string(message) + ": " + error_string);
  }
}
#elif defined(BACKEND_MUSA)
#define checkMusaErrors(err) __checkMusaErrors(err, __FILE__, __LINE__)

// Error handling function using exceptions instead of exit()
inline void __checkMusaErrors(MUresult code, const char* file, const int line) {
  if (code != MUSA_SUCCESS) {
    const char* error_string;
    muGetErrorString(code, &error_string);
    fprintf(stderr,
            "MUSA Driver API error = %04d from file <%s>, line %i. Detail: <%s>\n",
            code,
            file,
            line,
            error_string);
    throw std::runtime_error(error_string);
  }
}
#elif defined(BACKEND_MACA)
#define checkMacaErrors(err) __checkMacaErrors(err, __FILE__, __LINE__)

inline void __checkMacaErrors(mcError_t code, const char* file, const int line) {
  if (code != mcSuccess) {
    const char* error_string = mcGetErrorString(code);
    fprintf(stderr,
            "MACA Runtime API error = %04d from file <%s>, line %i. Detail: <%s>\n",
            code,
            file,
            line,
            error_string);
    throw std::runtime_error(error_string ? error_string : "Unknown MACA error");
  }
}
#elif defined(BACKEND_MLU)
#define checkMluErrors(err) __checkMluErrors(err, __FILE__, __LINE__)

// Error handling function using exceptions instead of exit()
inline void __checkMluErrors(CNresult code, const char* file, const int line) {
  if (code != CN_SUCCESS){
    const char* error_string;
    cnGetErrorString(code, &error_string);
    fprintf(stderr, "MLU Driver API error = %04d from file <%s>, line %i. Detail: <%s>\n",
        code,
        file,
        line,
        error_string);
    throw std::runtime_error(error_string);
  }
}
#elif defined(BACKEND_GCU)
// GCU error handling is done inline in gcu_backend.h
#elif defined(BACKEND_HCU)
#define checkHcuErrors(err) __checkHcuErrors(err, __FILE__, __LINE__)

inline void __checkHcuErrors(hipError_t code, const char* file, const int line) {
  if (code != hipSuccess) {
    const char* error_string = hipGetErrorString(code);
    fprintf(stderr,
            "HCU Runtime API error = %04d from file <%s>, line %i. Detail: <%s>\n",
            static_cast<int>(code),
            file,
            line,
            error_string);
    throw std::runtime_error(error_string);
  }
}
#else
void ensure_cuda_context();

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// Error handling function using exceptions instead of exit()
inline void __checkCudaErrors(CUresult code, const char* file, const int line) {
  if (code != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(code, &error_string);
    fprintf(stderr,
            "CUDA Driver API error = %04d from file <%s>, line %i. Detail: <%s>\n",
            code,
            file,
            line,
            error_string);
    throw std::runtime_error(error_string);
  }
}
#endif

inline std::string join_sig(const c10::SmallVector<std::string>& signature) {
  std::stringstream ss;
  for (size_t i = 0; i < signature.size(); i++) {
    if (i == 0) {
      ss << signature[i];
    } else {
      ss << "," << signature[i];
    }
  }
  return ss.str();
}

}  // namespace triton_jit
