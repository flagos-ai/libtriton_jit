#pragma once

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>

#include "c10/util/Logging.h"  // use torch's logging
#include "torch/torch.h"

#ifdef BACKEND_NPU
#include "acl/acl.h"
#elif defined(BACKEND_MUSA)
#include <musa.h>
#elif defined(BACKEND_MLU)
#include <cn_api.h>
#elif defined(BACKEND_HCU)
#include <hip/hip_runtime.h>
#else
#include "cuda.h"
#endif

namespace triton_jit {

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

#undef DEFINE_TRITON_TYPE

template <typename T>
struct triton_type : triton_type_helper<std::remove_cv_t<std::remove_reference_t<T>>> {};

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
#elif defined(BACKEND_HCU)
#define checkHcuErrors(err) __checkHcuErrors(err, __FILE__, __LINE__)

// Error handling function for HCU runtime API using exceptions
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

}  // namespace triton_jit
