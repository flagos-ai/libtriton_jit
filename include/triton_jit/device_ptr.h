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
#include <type_traits>

// Device pointer arguments for callers that do not hold an at::Tensor.
//
// The typed launch path derives a pointer argument's signature token from the
// tensor it is given: the address supplies the alignment hint and the tensor
// supplies the element dtype. A C or C++ caller holding a raw device address
// (CUdeviceptr, CNaddr, void*) has no tensor to pass, and the address alone
// does not carry an element type.
//
// TritonDevicePtr closes that gap by pairing the address with an explicit
// dtype. It is accepted wherever an at::Tensor is, and yields the same
// signature token and the same alignment specialization as the equivalent
// tensor argument. HCU additionally specializes a tensor on the extent of its
// storage; a bare pointer carries no extent, so there it matches the tensor
// only when wrapped with with_pointer_range. Note that a tensor's extent is
// measured from the start of its storage, a bounded pointer's from the pointer
// itself: for an offset view, pass the storage extent to get the very same
// specialization (and compiled kernel) as the tensor.
//
// This header intentionally has no torch dependency, so tooling that only
// needs the dtype vocabulary can include it on its own.

namespace triton_jit {

/// Element types a device pointer may address, spelled as in Triton's
/// signature grammar (see to_triton_typename). The fp8 entries have no C++
/// scalar counterpart and are therefore only reachable through the
/// runtime-dtype overload of device_ptr.
enum class TritonDType : int8_t {
  kI1 = 0,
  kI8,
  kI16,
  kI32,
  kI64,
  kU8,
  kU16,
  kU32,
  kU64,
  kFp16,
  kBf16,
  kFp32,
  kFp64,
  kFp8E4NV,
  kFp8E5,
};

/**
 * @brief Spell a dtype tag as Triton spells it in a kernel signature.
 */
constexpr const char* to_triton_typename(TritonDType t) {
  switch (t) {
    case TritonDType::kI1:
      return "i1";
    case TritonDType::kI8:
      return "i8";
    case TritonDType::kI16:
      return "i16";
    case TritonDType::kI32:
      return "i32";
    case TritonDType::kI64:
      return "i64";
    case TritonDType::kU8:
      return "u8";
    case TritonDType::kU16:
      return "u16";
    case TritonDType::kU32:
      return "u32";
    case TritonDType::kU64:
      return "u64";
    case TritonDType::kFp16:
      return "fp16";
    case TritonDType::kBf16:
      return "bf16";
    case TritonDType::kFp32:
      return "fp32";
    case TritonDType::kFp64:
      return "fp64";
    case TritonDType::kFp8E4NV:
      return "fp8e4nv";
    case TritonDType::kFp8E5:
      return "fp8e5";
  }
  return "<unsupported_dtype>";
}

/// A device address together with the element type stored at it. Trivially
/// copyable, and packed into the runtime argument buffer as the eight bytes of
/// `value`, which is the ABI a void* or CUdeviceptr argument already uses.
struct TritonDevicePtr {
  std::uintptr_t value;
  TritonDType dtype;
};

/// Maps a C++ element type to its dtype tag, so that device_ptr can deduce
/// the tag from a typed pointer.
///
/// Types without a mapping are rejected at compile time rather than guessed
/// at. A library with its own storage types registers them by specializing
/// this trait, after which its pointers deduce like any built-in type:
///
///     template <>
///     struct triton_jit::triton_dtype_of<MyHalf> {
///       static constexpr TritonDType value = TritonDType::kFp16;
///     };
template <typename T, typename Enable = void>
struct triton_dtype_of {
  static_assert(sizeof(T) == 0,
                "no TritonDType mapping for this element type; "
                "use device_ptr(p, TritonDType) or specialize triton_dtype_of");
};

template <typename T>
struct triton_dtype_of<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
 private:
  static constexpr TritonDType compute() {
    if constexpr (std::is_same_v<T, bool>) {
      return TritonDType::kI1;
    } else if constexpr (std::is_same_v<T, float>) {
      return TritonDType::kFp32;
    } else if constexpr (std::is_same_v<T, double>) {
      return TritonDType::kFp64;
    } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
      // Dispatch on width rather than on type identity: long and long long
      // are distinct types of the same width, and both must map.
      static_assert(sizeof(T) <= 8, "integer element type wider than 64 bits");
      return sizeof(T) == 1   ? TritonDType::kI8
             : sizeof(T) == 2 ? TritonDType::kI16
             : sizeof(T) == 4 ? TritonDType::kI32
                              : TritonDType::kI64;
    } else if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
      static_assert(sizeof(T) <= 8, "integer element type wider than 64 bits");
      return sizeof(T) == 1   ? TritonDType::kU8
             : sizeof(T) == 2 ? TritonDType::kU16
             : sizeof(T) == 4 ? TritonDType::kU32
                              : TritonDType::kU64;
    } else {
      static_assert(sizeof(T) == 0,
                    "no TritonDType mapping for this arithmetic type "
                    "(long double is not supported)");
    }
  }

 public:
  static constexpr TritonDType value = compute();
};

// device_ptr overloads
// -------------------------------------------------------------------------
// Four ways to name a device pointer, covering how callers come by one: a
// typed pointer, a typed pointer read as another element type, an integer
// device handle, and an element type only known at run time. Overload
// resolution separates them by argument form, so all four share one name.

/**
 * @brief Wrap a typed device pointer, deducing its element type.
 *
 * @code
 * triton_jit::device_ptr(arguments.A);   // float* -> *fp32
 * @endcode
 */
template <typename T, typename = decltype(triton_dtype_of<std::remove_cv_t<T>>::value)>
inline TritonDevicePtr device_ptr(T* p) {
  return TritonDevicePtr {reinterpret_cast<std::uintptr_t>(p), triton_dtype_of<std::remove_cv_t<T>>::value};
}

/**
 * @brief Wrap a device pointer under a different element type than it is
 *        declared with, for buffers the kernel reads through another view --
 *        an interleaved complex buffer addressed as pairs of fp32, say.
 *
 * @code
 * triton_jit::device_ptr<float>(arguments.x);   // Complex* -> *fp32
 * @endcode
 */
template <typename Elem,
          typename T,
          typename = std::enable_if_t<!std::is_same_v<std::remove_cv_t<Elem>, std::remove_cv_t<T>>>>
inline TritonDevicePtr device_ptr(T* p) {
  return TritonDevicePtr {reinterpret_cast<std::uintptr_t>(p),
                          triton_dtype_of<std::remove_cv_t<Elem>>::value};
}

/**
 * @brief Wrap an integer device handle, such as a CUdeviceptr. The element
 *        type must be given: an integer address carries none.
 *
 * @code
 * triton_jit::device_ptr<float>(cu_device_ptr);
 * @endcode
 */
template <typename Elem>
inline TritonDevicePtr device_ptr(std::uintptr_t raw) {
  return TritonDevicePtr {raw, triton_dtype_of<std::remove_cv_t<Elem>>::value};
}

/**
 * @brief Wrap a device pointer whose element type is chosen at run time, as
 *        when a C API takes the type as a parameter. This is also the only
 *        way to name the fp8 types, which have no C++ counterpart.
 *
 * @code
 * triton_jit::device_ptr(arguments.A, to_dtype(arguments.Atype));
 * @endcode
 */
inline TritonDevicePtr device_ptr(const void* p, TritonDType dtype) {
  return TritonDevicePtr {reinterpret_cast<std::uintptr_t>(p), dtype};
}

/// @overload
inline TritonDevicePtr device_ptr(std::uintptr_t raw, TritonDType dtype) {
  return TritonDevicePtr {raw, dtype};
}

/// A device pointer together with the number of bytes the kernel may address
/// from it, i.e. [pointer.value, pointer.value + size_bytes). The extent is
/// compile-time metadata only: the runtime ABI is that of the bare pointer.
///
/// An at::Tensor carries this extent in its storage; a raw pointer does not,
/// so a caller that knows it passes it explicitly. On HCU the backend uses the
/// extent like the native Python runtime does for a tensor: an extent of at
/// most INT32_MAX bytes lets the kernel use 32-bit buffer addressing
/// (tt.pointer_range = 32). Other backends ignore it.
///
/// The extent is a promise: the kernel must not touch bytes outside it.
/// size_bytes == 0 means "unknown" and never narrows the range.
struct TritonBoundedDevicePtr {
  TritonDevicePtr pointer;
  std::uint64_t size_bytes;
};

/**
 * @brief Attach the accessible byte extent to a device pointer.
 *
 * @code
 * // A is an n x n column-major matrix with leading dimension lda.
 * // Widen before multiplying: lda * n in int overflows for large matrices, and an
 * // extent that is too small makes the kernel silently miss data.
 * triton_jit::with_pointer_range(triton_jit::device_ptr(A),
 *                                static_cast<std::uint64_t>(lda) * n * sizeof(float));
 * @endcode
 */
inline TritonBoundedDevicePtr with_pointer_range(TritonDevicePtr pointer, std::uint64_t size_bytes) {
  return TritonBoundedDevicePtr {pointer, size_bytes};
}

}  // namespace triton_jit
