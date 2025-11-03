/**
 * @file triton_kernel.h
 * @brief TritonKernel public interface (Policy-Based Design v2.0)
 *
 * This file provides the public interface for TritonKernel.
 * The actual implementation is in triton_kernel_impl.h as a template class.
 *
 * For user code, use the type alias TritonKernel from backend_config.h.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include "triton_jit/backend_config.h"
#include "triton_jit/triton_kernel_impl.h"

// The TritonKernel type alias is defined in backend_config.h
// User code can use it directly without template parameters:
//
//   triton_jit::TritonKernel kernel(...);
//   kernel.launch(...);
//
// The backend is selected at compile time via CMake options.

namespace triton_jit {

// For backward compatibility, ensure TritonKernel is move constructible
static_assert(std::is_move_constructible_v<TritonKernel>,
              "TritonKernel must be move constructible");

} // namespace triton_jit
