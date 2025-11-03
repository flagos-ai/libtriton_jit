/**
 * @file triton_jit_function.h
 * @brief TritonJITFunction public interface (Policy-Based Design v2.0)
 *
 * This file provides the public interface for TritonJITFunction.
 * The actual implementation is in triton_jit_function_impl.h as a template class.
 *
 * For user code, use the type alias TritonJITFunction from backend_config.h.
 *
 * @version 2.0.0
 * @date 2025-11-03
 */

#pragma once

#include "triton_jit/backend_config.h"
#include "triton_jit/triton_jit_function_impl.h"

// The TritonJITFunction type alias is defined in backend_config.h
// User code can use it directly without template parameters:
//
//   auto& func = triton_jit::TritonJITFunction::get_instance("kernel.py", "my_kernel");
//   func(stream, grid_x, grid_y, grid_z, num_warps, num_stages, arg1, arg2, ...);
//
// The backend is selected at compile time via CMake options.

namespace triton_jit {

// Re-export ArgType and StaticSignature for user code
using ArgType = ArgType;
using StaticSignature = StaticSignature;

// For backward compatibility, ensure TritonJITFunction is move constructible
static_assert(std::is_move_constructible_v<TritonJITFunction>,
              "TritonJITFunction must be move constructible");

} // namespace triton_jit
