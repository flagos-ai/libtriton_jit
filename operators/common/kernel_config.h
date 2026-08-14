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

#include "triton_jit/backend_config.h"

namespace triton_jit::ops {

// ---- Matmul config (mm, addmm) ----
struct MatmulConfig {
  int64_t BLOCK_M;
  int64_t BLOCK_N;
  int64_t BLOCK_K;
  int64_t GROUP_M;
  int num_warps;
  int num_stages;
};

inline constexpr MatmulConfig default_matmul_config() {
#if defined(BACKEND_NPU)
  return {32, 32, 32, 4, 1, 1};
#else
  return {64, 64, 32, 8, 4, 2};
#endif
}

// ---- Reduce-sum config ----
struct ReduceSumConfig {
  int64_t BLOCK_M;
  int64_t BLOCK_N;
  int num_warps;
  int num_stages;
};

inline constexpr ReduceSumConfig default_reduce_sum_config() {
#if defined(BACKEND_NPU)
  return {4, 256, 1, 1};
#else
  return {4, 512, 8, 2};
#endif
}

// ---- Softmax config ----
struct SoftmaxConfig {
  int64_t max_tile_n;
  int num_warps;
  int num_stages;
};

inline constexpr SoftmaxConfig default_softmax_config() {
#if defined(BACKEND_NPU)
  return {2048, 1, 1};
#else
  return {4096, 4, 1};
#endif
}

// ---- Norm config (fused_add_rms_norm) ----
struct NormConfig {
  int64_t max_block_size;  // 0 = no limit
  int num_warps;
  int num_stages;
};

inline constexpr NormConfig default_norm_config() {
#if defined(BACKEND_NPU)
  return {1024, 1, 1};
#elif defined(BACKEND_MLU)
  return {2048, 1, 1};
#else
  return {0, 4, 1};
#endif
}

// ---- Rotary embedding config ----
struct RotaryConfig {
  int64_t BLOCK_N;
  int64_t BLOCK_H;
  int num_warps;
  int num_stages;
};

inline constexpr RotaryConfig default_rotary_config() {
#if defined(BACKEND_NPU)
  return {4, 4, 1, 1};
#elif defined(BACKEND_MLU)
  return {32, 4, 1, 1};
#else
  return {8, 4, 4, 1};
#endif
}

// ---- Pointwise config ----
struct PointwiseConfig {
  int64_t tile_size;
  int num_warps;
  int num_stages;
};

inline constexpr PointwiseConfig default_pointwise_config() {
#if defined(BACKEND_MLU)
  return {4096, 4, 1};
#else
  return {1024, 8, 1};
#endif
}

// ---- Basic config ----
struct BasicConfig {
  int num_warps;
  int num_stages;
};

inline constexpr BasicConfig default_basic_config() {
#if defined(BACKEND_MLU)
  return {1, 1};
#else
  return {4, 1};
#endif
}

}  // namespace triton_jit::ops
