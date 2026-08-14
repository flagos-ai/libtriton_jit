<!--
 Copyright 2026 FlagOS Contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triton JIT is a C++ runtime for calling Triton JIT functions from C++ code. It reduces Python overhead by reimplementing the JIT runtime in C++ while reusing the Triton compiler. The core class `TritonJITFunction` manages kernel compilation, caching, and execution across multiple hardware backends.

## Build Commands

```bash
# Install dependencies (use virtualenv, avoid Anaconda)
pip install "torch>=2.5" "triton>=3.1.0,<3.4.0" "cmake" "ninja" "packaging" "pybind11" "numpy"

# Configure (select backend: CUDA, MUSA, NPU, or IX)
cmake -S . -B build/ -DPython_ROOT="$(which python)/../.." -DBACKEND=CUDA

# Build
cmake --build build/ --parallel

# Run all tests
python tests/run_all_tests.py --backend CUDA --build-dir build

# Run specific test (必须从算子目录运行，见下方说明)
(cd build/operators/<category>/<op> && ./test_<op>)
# Example: (cd build/operators/matmul/addmm && ./test_addmm)

# Filter tests by category or operator
python tests/run_all_tests.py --category pointwise --backend CUDA --build-dir build
python tests/run_all_tests.py --operator add --backend CUDA --build-dir build
```

### 测试运行目录要求

**重要**：单独运行算子测试时，**必须从算子的构建目录执行**，否则会报 `FileNotFoundError: xxx.py`。

原因：`TritonJITFunction::get_instance("kernel.py", ...)` 使用相对路径查找 Python kernel 文件，CMake 会将 `.py` 文件复制到对应的构建目录。

```bash
# 错误方式（从项目根目录运行）
./build/operators/matmul/addmm/test_addmm
# FileNotFoundError: '/path/to/triton_jit/addmm.py'

# 正确方式（从算子构建目录运行）
(cd build/operators/matmul/addmm && ./test_addmm)
# 或
cd build/operators/matmul/addmm && ./test_addmm
```

## Architecture

**Backend Policy Design**: Uses C++20 concepts for compile-time backend polymorphism. Each backend implements `StreamType`, `ContextType`, `KernelHandle`, `WARP_SIZE`, and methods for kernel loading/launching.

**Supported Backends**:
- **CUDA**: NVIDIA GPUs, warp size 32
- **MUSA**: Moore Threads GPUs, warp size 32, maps to "mtgpu" in Triton
- **NPU**: Ascend/Huawei, uses ACL API, limited operator support
- **IX**: Tianshu GPUs, warp size 64

**Key Components**:
- `TritonJITFunction`: Singleton registry, manages JIT compilation and kernel caching
- `TritonKernel`: Lazy-loads compiled kernels, handles backend-specific launch
- `ParameterBuffer`: Serializes kernel arguments with proper alignment
- Backend headers in `include/triton_jit/backends/`

**Operator Pattern**: Each operator has:
1. Python kernel file (e.g., `add.py`) with `@triton.jit` decorated function
2. C++ operator file (e.g., `add_op.cpp`) that calls `TritonJITFunction`
3. Test executable (e.g., `test_add.cpp`)
4. Registration via `TORCH_LIBRARY` macro

## Directory Structure

- `include/triton_jit/`: Public headers, backend implementations
- `src/`: Core runtime (`triton_jit_function_impl.cpp`)
- `operators/`: Operator implementations by category
  - `pointwise/`: add, fill, zeros, contiguous, exponential_
  - `reduce/`: sum, max, argmax, topk (argmax/topk 需要索引返回，NPU 不支持)
  - `matmul/`: mm, bmm, addmm
  - `normalization/`: softmax, rms_norm, fused_add_rms_norm (not on NPU)
  - `fusion/`: apply_rotary_pos_emb, rwkv_* (not on NPU)
  - `index/`: cat, embedding, nonzero, reshape_and_cache_flash
  - `common/`: Test framework
- `scripts/`: `standalone_compile.py` (kernel compilation), `gen_ssig.py` (signature generation)
- `cmake/`: Backend CMake modules (`BackendCUDA.cmake`, `BackendNPU.cmake`, etc.)

## Backend Constraints

**NPU (Ascend)**: 基础 reduce 操作（sum/max/min）已在 triton-ascend 3.2.0 中支持，但 argmax/argmin 仍不支持。Normalization 和 fusion 算子需要进一步验证。

**Device Types**: NPU and MUSA use `at::DeviceType::PrivateUse1`; CUDA/IX use `at::DeviceType::CUDA`.

### NPU linalg.reduce 限制（部分解除）

> **更新 (2026-02-04)**：`triton-ascend 3.2.0` 已支持基础 reduce 操作，但带索引返回的操作仍不支持。

**Triton 操作支持情况**（triton-ascend 3.2.0 + CANN 8.5.0）：

| Triton API | NPU 支持 | 备注 |
|------------|---------|------|
| `tl.sum(vals, axis)` | ✅ | 新版本已支持 |
| `tl.max(vals, axis)` | ✅ | 新版本已支持 |
| `tl.min(vals, axis)` | ✅ | 新版本已支持 |
| `tl.max(..., return_indices=True)` | ❌ | linalg.reduce 失败 |
| `tl.argmax()` | ❌ | linalg.reduce 失败 |
| `tl.argmin()` | ❌ | linalg.reduce 失败 |

**结论**：基础的 reduce 操作（sum/max/min）现在可用，但需要索引返回的操作（argmax/argmin）仍被 BiShengHIR 拒绝。

**验证脚本**：
```python
# 测试 tl.max（预期成功）
import torch, torch_npu, triton, triton.language as tl

@triton.jit
def test_max(inp, out, N: tl.constexpr):
    vals = tl.load(inp + tl.arange(0, N))
    tl.store(out, tl.max(vals, axis=0))

inp = torch.randn(128, device="npu:0")
out = torch.empty(1, device="npu:0")
test_max[(1,)](inp, out, 128)
print(f"tl.max: {out.item():.4f}, torch.max: {inp.max().item():.4f}")
```

**仍然失败的操作**（带索引返回）：
```
MLIRCompilationError: failed to legalize operation 'linalg.reduce' that was explicitly marked illegal
```

### NPU Block Size 配置指南

NPU 的 Unified Buffer (UB) 容量约 **192KB**，matmul 类算子必须使用较小的 block 配置，否则 BiShengHIR 编译时会报 `ub overflow` 错误。

**错误示例**：
```
ub overflow, requires 2097152 bits while 1572864 bits available!
(possible reason: tiling basic block is too large...)
```

**NPU 推荐配置**（参考 `mm_op.cpp`）：
```cpp
#if defined(BACKEND_NPU)
    constexpr int64_t BLOCK_M = 32;
    constexpr int64_t BLOCK_N = 32;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 1;
    constexpr int num_stages = 1;  // 禁用 double-buffering
#else
    // CUDA/MUSA: 可使用更大 block
    constexpr int64_t BLOCK_M = 64;
    constexpr int64_t BLOCK_N = 64;
    constexpr int64_t BLOCK_K = 32;
    constexpr int num_warps = 4;
    constexpr int num_stages = 2;
#endif
```

**内存估算公式**（fp32）：
- A tile: `BLOCK_M × BLOCK_K × 4` bytes
- B tile: `BLOCK_K × BLOCK_N × 4` bytes
- Acc: `BLOCK_M × BLOCK_N × 4` bytes
- 总计需乘以 `num_stages`（multi-buffering）

**经验法则**：NPU 上 `BLOCK_M × BLOCK_N ≤ 32×32`，`num_stages=1`。

## Coding Conventions

- C++20 standard, 2-space indentation, `snake_case` for functions/variables
- Python: 4-space indentation, no relative imports (scripts must be directly importable)
- Commit format: `<scope>: <action>` or `[TAG] <action>` (e.g., `operators: add topk test`, `[FIX] stream handling`)

## Logging

Enable runtime logging: `TORCH_CPP_LOG_LEVEL=INFO`

---

## Debugging Docs

排障复盘文档入口：[docs/debugging_index.md](docs/debugging_index.md)

记录开发过程中遇到的典型问题、排查过程、根因分析与修复方案，便于知识沉淀与后续参考。

---

## Current Development Status (2026-02-04)

**状态**：NPU 后端 matmul 类算子已修复，normalization 算子（fused_add_rms_norm）已适配，基础 reduce 操作（sum/max/min）已在新版 triton-ascend 中可用

**已解决问题**：
1. **workspace 未分配**（aicore exception）→ 已在 `npu_backend.h` 添加 `workspace_size` 解析与 `rtMalloc` 分配
2. **global scratch pointer 多余参数**（结果错误）→ 已在 `triton_jit_function_impl.h` 为 NPU 禁用 `append_global_scratch()`
3. **addmm UB 溢出**（编译失败）→ 已在 `addmm_op.cpp` 添加 NPU 专用 block 配置（32×32）
4. **argmax 函数名/参数不匹配**（运行时错误）→ 已修复 `argmax_op.cpp` 中的函数名和参数
5. **基础 reduce 支持**（triton-ascend 3.2.0 更新）→ `tl.sum/max/min` 现已可用
6. **fused_add_rms_norm 参数不匹配**（类型错误）→ 已修复 C++/Python 参数对齐，移除多余 output tensor
7. **fused_add_rms_norm UB 溢出**（benchmark 失败）→ 已限制 NPU BLOCK_SIZE ≤ 1024

**NPU 后端限制（仍存在）**：
- `tl.argmax()` / `tl.argmin()` / `tl.max(..., return_indices=True)` 仍被 BiShengHIR 拒绝
- 详见：[NPU linalg.reduce 限制](#npu-linalgreduce-限制部分解除)

**验证命令**：
```bash
# matmul 类算子（预期通过）
(cd build/operators/matmul/addmm && ./test_addmm)

# fused_add_rms_norm 算子（预期通过）
(cd build/operators/normalization/fused_add_rms_norm && ./test_fused_add_rms_norm)

# 基础 reduce 算子（现已可用，需重新实现测试）
# tl.sum, tl.max, tl.min 可直接使用

# argmax 算子（预期失败于 linalg.reduce）
(cd build/operators/reduce/argmax && ./test_argmax)
```

**详细排障记录**：见 [docs/debugging_index.md](docs/debugging_index.md)
