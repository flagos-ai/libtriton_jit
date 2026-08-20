# P800 三个算子编译失败分析

## 1. 背景和共同原理

P800 通过 CUDA ABI 让 PyTorch tensor 继续使用 `device='cuda'`，但 Triton kernel 最终由 Kunlunxin XPU compiler/runtime 执行。因此 Python/Triton 语义可以保持不变，失败点出现在 XPU lowering：Triton pointer expression 会先变成带 encoding 的 tensor，再由 `tt.addptr`、OffsetAnalysis 和后续 SRAM pass 处理。NVIDIA backend 能接受的广播 layout，不代表 P800 lowering 能接受。

这三个修复都保持数学结果不变，目标是减少不规则的二维广播，把地址计算改写成 P800 能稳定分析的一维布局。

## 2. `max`

### 原始实现

原 kernel 每个 program 同时处理 `BLOCK_M` 行和 `BLOCK_N` 个 reduction 元素：

```python
m_offset = work_id * BLOCK_M + tl.arange(0, BLOCK_M)
n_offset = start_n + tl.arange(0, BLOCK_N)
offset = m_offset[:, None] * N + n_offset[None, :]
inp_vals = tl.load(inp + offset, mask=..., other=float('-inf'))
local_max = tl.max(inp_vals, 1)
```

### 报错

```text
'tt.addptr' op requires the same encoding for all operands
OutOfResources: uni_sram PassManager::run failed
```

### 原理

`m_offset[:, None]` 和 `n_offset[None, :]` 是两个不同来源的一维 tensor。广播后形成二维 pointer tensor，但两个 operand 的 Triton encoding/shape provenance 不一致。P800 的 `tt.addptr` lowering 要求参与指针加法的 operand 使用相同 encoding，无法像 NVIDIA backend 一样自由完成该广播；进入 `uni_sram` 后又无法分配这个二维布局。

### 修改

`operators/reduce/max/max.py` 改为一个 program 只负责一行，只有 N 维保留 vector：

```python
row = tl.program_id(0)
for start_n in range(0, N, BLOCK_N):
    n_offset = start_n + tl.arange(0, BLOCK_N)
    inp_vals = tl.load(inp + row * N + n_offset,
                       mask=n_offset < N, other=float('-inf'))
    local_max = tl.max(inp_vals, axis=0)
    local_argmax = tl.argmax(inp_vals, axis=0)
    update = local_max > max_value
    max_value = tl.where(update, local_max, max_value)
    argmax_value = tl.where(update, start_n + local_argmax, argmax_value)
tl.store(out_vals + row, max_value, mask=row < M)
```

`operators/reduce/max/max_op.cpp` 同步改为 `BLOCK_M=1`、`BLOCK_N=256`、4 warps、`num_blocks=M`。每行的分块扫描仍覆盖完整 N 维，跨块通过 scalar max/index 状态合并，结果语义不变。

## 3. `argmax`

### 原始实现和报错

原实现使用 `m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k`，与 `max` 相同地创建二维广播 pointer。P800 报同样的：

```text
'tt.addptr' op requires the same encoding for all operands
OutOfResources: uni_sram PassManager::run failed
```

### 修改和原理

`operators/reduce/argmax/argmax.py` 改为 `row = tl.program_id(0)`，每次只加载一行 N 元素；`tl.argmax(..., axis=0)` 得到当前 block 的局部 index，再加上 `start_n` 形成全局 index。C++ wrapper 使用 `BLOCK_M=1`、`BLOCK_N=256`、4 warps、`num_blocks=M`。

该改法同时移除了原先为通用 K 维保留的二维地址组合；当前 wrapper 在 permute 后使用 `K=1`，所以 row-local 地址与实际 contiguous reduction layout 一致。

## 4. `apply_rotary_pos_emb`

### 原始实现和报错

原 kernel 同时生成 token、head 和 dim 向量，并交给 helper 广播：

```python
token_range = token_index * BLOCK_N + tl.arange(0, BLOCK_N)
head_range = head_index * BLOCK_H + tl.arange(0, BLOCK_H)
dim_range_x = tl.arange(0, BLOCK_D // 2)
rotary_embedding_rw_kernel(..., token_range, head_range,
                           dim_range_x, dim_range_y)
```

P800 OffsetAnalysis 在地址乘法分析阶段报：

```text
Assertion `lhs.size() == rhs.size() &&
"Two operands size must be equal"' failed
```

### 原理

OffsetAnalysis 需要对 `arith::MulIOp` 的左右 offset 向量建立逐元素关系。token/head/dim 三个不同长度的向量经 helper 广播后，某些乘法 operand 的 offset list size 不相等，导致 compiler assertion。问题在地址表达式的形状，不在旋转公式本身。

### 修改

`operators/fusion/apply_rotary_pos_emb/apply_rotary_pos_emb.py` 改为一个 program 处理一个 token/head，仅 dim 保留一维：

```python
token = tl.program_id(0)
head = tl.program_id(1)
dim = tl.arange(0, BLOCK_D // 2)
state_base = token * stride_state_n + head * stride_state_h
state_x_offset = state_base + dim * stride_state_d
state_y_offset = state_base + (dim + BLOCK_D // 2) * stride_state_d
cos_offset = token * stride_cos_n + dim * stride_cos_d
```

随后分别加载 x/y、cos/sin，在 fp32 中计算旋转并写回。`apply_rotary_pos_emb_op.cpp` 将 Q/K grid 从 block 数改为 `num_tokens x num_heads`，与新的 program 粒度严格匹配。

## 5. 与旧 P800 分支的关系

旧分支 `klx/p800-xpu-launch` 的 `b593f60` 已包含上述六个算子源码文件的同类修改；rebase 不会自动带入另一条分支的 commit。本 PR 在 PR-20 的 Kunlunxin backend 基线上重新应用这些已验证的 layout 改写，并额外修复新的 launch ABI、依赖和测试框架问题。

## 6. 代码位置

完整代码差异以本 PR 的 **Files changed** 为准，重点文件：

- `operators/reduce/max/max.py`, `max_op.cpp`
- `operators/reduce/argmax/argmax.py`, `argmax_op.cpp`
- `operators/fusion/apply_rotary_pos_emb/apply_rotary_pos_emb.py`, `apply_rotary_pos_emb_op.cpp`
- `include/triton_jit/backends/kunlunxin_backend.h`
