import torch
#import torch_npu  # 单独执行python引用该包
import triton
from triton import language as tl
import time


@triton.jit
def binary_pointwise_kernel(X, Y, Out, n, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n

    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    o = x + y
    tl.store(Out + offsets, o, mask=mask)


def binary_add_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # lets be simple and assume x and y are the same shape,
    # all-contiguous, the same dtype
    assert x.shape == y.shape, f"SHAPE MISMATCH: x={x.shape}, y={y.shape}"
    assert x.is_contiguous(), "TENSOR X MUST BE CONTIGUOUS"
    assert y.is_contiguous(), "TENSOR Y MUST BE CONTIGUOUS"
    
    out = torch.empty_like(x, dtype=x.dtype)
    n = out.numel()
    BLOCK_N = 1024
    grid = (triton.cdiv(n, BLOCK_N), 1, 1)
    
    binary_pointwise_kernel[grid](
        x, y, out, n, BLOCK_N=BLOCK_N, num_warps=8, num_stages=1
    )
    return out


if __name__ == "__main__":
    torch.npu.set_device(7)
    torch.manual_seed(0)
    
    print("=" * 60)
    print("Triton ADD TEST")
    
    # 测试不同的向量大小
    test_sizes = [
        (128, "128 elements"),
        # (128 * 1024, "128K elements"),
        # (1024 * 1024, "1M elements"),
        # (16 * 1024 * 1024, "16M elements"),
    ]
    
    device = "npu"
    print(f"USE DEVICE: {device}")
    
    for N, size_name in test_sizes:
        print("\n" + "=" * 60)
        print(f"TEST SIZE: {size_name} (N={N:,})")
        
        # 创建测试数据 - 华为NPU
        x = torch.randn(N, device=device, dtype=torch.float32)
        y = torch.randn(N, device=device, dtype=torch.float32)
        
        memory_size_kb = x.numel() * x.element_size() / 1024
        print(f"\nTensor shape: {x.shape}")
        print(f"Tensor dtype: {x.dtype}")
        print(f"Memory size per tensor: {memory_size_kb:.2f} KB")
        
        # ========== 首次运行（包含 JIT 编译） ==========
        print("\n" + "-" * 60)
        print("FIRST RUN (includes JIT compilation)")
        
        t_first_total_start = time.perf_counter()
        triton_output_1 = binary_add_tensor(x, y)
        torch.npu.synchronize()
        t_first_total_end = time.perf_counter()
        t_first_total = (t_first_total_end - t_first_total_start) * 1000
        
        print(f"[TIMING] First run total: {t_first_total:.3f} ms")
        print("  (includes: kernel compilation + code generation + binary loading)")
        
        # ========== 第二次运行（使用缓存） ==========
        print("\n" + "-" * 60)
        print("SECOND RUN (uses cached kernel)")
        
        torch.npu.synchronize()
        t_second_total_start = time.perf_counter()
        triton_output_2 = binary_add_tensor(x, y)
        torch.npu.synchronize()
        t_second_total_end = time.perf_counter()
        t_second_total = (t_second_total_end - t_second_total_start) * 1000
        
        print(f"[TIMING] Second run total: {t_second_total:.3f} ms")
        print("  (only: kernel launch + execution)")
        
        # ========== 计算编译开销 ==========
        t_compilation = t_first_total - t_second_total
        print(f"\n[TIMING] Compilation overhead: {t_compilation:.3f} ms")
        
        # ========== 多次运行取平均（性能基准） ==========
        print("\n" + "-" * 60)
        print("BENCHMARK (100 iterations)")
        
        num_iterations = 100
        torch.npu.synchronize()
        
        t_benchmark_start = time.perf_counter()
        for _ in range(num_iterations):
            _ = binary_add_tensor(x, y)
        torch.npu.synchronize()
        t_benchmark_end = time.perf_counter()
        
        t_avg = ((t_benchmark_end - t_benchmark_start) * 1000) / num_iterations
        print(f"[TIMING] Average kernel time: {t_avg:.3f} ms")
        
             
        # ========== 结果验证 ==========
        print("\n" + "-" * 60)
        print("RESULT VERIFICATION")
        
        torch_result = torch.add(x, y)
        max_diff = torch.max(torch.abs(triton_output_2 - torch_result)).item()
        is_close = torch.allclose(triton_output_2, torch_result, rtol=1e-5, atol=1e-5)
        
        print(f"Triton output (first 10): {triton_output_2[:10].tolist()}")
        print(f"PyTorch output (first 10): {torch_result[:10].tolist()}")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Results match: {'✅ YES' if is_close else '❌ NO'}")
        
        if not is_close:
            print(f"⚠️  Warning: Results do not match within tolerance!")
    
