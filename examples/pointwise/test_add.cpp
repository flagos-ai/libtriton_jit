#include "add_op.h"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <torch/torch.h>
#include <torch/script.h>

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include <torch_npu/torch_npu.h>
#include <cstdlib>
#include "triton_jit/triton_jit_function.h"


using time_point_t = std::chrono::high_resolution_clock::time_point;

inline time_point_t now() {
  return std::chrono::high_resolution_clock::now();
}

inline double elapsed_ms(time_point_t start, time_point_t end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
  setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", 1);

  int32_t deviceId = 0;  // 默认设备 0
  const char* deviceEnv = std::getenv("NPU_DEVICE_ID");
  if (deviceEnv != nullptr) {
      deviceId = std::atoi(deviceEnv);
      std::cout << "Using NPU device from environment variable: " << deviceId << std::endl;
  } else {
      std::cout << "NPU_DEVICE_ID not set, using default device: " << deviceId << std::endl;
  }
  // Initialize ACL and set device
  aclError ret = aclrtSetDevice(deviceId);
  if (ret != ACL_SUCCESS) {
      std::cerr << "aclrtSetDevice failed with error code: " << ret << std::endl;
      return -1;
  }

  // Initialize NPU
  std::string npu_device_str = "npu:" + std::to_string(deviceId);
  torch_npu::init_npu(npu_device_str);
  auto device = at::Device(npu_device_str);
  std::cout << "NPU initialized: " << device << std::endl;

  at::Tensor a = at::rand({128 * 1}).to(device);
  at::Tensor b = at::rand({128 * 1}).to(device);

  std::cout << "First 10 elements of tensor a: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << a[i].item<float>() << " "<< std::endl;
  }
  std::cout << "First 10 elements of tensor b: ";
  for (int i = 0; i < 10; ++i) {
    std::cout << b[i].item<float>() << " "<< std::endl;
  }  

  at::Tensor result;
  // ========== 测量点 1: 首次运行（包含 JIT 编译和加载） ==========
  // Perform Triton add
  std::cout << "\n=== FIRST RUN (includes JIT compilation + loading) ===" << std::endl;
  auto t_first_start = now();
  try {
      result = my_ops::add_tensor(a, b);
      aclrtSynchronizeDevice();
      std::cout << "✓ Triton ADD completed" << std::endl;
  } catch (const std::exception& e) {
      std::cerr << "❌ Exception during Triton ADD: " << e.what() << std::endl;
      return -1;
  }
  auto t_first_end = now();

  double t_first = elapsed_ms(t_first_start, t_first_end);
  std::cout << "[TIMING] First run total: " << t_first << " ms" << std::endl;
  std::cout << "  (includes: Python initialization + kernel compilation + loading + execution)" << std::endl;
  
// ========== 测量点 2: 第二次运行（使用缓存） ==========
std::cout << "\n=== SECOND RUN (uses cached kernel) ===" << std::endl;

// aclrtSynchronizeDevice();
auto t_second_start = now();
result = my_ops::add_tensor(a, b);
aclrtSynchronizeDevice();
auto t_second_end = now();

double t_second = elapsed_ms(t_second_start, t_second_end);
std::cout << "[TIMING] Second run total: " << t_second << " ms" << std::endl;
std::cout << "  (only: parameter packing + kernel launch + execution)" << std::endl;

// ========== 测量点 3: 性能基准测试 ==========
std::cout << "\n=== BENCHMARK (100 iterations) ===" << std::endl;

const int num_iterations = 100;
// aclrtSynchronizeDevice();

auto t_benchmark_start = now();
for (int i = 0; i < num_iterations; ++i) {
  result = my_ops::add_tensor(a, b);  // ✅ 异步启动，允许流水线执行
}

aclrtSynchronizeDevice();
auto t_benchmark_end = now();

double t_total_benchmark = elapsed_ms(t_benchmark_start, t_benchmark_end);
double t_avg = t_total_benchmark / num_iterations;

std::cout << "[TIMING] Total time (" << num_iterations << " iters): " 
          << t_total_benchmark << " ms" << std::endl;
std::cout << "[TIMING] Average kernel time: " << t_avg << " ms" << std::endl;


// ========== 结果验证 ==========
  std::cout << "\n=== RESULT VERIFICATION ===" << std::endl;
  auto r = result.to(at::kCPU).contiguous();
  float* ptr = r.data_ptr<float>();
  std::cout << "First 10 elements of result: ";
  for (int i = 0; i < std::min<int64_t>(10, r.numel()); ++i) {
    std::cout << ptr[i] << " ";
  }
  std::cout << std::endl;
  
  ret = aclrtSynchronizeDevice();
  ret = aclrtResetDevice(deviceId);
  ret = aclFinalize();
  return 0;
}  
