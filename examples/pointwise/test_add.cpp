#include "add_op.h"
#include "c10/cuda/CUDAFunctions.h"
#include "torch/torch.h"
#include <iostream>

int main() {
    at::Tensor a = at::rand({128 * 1024}, at::kCUDA);
    at::Tensor b = at::rand({128 * 1024}, at::kCUDA);
    
    // 打印输入信息
    std::cout << "=== 输入张量信息 ===" << std::endl;
    std::cout << "张量大小: " << a.size(0) << " 元素" << std::endl;
    std::cout << "设备: " << a.device() << std::endl;
    
    // 复制到CPU后打印
    at::Tensor a_cpu = a.cpu();
    at::Tensor b_cpu = b.cpu();
    std::cout << "a的前5个元素: " << a_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "b的前5个元素: " << b_cpu.slice(0, 0, 5) << std::endl;
    
    // warm up
    std::cout << "\n=== 执行计算 ===" << std::endl;
    at::Tensor result1 = my_ops::add_tensor(a, b);
    at::Tensor result2 = at::add(a, b);
    c10::cuda::device_synchronize();
    
    // 复制结果到CPU后打印
    std::cout << "\n=== 计算结果 ===" << std::endl;
    at::Tensor result1_cpu = result1.cpu();
    at::Tensor result2_cpu = result2.cpu();
    std::cout << "my_ops::add_tensor 结果的前5个元素: " 
              << result1_cpu.slice(0, 0, 5) << std::endl;
    std::cout << "at::add 结果的前5个元素: " 
              << result2_cpu.slice(0, 0, 5) << std::endl;
    
    // 验证结果是否一致
    bool is_close = at::allclose(result1, result2);
    std::cout << "\n结果是否一致: " << (is_close ? "是" : "否") << std::endl;
    if (!is_close) {
        auto diff = at::abs(result1 - result2);
        std::cout << "最大差异: " << at::max(diff).item<float>() << std::endl;
    }
    
    // 原有的性能测试循环
    for (int i = 0; i < 10; ++i) {
        auto tmp = at::add(a, b);
    }
    c10::cuda::device_synchronize();
    
    for (int i = 0; i < 10; ++i) {
        auto tmp = my_ops::add_tensor(a, b);
    }
    c10::cuda::device_synchronize();
    
    std::cout << "\n程序执行完成!" << std::endl;
    return 0;
}