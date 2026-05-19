#pragma once

#include <torch/torch.h>

#if defined(BACKEND_NPU) || defined(BACKEND_MUSA) || defined(BACKEND_MLU)
#define TRITON_DISPATCH_KEY PrivateUse1
#else
#define TRITON_DISPATCH_KEY CUDA
#endif

// Usage: REGISTER_TRITON_OP(my_ops, "add_tensor", add_tensor)
#define REGISTER_TRITON_OP(lib, name, func)         \
  TORCH_LIBRARY_IMPL(lib, TRITON_DISPATCH_KEY, m) { \
    m.impl(name, TORCH_FN(func));                   \
  }
