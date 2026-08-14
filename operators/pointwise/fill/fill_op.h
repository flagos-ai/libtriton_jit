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

#ifndef TRITON_JIT_FILL_OP_H
#define TRITON_JIT_FILL_OP_H

#include <torch/torch.h>

namespace my_ops {

/**
 * @brief Fill tensor with a constant value (out-of-place)
 * @param input Input tensor (shape is preserved)
 * @param value Fill value
 * @return New tensor filled with value
 */
at::Tensor fill_tensor(const at::Tensor& input, double value);

/**
 * @brief Fill tensor with a constant value (in-place)
 * @param input Input tensor to modify
 * @param value Fill value
 * @return Reference to modified input
 */
at::Tensor& fill_tensor_(at::Tensor& input, double value);

}  // namespace my_ops

#endif  // TRITON_JIT_FILL_OP_H
