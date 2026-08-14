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

#ifndef TRITON_JIT_APPLY_ROTARY_POS_EMB_OP_H
#define TRITON_JIT_APPLY_ROTARY_POS_EMB_OP_H

#include <torch/torch.h>

namespace my_ops {

std::tuple<at::Tensor, at::Tensor> apply_rotary_pos_emb(const at::Tensor& q,
                                                        const at::Tensor& k,
                                                        const at::Tensor& cos,
                                                        const at::Tensor& sin,
                                                        int64_t rotary_dim);

}  // namespace my_ops

#endif  // TRITON_JIT_APPLY_ROTARY_POS_EMB_OP_H
