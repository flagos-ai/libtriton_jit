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

#include <concepts>
#include <string>
#include <type_traits>

namespace triton_jit {

template <typename T>
concept BackendPolicy = requires {
  typename T::StreamType;
  typename T::ContextType;
  typename T::KernelHandle;
  typename T::LaunchOptions;

  // Each backend must define its warp size (CUDA: 32, IX: 64)
  { T::WARP_SIZE } -> std::convertible_to<unsigned int>;
}
&&requires(typename T::StreamType stream,
           typename T::KernelHandle kernel,
           unsigned grid_x,
           unsigned grid_y,
           unsigned grid_z,
           unsigned block_x,
           unsigned block_y,
           unsigned block_z,
           void** args,
           const typename T::LaunchOptions& opts) {
  {
    T::launch_kernel(stream, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z, args, opts)
    } -> std::same_as<void>;

  { T::ensure_context() } -> std::same_as<void>;

  { T::get_device_index() } -> std::same_as<int>;
}
&&requires(const std::string& dir, const std::string& name) {
  { T::load_kernel(dir, name) } -> std::same_as<typename T::KernelHandle>;

  { T::get_shared_memory(dir, name) } -> std::same_as<unsigned int>;
}
&&requires(const std::string& dir,
           const std::string& name,
           unsigned int shared_mem,
           const std::string& sig,
           size_t num_args) {
  { T::prepare_launch(dir, name, shared_mem, sig, num_args) } -> std::same_as<typename T::LaunchOptions>;
};

}  // namespace triton_jit
