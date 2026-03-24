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
