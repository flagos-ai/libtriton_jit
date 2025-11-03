#pragma once

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "cuda.h"
#include "triton_jit/jit_utils.h"

namespace triton_jit {

class TritonJITFunction;

class TritonKernel {
 private:
  // * The directory that contain the IRs(ttir, ttgir, llir, ptx, cubin) & metadata(json file))*/
  std::string dir_;
  /* name of the kernel in cubin */
  std::string kernel_name_;
  unsigned int shared_; /* amount of static shared memory per block (in bytes) required for the cubin*/
  unsigned int arch_;   /* cuda arch */

  mutable CUmodule mod_;
  mutable CUfunction fn_;
  mutable bool loaded_ = false;

 public:
  TritonKernel(const TritonKernel&) = delete;
  TritonKernel& operator=(const TritonKernel&) = delete;
  TritonKernel(TritonKernel&&) = default;
  TritonKernel& operator=(TritonKernel&&) = default;
  TritonKernel() = default;

  void launch(unsigned int grid_x,
              unsigned int grid_y,
              unsigned int grid_z,
              int num_warps,
              CUstream stream,
              void** args) const;
  friend TritonJITFunction;

 private:
  TritonKernel(std::string_view dir, std::string_view kernel_name);
  /* load cubin into a cumodule for a device */
  void lazy_init_handle() const;
};
static_assert(std::is_move_constructible_v<TritonKernel>);
}  // namespace triton_jit
