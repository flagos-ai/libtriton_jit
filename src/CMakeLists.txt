# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Since the library links torch, we need to ensure the same ABI with pre-compiled torch
# libtorch installed from pypi is not built with cxx11 abi,
# if we split the library into several targets, we need to ensure that the ABI is the same
# with torch. It is a bit ugly to specify a target that does not depend on torch to use
# the cxx flags from torch, so we just merge then as one target, for simplicity
# then it can use the same cxx flags with public dependency transitivity
# --------------------------- triton jit function ---------------------------
add_library(
  triton_jit SHARED
  triton_jit_function.cpp
  jit_function_arg.cpp
  jit_utils.cpp
  kernel_metadata.cpp
  launch_hooks.cpp)
target_include_directories(triton_jit
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Backend-specific link libraries
if(BACKEND STREQUAL "NPU")
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch Ascend::ascendcl Ascend::runtime fmt::fmt-header-only
      PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "MUSA")
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch MUSA::musa_runtime fmt::fmt-header-only
      PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "MACA")
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch MACA::mcruntime fmt::fmt-header-only
      PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "MLU")
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch MLU::mlu_runtime fmt::fmt-header-only
      PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "GCU")
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch GCU::efrt fmt::fmt-header-only
      PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "HCU")
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch hip::host fmt::fmt-header-only
      PRIVATE pybind11::embed)
else()
    target_link_libraries(triton_jit
      PUBLIC Torch::Torch CUDA::cuda_driver fmt::fmt-header-only
      PRIVATE pybind11::embed)
endif()

# Set backend compile definition based on BACKEND variable
target_compile_definitions(triton_jit PUBLIC BACKEND_${BACKEND})

# nlohmann_json is only used in kernel_metadata.cpp, keep it PRIVATE
target_link_libraries(triton_jit PRIVATE nlohmann_json::nlohmann_json)

# --------------------------- alias targets ---------------------------
# This is the target used in FetchContent, since FetchContent use add_subdirectory (a sub project build)
# So to use consistent target name, add namespace here
add_library(TritonJIT::triton_jit ALIAS triton_jit)

# --------------------------- install ---------------------------
if(TRITON_JIT_INSTALL)
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/triton_jit" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/triton_jit/" FILES_MATCHING PATTERN "*.py")
  install(
    TARGETS triton_jit
    EXPORT TritonJITTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(
    EXPORT TritonJITTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
    NAMESPACE TritonJIT::
    FILE TritonJITTargets.cmake
  )
  # This is for find_package to work with the build tree, instead of the install tree
  export(EXPORT TritonJITTargets FILE "${PROJECT_BINARY_DIR}/TritonJITTargets.cmake")
endif()
