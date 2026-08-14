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

# operators/common/CMakeLists.txt
# Common test framework library

# Create static library for test framework
add_library(triton_test_framework STATIC
    test_framework.cpp
)

target_include_directories(triton_test_framework
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(triton_test_framework
    PUBLIC
        Torch::Torch
)

# Backend-specific configuration
if(BACKEND STREQUAL "NPU")
    target_link_ascend_libraries(triton_test_framework)
    if(TORCH_NPU_INCLUDE_PATHS)
        target_include_directories(triton_test_framework PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
    endif()
elseif(BACKEND STREQUAL "MUSA")
    target_link_musa_libraries(triton_test_framework)
    target_link_libraries(triton_test_framework PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "MLU")
    target_link_mlu_libraries(triton_test_framework)
    target_link_libraries(triton_test_framework PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "GCU")
    target_link_gcu_libraries(triton_test_framework)
    target_link_libraries(triton_test_framework PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "HCU")
    target_link_hcu_libraries(triton_test_framework)
    target_link_libraries(triton_test_framework PRIVATE pybind11::embed)
elseif(BACKEND STREQUAL "IX")
    target_link_libraries(triton_test_framework PRIVATE CUDA::cuda_driver CUDA::cudart)
else()  # CUDA
    target_link_libraries(triton_test_framework PRIVATE CUDA::cuda_driver CUDA::cudart)
endif()

# Create alias for easier use
add_library(TritonJIT::test_framework ALIAS triton_test_framework)
