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
elseif(BACKEND STREQUAL "HCU")
    target_link_libraries(triton_test_framework PRIVATE hip::host)
elseif(BACKEND STREQUAL "IX")
    target_link_libraries(triton_test_framework PRIVATE CUDA::cuda_driver CUDA::cudart)
else()  # CUDA
    target_link_libraries(triton_test_framework PRIVATE CUDA::cuda_driver CUDA::cudart)
endif()

# Create alias for easier use
add_library(TritonJIT::test_framework ALIAS triton_test_framework)
