# ==============================================================================
# Pointwise Examples
# ==============================================================================

# ------------------------------------------------------------------------------
# Helper function: Configure NPU-related settings for a target
# ------------------------------------------------------------------------------
function(configure_npu_target TARGET_NAME)
    target_link_ascend_libraries(${TARGET_NAME})
    target_compile_definitions(${TARGET_NAME} PRIVATE WITH_NPU)
    
    if(TORCH_NPU_INCLUDE_PATHS)
        target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
    endif()
    
    if(TORCH_NPU_BASE_PATH)
        target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_BASE_PATH})
    endif()
    
    if(TORCH_NPU_LIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_NPU_LIB})
    endif()
    
    if(GFORTRAN_LIB)
        target_link_libraries(${TARGET_NAME} PRIVATE ${GFORTRAN_LIB})
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Copy Python sources to build directory
# ------------------------------------------------------------------------------
add_custom_target(copy_triton_pointwise_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/add_triton_cpp_rt.py
    DEPENDS 
            ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
)

# ------------------------------------------------------------------------------
# add_op library
# ------------------------------------------------------------------------------
add_library(add_op SHARED add_op.cpp)

target_include_directories(add_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(add_op
    PUBLIC  Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_npu_target(add_op)
add_dependencies(add_op copy_triton_pointwise_src)

# ------------------------------------------------------------------------------
# test_add executable
# ------------------------------------------------------------------------------
add_executable(test_add test_add.cpp)

target_include_directories(test_add
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_add
    PRIVATE add_op
    PRIVATE Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_npu_target(test_add)
add_dependencies(test_add copy_triton_pointwise_src)

