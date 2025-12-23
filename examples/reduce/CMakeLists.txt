# ==============================================================================
# Reduce Examples - Multi-backend Support
# ==============================================================================

# ------------------------------------------------------------------------------
# Backend-specific Target Configuration
# ------------------------------------------------------------------------------
function(configure_backend_target TARGET_NAME)
    if(BACKEND STREQUAL "NPU")
        # ------------------------- NPU Backend --------------------------------
        target_link_ascend_libraries(${TARGET_NAME})

        if(TORCH_NPU_INCLUDE_PATHS)
            target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
        endif()

        if(TORCH_NPU_LIB)
            target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_NPU_LIB})
        endif()

    elseif(BACKEND STREQUAL "MUSA")
        # ------------------------- MUSA Backend -------------------------------
        # TODO: Add MUSA-specific configuration
        # target_link_musa_libraries(${TARGET_NAME})

    else()
        # ------------------------- CUDA / IX Backend --------------------------
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver)

    endif()
endfunction()

# ------------------------------------------------------------------------------
# Copy Python Sources to Build Directory
# ------------------------------------------------------------------------------
add_custom_target(copy_triton_reduce_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_compile.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum_compile.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_triton_cpp_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum_triton_cpp_rt.py
    DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/sum.py
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_compile.py
            ${CMAKE_CURRENT_SOURCE_DIR}/sum_triton_cpp_rt.py
)

# ------------------------------------------------------------------------------
# sum_op Library
# ------------------------------------------------------------------------------
add_library(sum_op SHARED sum_op.cpp)

target_include_directories(sum_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(sum_op
    PUBLIC  Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_backend_target(sum_op)
add_dependencies(sum_op copy_triton_reduce_src)

# ------------------------------------------------------------------------------
# test_sum Executable
# ------------------------------------------------------------------------------
add_executable(test_sum test_sum.cpp)

target_include_directories(test_sum
    PRIVATE ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_sum
    PRIVATE sum_op
    PRIVATE Torch::Torch
    PRIVATE TritonJIT::triton_jit
)

configure_backend_target(test_sum)
add_dependencies(test_sum copy_triton_reduce_src)
