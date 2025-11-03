
add_custom_target(
    copy_triton_pointwise_src
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
            ${CMAKE_CURRENT_BINARY_DIR}/add_triton_cpp_rt.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add.py
            ${CMAKE_CURRENT_SOURCE_DIR}/add_triton_cpp_rt.py
)

add_library(add_op SHARED add_op.cpp)
target_include_directories(add_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(add_op
    PUBLIC Torch::Torch
    PRIVATE TritonJIT::triton_jit
)
add_dependencies(add_op copy_triton_pointwise_src)

# Add NPU support if enabled
if(TRITON_JIT_ENABLE_NPU)
    if(TORCH_NPU_INCLUDE_PATHS)
        target_include_directories(add_op PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
    endif()

    if(COMMAND target_link_ascend_libraries)
        target_link_ascend_libraries(add_op)
    endif()

    target_compile_definitions(add_op PRIVATE WITH_NPU)
endif()

add_executable(test_add test_add.cpp)
target_link_libraries(test_add
    PRIVATE add_op Torch::Torch)
add_dependencies(test_add copy_triton_pointwise_src)

# Add NPU support for test_add if enabled
if(TRITON_JIT_ENABLE_NPU)
    if(TORCH_NPU_INCLUDE_PATHS)
        target_include_directories(test_add PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
    endif()

    if(COMMAND target_link_ascend_libraries)
        target_link_ascend_libraries(test_add)
    endif()

    target_compile_definitions(test_add PRIVATE WITH_NPU)
endif()
