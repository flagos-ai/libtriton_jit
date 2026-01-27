# operators/pointwise/CMakeLists.txt
# Pointwise operators: add, fill, fill_, zeros, exponential_, contiguous

# ==============================================================================
# Backend-specific Target Configuration
# ==============================================================================
function(configure_pointwise_target TARGET_NAME)
    if(BACKEND STREQUAL "NPU")
        target_link_ascend_libraries(${TARGET_NAME})
        if(TORCH_NPU_INCLUDE_PATHS)
            target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
        endif()
        if(TORCH_NPU_LIB)
            target_link_libraries(${TARGET_NAME} PRIVATE ${TORCH_NPU_LIB})
        endif()
    elseif(BACKEND STREQUAL "MUSA")
        target_link_musa_libraries(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    else()  # CUDA / IX
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver)
    endif()
endfunction()

# ==============================================================================
# add operator
# ==============================================================================

# Copy Python sources
add_custom_target(copy_pointwise_add_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/add
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/add/add.py
            ${CMAKE_CURRENT_BINARY_DIR}/add/add.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/add/add.py
)

# add_op library
add_library(add_op SHARED add/add_op.cpp)
target_include_directories(add_op
    PRIVATE ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/add
)
target_link_libraries(add_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_pointwise_target(add_op)
add_dependencies(add_op copy_pointwise_add_python)
set_target_properties(add_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/add)

# test_add executable
add_executable(test_add add/test_add.cpp)
target_include_directories(test_add PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_add PRIVATE add_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_pointwise_target(test_add)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_add PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_add copy_pointwise_add_python)
set_target_properties(test_add PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/add)

# ==============================================================================
# fill operator
# ==============================================================================

add_custom_target(copy_pointwise_fill_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fill
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/fill/fill.py
            ${CMAKE_CURRENT_BINARY_DIR}/fill/fill.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fill/fill.py
)

add_library(fill_op SHARED fill/fill_op.cpp)
target_include_directories(fill_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fill)
target_link_libraries(fill_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_pointwise_target(fill_op)
add_dependencies(fill_op copy_pointwise_fill_python)
set_target_properties(fill_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fill)

add_executable(test_fill fill/test_fill.cpp)
target_include_directories(test_fill PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_fill PRIVATE fill_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_pointwise_target(test_fill)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_fill PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_fill copy_pointwise_fill_python)
set_target_properties(test_fill PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fill)

# ==============================================================================
# zeros operator
# ==============================================================================

add_custom_target(copy_pointwise_zeros_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/zeros
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/zeros/zeros.py
            ${CMAKE_CURRENT_BINARY_DIR}/zeros/zeros.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/zeros/zeros.py
)

add_library(zeros_op SHARED zeros/zeros_op.cpp)
target_include_directories(zeros_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/zeros)
target_link_libraries(zeros_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_pointwise_target(zeros_op)
add_dependencies(zeros_op copy_pointwise_zeros_python)
set_target_properties(zeros_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zeros)

add_executable(test_zeros zeros/test_zeros.cpp)
target_include_directories(test_zeros PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_zeros PRIVATE zeros_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_pointwise_target(test_zeros)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_zeros PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_zeros copy_pointwise_zeros_python)
set_target_properties(test_zeros PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/zeros)

# ==============================================================================
# exponential_ operator (in-place)
# ==============================================================================

add_custom_target(copy_pointwise_exponential_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/exponential_
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/exponential_/exponential_.py
            ${CMAKE_CURRENT_BINARY_DIR}/exponential_/exponential_.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exponential_/exponential_.py
)

add_library(exponential_op SHARED exponential_/exponential_op.cpp)
target_include_directories(exponential_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/exponential_)
target_link_libraries(exponential_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_pointwise_target(exponential_op)
add_dependencies(exponential_op copy_pointwise_exponential_python)
set_target_properties(exponential_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/exponential_)

add_executable(test_exponential_ exponential_/test_exponential_.cpp)
target_include_directories(test_exponential_ PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_exponential_ PRIVATE exponential_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_pointwise_target(test_exponential_)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_exponential_ PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_exponential_ copy_pointwise_exponential_python)
set_target_properties(test_exponential_ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/exponential_)

# ==============================================================================
# fill_ operator (in-place)
# ==============================================================================

add_custom_target(copy_pointwise_fill_inplace_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fill_
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/fill_/fill_.py
            ${CMAKE_CURRENT_BINARY_DIR}/fill_/fill_.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fill_/fill_.py
)

add_library(fill_inplace_op SHARED fill_/fill_op.cpp)
target_include_directories(fill_inplace_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fill_)
target_link_libraries(fill_inplace_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_pointwise_target(fill_inplace_op)
add_dependencies(fill_inplace_op copy_pointwise_fill_inplace_python)
set_target_properties(fill_inplace_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fill_)

add_executable(test_fill_ fill_/test_fill_.cpp)
target_include_directories(test_fill_ PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_fill_ PRIVATE fill_inplace_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_pointwise_target(test_fill_)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_fill_ PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_fill_ copy_pointwise_fill_inplace_python)
set_target_properties(test_fill_ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fill_)

# ==============================================================================
# contiguous operator
# ==============================================================================

add_custom_target(copy_pointwise_contiguous_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/contiguous
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/contiguous/contiguous.py
            ${CMAKE_CURRENT_BINARY_DIR}/contiguous/contiguous.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/contiguous/contiguous.py
)

add_library(contiguous_op SHARED contiguous/contiguous_op.cpp)
target_include_directories(contiguous_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/contiguous)
target_link_libraries(contiguous_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_pointwise_target(contiguous_op)
add_dependencies(contiguous_op copy_pointwise_contiguous_python)
set_target_properties(contiguous_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/contiguous)

add_executable(test_contiguous contiguous/test_contiguous.cpp)
target_include_directories(test_contiguous PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_contiguous PRIVATE contiguous_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_pointwise_target(test_contiguous)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_contiguous PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_contiguous copy_pointwise_contiguous_python)
set_target_properties(test_contiguous PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/contiguous)
