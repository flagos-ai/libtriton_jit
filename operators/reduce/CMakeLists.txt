# operators/reduce/CMakeLists.txt
# Reduction operators: sum, max, argmax, topk

# ==============================================================================
# Backend-specific Target Configuration
# ==============================================================================
function(configure_reduce_target TARGET_NAME)
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
# sum operator
# ==============================================================================

add_custom_target(copy_reduce_sum_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/sum
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/sum/sum.py
            ${CMAKE_CURRENT_BINARY_DIR}/sum/sum.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/sum/sum.py
)

add_library(sum_op SHARED sum/sum_op.cpp)
target_include_directories(sum_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/sum)
target_link_libraries(sum_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_reduce_target(sum_op)
add_dependencies(sum_op copy_reduce_sum_python)
set_target_properties(sum_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sum)

add_executable(test_sum sum/test_sum.cpp)
target_include_directories(test_sum PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_sum PRIVATE sum_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_reduce_target(test_sum)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_sum PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_sum copy_reduce_sum_python)
set_target_properties(test_sum PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/sum)

# ==============================================================================
# max operator
# ==============================================================================

add_custom_target(copy_reduce_max_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/max
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/max/max.py
            ${CMAKE_CURRENT_BINARY_DIR}/max/max.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/max/max.py
)

add_library(max_op SHARED max/max_op.cpp)
target_include_directories(max_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/max)
target_link_libraries(max_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_reduce_target(max_op)
add_dependencies(max_op copy_reduce_max_python)
set_target_properties(max_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/max)

add_executable(test_max max/test_max.cpp)
target_include_directories(test_max PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_max PRIVATE max_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_reduce_target(test_max)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_max PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_max copy_reduce_max_python)
set_target_properties(test_max PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/max)

# ==============================================================================
# argmax operator
# ==============================================================================

add_custom_target(copy_reduce_argmax_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/argmax
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/argmax/argmax.py
            ${CMAKE_CURRENT_BINARY_DIR}/argmax/argmax.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/argmax/argmax.py
)

add_library(argmax_op SHARED argmax/argmax_op.cpp)
target_include_directories(argmax_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/argmax)
target_link_libraries(argmax_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_reduce_target(argmax_op)
add_dependencies(argmax_op copy_reduce_argmax_python)
set_target_properties(argmax_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/argmax)

add_executable(test_argmax argmax/test_argmax.cpp)
target_include_directories(test_argmax PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_argmax PRIVATE argmax_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_reduce_target(test_argmax)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_argmax PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_argmax copy_reduce_argmax_python)
set_target_properties(test_argmax PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/argmax)

# ==============================================================================
# topk operator
# ==============================================================================

add_custom_target(copy_reduce_topk_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/topk
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/topk/topk.py
            ${CMAKE_CURRENT_BINARY_DIR}/topk/topk.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/topk/topk.py
)

add_library(topk_op SHARED topk/topk_op.cpp)
target_include_directories(topk_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/topk)
target_link_libraries(topk_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_reduce_target(topk_op)
add_dependencies(topk_op copy_reduce_topk_python)
set_target_properties(topk_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/topk)

add_executable(test_topk topk/test_topk.cpp)
target_include_directories(test_topk PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_topk PRIVATE topk_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_reduce_target(test_topk)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_topk PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_topk copy_reduce_topk_python)
set_target_properties(test_topk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/topk)
