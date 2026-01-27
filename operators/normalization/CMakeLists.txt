# operators/normalization/CMakeLists.txt
# Normalization operators: softmax, rms_norm, fused_add_rms_norm

function(configure_norm_target TARGET_NAME)
    if(BACKEND STREQUAL "NPU")
        target_link_ascend_libraries(${TARGET_NAME})
        if(TORCH_NPU_INCLUDE_PATHS)
            target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
        endif()
    elseif(BACKEND STREQUAL "MUSA")
        target_link_musa_libraries(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver)
    endif()
endfunction()

# ==============================================================================
# softmax operator
# ==============================================================================

add_custom_target(copy_norm_softmax_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/softmax
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/softmax/softmax.py
            ${CMAKE_CURRENT_BINARY_DIR}/softmax/softmax.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/softmax/softmax.py
)

add_library(softmax_op SHARED softmax/softmax_op.cpp)
target_include_directories(softmax_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/softmax)
target_link_libraries(softmax_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_norm_target(softmax_op)
add_dependencies(softmax_op copy_norm_softmax_python)
set_target_properties(softmax_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/softmax)

add_executable(test_softmax softmax/test_softmax.cpp)
target_include_directories(test_softmax PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_softmax PRIVATE softmax_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_norm_target(test_softmax)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_softmax PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_softmax copy_norm_softmax_python)
set_target_properties(test_softmax PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/softmax)

# ==============================================================================
# rms_norm operator
# ==============================================================================

add_custom_target(copy_norm_rms_norm_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rms_norm
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm/rms_norm.py
            ${CMAKE_CURRENT_BINARY_DIR}/rms_norm/rms_norm.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm/rms_norm.py
)

add_library(rms_norm_op SHARED rms_norm/rms_norm_op.cpp)
target_include_directories(rms_norm_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm)
target_link_libraries(rms_norm_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_norm_target(rms_norm_op)
add_dependencies(rms_norm_op copy_norm_rms_norm_python)
set_target_properties(rms_norm_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rms_norm)

add_executable(test_rms_norm rms_norm/test_rms_norm.cpp)
target_include_directories(test_rms_norm PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_rms_norm PRIVATE rms_norm_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_norm_target(test_rms_norm)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_rms_norm PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_rms_norm copy_norm_rms_norm_python)
set_target_properties(test_rms_norm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rms_norm)

# ==============================================================================
# fused_add_rms_norm operator
# ==============================================================================

add_custom_target(copy_norm_fused_add_rms_norm_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/fused_add_rms_norm
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/fused_add_rms_norm/fused_add_rms_norm.py
            ${CMAKE_CURRENT_BINARY_DIR}/fused_add_rms_norm/fused_add_rms_norm.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/fused_add_rms_norm/fused_add_rms_norm.py
)

add_library(fused_add_rms_norm_op SHARED fused_add_rms_norm/fused_add_rms_norm_op.cpp)
target_include_directories(fused_add_rms_norm_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fused_add_rms_norm)
target_link_libraries(fused_add_rms_norm_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_norm_target(fused_add_rms_norm_op)
add_dependencies(fused_add_rms_norm_op copy_norm_fused_add_rms_norm_python)
set_target_properties(fused_add_rms_norm_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fused_add_rms_norm)

add_executable(test_fused_add_rms_norm fused_add_rms_norm/test_fused_add_rms_norm.cpp)
target_include_directories(test_fused_add_rms_norm PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_fused_add_rms_norm PRIVATE fused_add_rms_norm_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_norm_target(test_fused_add_rms_norm)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_fused_add_rms_norm PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_fused_add_rms_norm copy_norm_fused_add_rms_norm_python)
set_target_properties(test_fused_add_rms_norm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fused_add_rms_norm)
