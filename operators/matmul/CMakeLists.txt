# operators/matmul/CMakeLists.txt
# Matrix multiplication operators: mm, bmm, addmm

function(configure_matmul_target TARGET_NAME)
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
# mm operator
# ==============================================================================

add_custom_target(copy_matmul_mm_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/mm
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/mm/mm.py
            ${CMAKE_CURRENT_BINARY_DIR}/mm/mm.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mm/mm.py
)

add_library(mm_op SHARED mm/mm_op.cpp)
target_include_directories(mm_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/mm)
target_link_libraries(mm_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_matmul_target(mm_op)
add_dependencies(mm_op copy_matmul_mm_python)
set_target_properties(mm_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mm)

add_executable(test_mm mm/test_mm.cpp)
target_include_directories(test_mm PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_mm PRIVATE mm_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_matmul_target(test_mm)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_mm PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_mm copy_matmul_mm_python)
set_target_properties(test_mm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mm)

# ==============================================================================
# bmm operator
# ==============================================================================

add_custom_target(copy_matmul_bmm_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/bmm
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/bmm/bmm.py
            ${CMAKE_CURRENT_BINARY_DIR}/bmm/bmm.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/bmm/bmm.py
)

add_library(bmm_op SHARED bmm/bmm_op.cpp)
target_include_directories(bmm_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/bmm)
target_link_libraries(bmm_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_matmul_target(bmm_op)
add_dependencies(bmm_op copy_matmul_bmm_python)
set_target_properties(bmm_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bmm)

add_executable(test_bmm bmm/test_bmm.cpp)
target_include_directories(test_bmm PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_bmm PRIVATE bmm_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_matmul_target(test_bmm)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_bmm PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_bmm copy_matmul_bmm_python)
set_target_properties(test_bmm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bmm)

# ==============================================================================
# addmm operator
# ==============================================================================

add_custom_target(copy_matmul_addmm_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/addmm
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/addmm/addmm.py
            ${CMAKE_CURRENT_BINARY_DIR}/addmm/addmm.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/addmm/addmm.py
)

add_library(addmm_op SHARED addmm/addmm_op.cpp)
target_include_directories(addmm_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/addmm)
target_link_libraries(addmm_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_matmul_target(addmm_op)
add_dependencies(addmm_op copy_matmul_addmm_python)
set_target_properties(addmm_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/addmm)

add_executable(test_addmm addmm/test_addmm.cpp)
target_include_directories(test_addmm PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_addmm PRIVATE addmm_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_matmul_target(test_addmm)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_addmm PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_addmm copy_matmul_addmm_python)
set_target_properties(test_addmm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/addmm)
