# operators/fusion/CMakeLists.txt
# Fusion operators: apply_rotary_pos_emb, rwkv_ka_fusion, rwkv_mm_sparsity

function(configure_fusion_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
    if(BACKEND STREQUAL "NPU")
        target_link_ascend_libraries(${TARGET_NAME})
        if(TORCH_NPU_INCLUDE_PATHS)
            target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
        endif()
    elseif(BACKEND STREQUAL "MUSA")
        target_link_musa_libraries(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    elseif(BACKEND STREQUAL "MLU")
        target_link_mlu_libraries(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    else()
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver)
    endif()
endfunction()

# ==============================================================================
# apply_rotary_pos_emb operator
# ==============================================================================

add_custom_target(copy_fusion_apply_rotary_pos_emb_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/apply_rotary_pos_emb
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/apply_rotary_pos_emb/apply_rotary_pos_emb.py
            ${CMAKE_CURRENT_BINARY_DIR}/apply_rotary_pos_emb/apply_rotary_pos_emb.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/apply_rotary_pos_emb/apply_rotary_pos_emb.py
)

add_library(apply_rotary_pos_emb_op SHARED apply_rotary_pos_emb/apply_rotary_pos_emb_op.cpp)
target_include_directories(apply_rotary_pos_emb_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/apply_rotary_pos_emb)
target_link_libraries(apply_rotary_pos_emb_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_fusion_target(apply_rotary_pos_emb_op)
add_dependencies(apply_rotary_pos_emb_op copy_fusion_apply_rotary_pos_emb_python)
set_target_properties(apply_rotary_pos_emb_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/apply_rotary_pos_emb)

add_executable(test_apply_rotary_pos_emb apply_rotary_pos_emb/test_apply_rotary_pos_emb.cpp)
target_include_directories(test_apply_rotary_pos_emb PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_apply_rotary_pos_emb PRIVATE apply_rotary_pos_emb_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_fusion_target(test_apply_rotary_pos_emb)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_apply_rotary_pos_emb PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_apply_rotary_pos_emb copy_fusion_apply_rotary_pos_emb_python)
set_target_properties(test_apply_rotary_pos_emb PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/apply_rotary_pos_emb)

# ==============================================================================
# rwkv_ka_fusion operator
# ==============================================================================

add_custom_target(copy_fusion_rwkv_ka_fusion_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rwkv_ka_fusion
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/rwkv_ka_fusion/rwkv_ka_fusion.py
            ${CMAKE_CURRENT_BINARY_DIR}/rwkv_ka_fusion/rwkv_ka_fusion.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rwkv_ka_fusion/rwkv_ka_fusion.py
)

add_library(rwkv_ka_fusion_op SHARED rwkv_ka_fusion/rwkv_ka_fusion_op.cpp)
target_include_directories(rwkv_ka_fusion_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rwkv_ka_fusion)
target_link_libraries(rwkv_ka_fusion_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_fusion_target(rwkv_ka_fusion_op)
add_dependencies(rwkv_ka_fusion_op copy_fusion_rwkv_ka_fusion_python)
set_target_properties(rwkv_ka_fusion_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rwkv_ka_fusion)

add_executable(test_rwkv_ka_fusion rwkv_ka_fusion/test_rwkv_ka_fusion.cpp)
target_include_directories(test_rwkv_ka_fusion PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_rwkv_ka_fusion PRIVATE rwkv_ka_fusion_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_fusion_target(test_rwkv_ka_fusion)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_rwkv_ka_fusion PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_rwkv_ka_fusion copy_fusion_rwkv_ka_fusion_python)
set_target_properties(test_rwkv_ka_fusion PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rwkv_ka_fusion)

# ==============================================================================
# rwkv_mm_sparsity operator
# ==============================================================================

add_custom_target(copy_fusion_rwkv_mm_sparsity_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/rwkv_mm_sparsity
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/rwkv_mm_sparsity/rwkv_mm_sparsity.py
            ${CMAKE_CURRENT_BINARY_DIR}/rwkv_mm_sparsity/rwkv_mm_sparsity.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rwkv_mm_sparsity/rwkv_mm_sparsity.py
)

add_library(rwkv_mm_sparsity_op SHARED rwkv_mm_sparsity/rwkv_mm_sparsity_op.cpp)
target_include_directories(rwkv_mm_sparsity_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rwkv_mm_sparsity)
target_link_libraries(rwkv_mm_sparsity_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_fusion_target(rwkv_mm_sparsity_op)
add_dependencies(rwkv_mm_sparsity_op copy_fusion_rwkv_mm_sparsity_python)
set_target_properties(rwkv_mm_sparsity_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rwkv_mm_sparsity)

add_executable(test_rwkv_mm_sparsity rwkv_mm_sparsity/test_rwkv_mm_sparsity.cpp)
target_include_directories(test_rwkv_mm_sparsity PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_rwkv_mm_sparsity PRIVATE rwkv_mm_sparsity_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_fusion_target(test_rwkv_mm_sparsity)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_rwkv_mm_sparsity PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_rwkv_mm_sparsity copy_fusion_rwkv_mm_sparsity_python)
set_target_properties(test_rwkv_mm_sparsity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/rwkv_mm_sparsity)
