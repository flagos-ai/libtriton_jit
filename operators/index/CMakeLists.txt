# operators/index/CMakeLists.txt
# Indexing operators: embedding, nonzero, cat, reshape_and_cache_flash

function(configure_index_target TARGET_NAME)
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
# embedding operator
# ==============================================================================

add_custom_target(copy_index_embedding_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/embedding
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding.py
            ${CMAKE_CURRENT_BINARY_DIR}/embedding/embedding.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding.py
)

add_library(embedding_op SHARED embedding/embedding_op.cpp)
target_include_directories(embedding_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/embedding)
target_link_libraries(embedding_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_index_target(embedding_op)
add_dependencies(embedding_op copy_index_embedding_python)
set_target_properties(embedding_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/embedding)

add_executable(test_embedding embedding/test_embedding.cpp)
target_include_directories(test_embedding PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_embedding PRIVATE embedding_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_index_target(test_embedding)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_embedding PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_embedding copy_index_embedding_python)
set_target_properties(test_embedding PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/embedding)

# ==============================================================================
# cat operator
# ==============================================================================

add_custom_target(copy_index_cat_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/cat
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/cat/cat.py
            ${CMAKE_CURRENT_BINARY_DIR}/cat/cat.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cat/cat.py
)

add_library(cat_op SHARED cat/cat_op.cpp)
target_include_directories(cat_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cat)
target_link_libraries(cat_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_index_target(cat_op)
add_dependencies(cat_op copy_index_cat_python)
set_target_properties(cat_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cat)

add_executable(test_cat cat/test_cat.cpp)
target_include_directories(test_cat PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_cat PRIVATE cat_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_index_target(test_cat)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_cat PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_cat copy_index_cat_python)
set_target_properties(test_cat PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cat)

# ==============================================================================
# nonzero operator
# ==============================================================================

add_custom_target(copy_index_nonzero_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/nonzero
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/nonzero/nonzero.py
            ${CMAKE_CURRENT_BINARY_DIR}/nonzero/nonzero.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nonzero/nonzero.py
)

add_library(nonzero_op SHARED nonzero/nonzero_op.cpp)
target_include_directories(nonzero_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/nonzero)
target_link_libraries(nonzero_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_index_target(nonzero_op)
add_dependencies(nonzero_op copy_index_nonzero_python)
set_target_properties(nonzero_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nonzero)

add_executable(test_nonzero nonzero/test_nonzero.cpp)
target_include_directories(test_nonzero PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_nonzero PRIVATE nonzero_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_index_target(test_nonzero)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_nonzero PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_nonzero copy_index_nonzero_python)
set_target_properties(test_nonzero PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/nonzero)

# ==============================================================================
# reshape_and_cache_flash operator
# ==============================================================================

add_custom_target(copy_index_reshape_and_cache_flash_python ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/reshape_and_cache_flash
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/reshape_and_cache_flash/reshape_and_cache_flash.py
            ${CMAKE_CURRENT_BINARY_DIR}/reshape_and_cache_flash/reshape_and_cache_flash.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/reshape_and_cache_flash/reshape_and_cache_flash.py
)

add_library(reshape_and_cache_flash_op SHARED reshape_and_cache_flash/reshape_and_cache_flash_op.cpp)
target_include_directories(reshape_and_cache_flash_op PRIVATE ${PROJECT_SOURCE_DIR}/include PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/reshape_and_cache_flash)
target_link_libraries(reshape_and_cache_flash_op PUBLIC Torch::Torch PRIVATE TritonJIT::triton_jit)
configure_index_target(reshape_and_cache_flash_op)
add_dependencies(reshape_and_cache_flash_op copy_index_reshape_and_cache_flash_python)
set_target_properties(reshape_and_cache_flash_op PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reshape_and_cache_flash)

add_executable(test_reshape_and_cache_flash reshape_and_cache_flash/test_reshape_and_cache_flash.cpp)
target_include_directories(test_reshape_and_cache_flash PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_reshape_and_cache_flash PRIVATE reshape_and_cache_flash_op TritonJIT::triton_jit TritonJIT::test_framework Torch::Torch)
configure_index_target(test_reshape_and_cache_flash)
if(ITTNOTIFY_LIBRARY)
    target_link_libraries(test_reshape_and_cache_flash PRIVATE "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed")
endif()
add_dependencies(test_reshape_and_cache_flash copy_index_reshape_and_cache_flash_python)
set_target_properties(test_reshape_and_cache_flash PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/reshape_and_cache_flash)
