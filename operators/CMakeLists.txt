# operators/CMakeLists.txt
# Main CMake configuration for all operators

# ==============================================================================
# Common utility functions
# ==============================================================================

# Function to configure backend-specific target settings
function(configure_operator_target TARGET_NAME)
    if(BACKEND STREQUAL "NPU")
        target_link_ascend_libraries(${TARGET_NAME})
        if(TORCH_NPU_INCLUDE_PATHS)
            target_include_directories(${TARGET_NAME} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
        endif()
    elseif(BACKEND STREQUAL "MUSA")
        target_link_musa_libraries(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    elseif(BACKEND STREQUAL "HCU")
        target_link_libraries(${TARGET_NAME} PRIVATE hip::host)
    elseif(BACKEND STREQUAL "IX")
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver CUDA::cudart)
    elseif(BACKEND STREQUAL "MLU")
        target_link_mlu_libraries(${TARGET_NAME})
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    else()  # CUDA
        target_link_libraries(${TARGET_NAME} PRIVATE CUDA::cuda_driver CUDA::cudart)
    endif()
endfunction()

# Function to create an operator library
function(create_operator_library)
    cmake_parse_arguments(OP "" "NAME;CATEGORY" "SOURCES;PYTHON_FILES" ${ARGN})

    set(LIB_NAME ${OP_NAME}_op)
    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${OP_CATEGORY}/${OP_NAME})

    # Create shared library
    add_library(${LIB_NAME} SHARED)

    foreach(SRC ${OP_SOURCES})
        target_sources(${LIB_NAME} PRIVATE ${SRC_DIR}/${SRC})
    endforeach()

    target_include_directories(${LIB_NAME}
        PRIVATE ${PROJECT_SOURCE_DIR}/include
        PUBLIC ${SRC_DIR}
    )

    target_link_libraries(${LIB_NAME}
        PUBLIC Torch::Torch
        PRIVATE TritonJIT::triton_jit
    )

    configure_operator_target(${LIB_NAME})

    # Copy Python files
    if(OP_PYTHON_FILES)
        set(COPY_TARGET copy_${OP_NAME}_python)
        add_custom_target(${COPY_TARGET} ALL)
        foreach(PY_FILE ${OP_PYTHON_FILES})
            add_custom_command(TARGET ${COPY_TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${SRC_DIR}/${PY_FILE}
                    ${CMAKE_CURRENT_BINARY_DIR}/${OP_CATEGORY}/${OP_NAME}/${PY_FILE}
            )
        endforeach()
        add_dependencies(${LIB_NAME} ${COPY_TARGET})
    endif()

    # Set output directory
    set_target_properties(${LIB_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${OP_CATEGORY}/${OP_NAME}
    )
endfunction()

# Function to create a test executable
function(create_operator_test)
    cmake_parse_arguments(TEST "" "NAME;CATEGORY" "SOURCES" ${ARGN})

    set(TEST_NAME test_${TEST_NAME})
    set(LIB_NAME ${TEST_NAME}_op)
    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_CATEGORY}/${TEST_NAME})

    # Note: TEST_NAME already has test_ prefix stripped in the actual call
    # We need to derive from the passed NAME
    string(REGEX REPLACE "^test_" "" OP_NAME "${TEST_NAME}")
    set(ACTUAL_LIB ${OP_NAME}_op)
    set(ACTUAL_TEST test_${OP_NAME})

    add_executable(${ACTUAL_TEST})

    foreach(SRC ${TEST_SOURCES})
        target_sources(${ACTUAL_TEST} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_CATEGORY}/${OP_NAME}/${SRC})
    endforeach()

    target_include_directories(${ACTUAL_TEST}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/common
    )

    target_link_libraries(${ACTUAL_TEST}
        PRIVATE
            ${ACTUAL_LIB}
            TritonJIT::triton_jit
            TritonJIT::test_framework
            Torch::Torch
    )

    configure_operator_target(${ACTUAL_TEST})

    # Handle ittnotify if available
    if(ITTNOTIFY_LIBRARY)
        target_link_libraries(${ACTUAL_TEST} PRIVATE
            "-Wl,--no-as-needed" ${ITTNOTIFY_LIBRARY} "-Wl,--as-needed"
        )
    endif()

    # Set output directory
    set_target_properties(${ACTUAL_TEST} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${TEST_CATEGORY}/${OP_NAME}
    )
endfunction()

# ==============================================================================
# Add subdirectories
# ==============================================================================

# Common test framework
add_subdirectory(common)

# Pointwise operators (supported on all backends)
add_subdirectory(pointwise)

# ==============================================================================
# Backend-specific operator support
# NPU (Ascend) backend has limited support for reduce/normalization operations
# due to BiShengHIR not supporting linalg.reduce operations
# ==============================================================================

# ==============================================================================
# NPU backend: Enable all operators for validation
# FlagGems Ascend has alternate implementations that may work on NPU
# Individual operators will be skipped if compilation fails (NPU_SKIP_OPS)
# ==============================================================================

# Reduction operators
add_subdirectory(reduce)

# Normalization operators
add_subdirectory(normalization)

# Fusion operators
add_subdirectory(fusion)

# Matrix multiplication operators (supported on all backends)
add_subdirectory(matmul)

# Indexing operators (supported on all backends)
add_subdirectory(index)
