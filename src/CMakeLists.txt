# =============================================================================
# Triton JIT Library - NPU Backend
# =============================================================================
# Note: Merged into single target to ensure ABI compatibility with PyTorch
# (PyPI libtorch is not built with cxx11 ABI)

# -------------------- Library Target --------------------
add_library(triton_jit SHARED
  triton_jit_function.cpp
  jit_utils.cpp
  triton_kernel.cpp
)

# -------------------- Ascend NPU Setup --------------------
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
  set(ASCEND_HOME_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
  set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
message(STATUS "Ascend toolkit: ${ASCEND_HOME_PATH}")

# -------------------- Include & Link Directories --------------------
target_include_directories(triton_jit
  PUBLIC
    # Project headers
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    # Ascend headers
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

target_link_directories(triton_jit PUBLIC
  ${ASCEND_HOME_PATH}/lib64
)

# -------------------- Dependencies --------------------
target_link_libraries(triton_jit
  PUBLIC
    Torch::Torch                  # PyTorch C++ API
    fmt::fmt-header-only          # String formatting
    runtime                       # Ascend RT
    ascendcl                      # Ascend CL
  PRIVATE
    nlohmann_json::nlohmann_json  # JSON parsing
    pybind11::embed               # Python embedding
)

# -------------------- Alias Target --------------------
add_library(TritonJIT::triton_jit ALIAS triton_jit)

# -------------------- Installation --------------------
if(TRITON_JIT_INSTALL)
  # Headers
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/triton_jit"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  
  # Python scripts
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/triton_jit/"
    FILES_MATCHING PATTERN "*.py")
  
  # Library
  install(TARGETS triton_jit
    EXPORT TritonJITTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  
  # CMake config
  install(EXPORT TritonJITTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
    NAMESPACE TritonJIT::
    FILE TritonJITTargets.cmake)
  
  # Export for build tree
  export(EXPORT TritonJITTargets
    FILE "${PROJECT_BINARY_DIR}/TritonJITTargets.cmake")
endif()
