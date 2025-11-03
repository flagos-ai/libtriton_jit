# Since the library links torch, we need to ensure the same ABI with pre-compiled torch
# libtorch installed from pypi is not built with cxx11 abi,
# if we split the library into several targets, we need to ensure that the ABI is the same
# with torch. It is a bit ugly to specify a target that does not depend on torch to use
# the cxx flags from torch, so we just merge then as one target, for simplicity
# then it can use the same cxx flags with public dependency transitivity
# --------------------------- triton jit function ---------------------------
add_library(triton_jit SHARED triton_jit_function_impl.cpp jit_utils.cpp)
target_include_directories(triton_jit
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(triton_jit
  PUBLIC Torch::Torch CUDA::cuda_driver fmt::fmt-header-only
  PRIVATE pybind11::embed)

# Get nlohmann_json include directory for public headers (header-only library)
get_target_property(JSON_INCLUDE_DIR nlohmann_json::nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(triton_jit
  PUBLIC
    ${JSON_INCLUDE_DIR})

# --------------------------- alias targets ---------------------------
# This is the target used in FetchContent, since FetchContent use add_subdirectory (a sub project build)
# So to use consistent target name, add namespace here
add_library(TritonJIT::triton_jit ALIAS triton_jit)

# --------------------------- install ---------------------------
if(TRITON_JIT_INSTALL)
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/triton_jit" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/scripts"
    DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/triton_jit/" FILES_MATCHING PATTERN "*.py")
  install(
    TARGETS triton_jit
    EXPORT TritonJITTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(
    EXPORT TritonJITTargets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
    NAMESPACE TritonJIT::
    FILE TritonJITTargets.cmake
  )
  # This is for find_package to work with the build tree, instead of the install tree
  export(EXPORT TritonJITTargets FILE "${PROJECT_BINARY_DIR}/TritonJITTargets.cmake")
endif()
