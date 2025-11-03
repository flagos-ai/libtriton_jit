cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(TritonJIT LANGUAGES CUDA CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 20) # for C++20 Concepts
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
set(CMAKE_CUDA_STANDARD 17) # CUDA still uses C++17
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# It is also recommended to install them in a Python virtual environment
set(Python_FIND_VIRTUALENV FIRST)
if(NOT DEFINED ENV{VIRTUAL_ENV})
    message(WARNING "Warning: No Python virtual environment detected. Consider using a virtual environment for better package management.")
endif()


# --------------------------- RPATH settings ---------------------------
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
  set(_rpath_portable_origin "@loader_path")
else()
  set(_rpath_portable_origin $ORIGIN)
endif(APPLE)

# default Use separate rpaths during build and install phases
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
# Don't use the install-rpath during the build phase
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
# Automatically add all linked folders that are NOT in the build directory to
# the rpath (per library?)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --------------------------- project-wide settings ---------------------------
if (PROJECT_IS_TOP_LEVEL)
  message(STATUS "TritonJIT is the top level project")
endif()
option(TRITON_JIT_USE_EXTERNAL_JSON "whether to use external json library" OFF)
option(TRITON_JIT_USE_EXTERNAL_FMTLIB "whether to use external fmtlib" OFF)
option(TRITON_JIT_USE_EXTERNAL_PYBIND11 "whether to use external pybind11 library" ON)
option(TRITON_JIT_BUILD_EXAMPLES "whether to build examples" ${PROJECT_IS_TOP_LEVEL})
option(TRITON_JIT_BUILD_TESTS "whether to build tests" ${PROJECT_IS_TOP_LEVEL})
# a good practice for top-level project to control whether to install it as a dependency
option(TRITON_JIT_INSTALL "whether to install the packages" ${PROJECT_IS_TOP_LEVEL})
option(TRITON_JIT_ENABLE_NPU "Enable Ascend NPU support" OFF)


# --------------------------- dependencies ---------------------------
include(FetchContent)
# dependencies: cuda toolkit
find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)

# --------------------------- Ascend NPU Support (Optional) ---------------------------
if(TRITON_JIT_ENABLE_NPU)
  message(STATUS "Ascend NPU support is ENABLED")

  # Set default Ascend installation path
  set(ASCEND_INSTALL_PATH "/usr/local/Ascend" CACHE PATH "Ascend installation path")
  if(NOT EXISTS ${ASCEND_INSTALL_PATH})
    message(WARNING "Ascend installation not found at ${ASCEND_INSTALL_PATH}. NPU support may not work.")
  else()
    # Set Ascend CANN version and runtime path
    set(ASCEND_CANN_VERSION "8.2" CACHE STRING "Ascend CANN version")
    set(ASCEND_RUNTIME_PATH "${ASCEND_INSTALL_PATH}/ascend-toolkit/${ASCEND_CANN_VERSION}")

    # Check if the specified version exists, fallback to latest if not
    if(NOT EXISTS ${ASCEND_RUNTIME_PATH})
      set(ASCEND_RUNTIME_PATH "${ASCEND_INSTALL_PATH}/ascend-toolkit/latest")
      if(NOT EXISTS ${ASCEND_RUNTIME_PATH})
        set(ASCEND_RUNTIME_PATH "${ASCEND_INSTALL_PATH}/nnae/latest")
      endif()
    endif()

    if(EXISTS ${ASCEND_RUNTIME_PATH})
      message(STATUS "Using Ascend runtime path: ${ASCEND_RUNTIME_PATH}")

      # Detect architecture and set appropriate paths
      set(ASCEND_ARCH_PATH "${ASCEND_RUNTIME_PATH}/aarch64-linux")
      if(NOT EXISTS ${ASCEND_ARCH_PATH})
        set(ASCEND_ARCH_PATH "${ASCEND_RUNTIME_PATH}/x86_64-linux")
      endif()

      if(EXISTS ${ASCEND_ARCH_PATH})
        message(STATUS "Using Ascend architecture path: ${ASCEND_ARCH_PATH}")
        set(ASCEND_INCLUDE_DIR "${ASCEND_ARCH_PATH}/include")
        set(ASCEND_LIB_DIR "${ASCEND_ARCH_PATH}/lib64")
      else()
        set(ASCEND_INCLUDE_DIR "${ASCEND_RUNTIME_PATH}/include")
        set(ASCEND_LIB_DIR "${ASCEND_RUNTIME_PATH}/lib64")
      endif()

      # Add Ascend include directories
      include_directories(${ASCEND_INCLUDE_DIR})

      # Add Ascend library directories
      link_directories(${ASCEND_LIB_DIR})
      link_directories(${ASCEND_RUNTIME_PATH}/fwkacllib/lib64)

      # Find Ascend libraries
      find_library(ASCEND_RUNTIME_LIB ascendcl PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64)
      find_library(ASCEND_RUNTIME_API_LIB runtime PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/runtime/lib64)

      if(ASCEND_RUNTIME_LIB)
        message(STATUS "Found Ascend runtime library: ${ASCEND_RUNTIME_LIB}")
      endif()
      if(ASCEND_RUNTIME_API_LIB)
        message(STATUS "Found Ascend Runtime API library: ${ASCEND_RUNTIME_API_LIB}")
      endif()

      # Try to find torch_npu
      execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
        OUTPUT_VARIABLE TORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )

      if(TORCH_NPU_PATH)
        message(STATUS "Found torch_npu at: ${TORCH_NPU_PATH}")

        # Find the torch_npu library
        find_library(TORCH_NPU_LIB
          NAMES torch_npu
          PATHS "${TORCH_NPU_PATH}/lib"
          NO_DEFAULT_PATH
        )

        if(TORCH_NPU_LIB)
          message(STATUS "Found torch_npu library: ${TORCH_NPU_LIB}")
        endif()

        # Add torch_npu include paths if they exist
        set(TORCH_NPU_INCLUDE_PATHS)
        if(EXISTS "${TORCH_NPU_PATH}/csrc")
          list(APPEND TORCH_NPU_INCLUDE_PATHS "${TORCH_NPU_PATH}/csrc")
        endif()
        if(EXISTS "${TORCH_NPU_PATH}/include")
          list(APPEND TORCH_NPU_INCLUDE_PATHS "${TORCH_NPU_PATH}/include")
        endif()

        # Add library directory to link directories
        link_directories("${TORCH_NPU_PATH}/lib")
      else()
        message(WARNING "torch_npu not found. NPU support will be limited.")
      endif()

      # Add Ascend-specific compile definitions
      add_compile_definitions(ASCEND_PLATFORM)
      add_compile_definitions(TRITON_JIT_NPU_ENABLED)

      # Add a helper function to link Ascend libraries
      function(target_link_ascend_libraries target_name)
        if(ASCEND_RUNTIME_LIB)
          target_link_libraries(${target_name} PRIVATE ${ASCEND_RUNTIME_LIB})
        endif()
        if(ASCEND_RUNTIME_API_LIB)
          target_link_libraries(${target_name} PRIVATE ${ASCEND_RUNTIME_API_LIB})
        endif()
        if(TORCH_NPU_LIB)
          target_link_libraries(${target_name} PRIVATE ${TORCH_NPU_LIB})
        endif()

        # Set runtime path for the target
        if(ASCEND_LIB_DIR)
          set_target_properties(${target_name} PROPERTIES
            INSTALL_RPATH "${ASCEND_LIB_DIR}:${ASCEND_RUNTIME_PATH}/fwkacllib/lib64"
            BUILD_WITH_INSTALL_RPATH FALSE
          )
        endif()
      endfunction()
    endif()
  endif()
else()
  message(STATUS "Ascend NPU support is DISABLED")
endif()

# dependencies: python: we will use it for embeddeing the interpreter
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python site-packages path: ${Python_SITELIB}")
# dependencies: torch
# This is the FindTorch.cmake then it finds the TorchConfig.cmake provided by torch
find_package(Torch MODULE REQUIRED) # Note: it sets ABI

# dependencies: json
if (TRITON_JIT_USE_EXTERNAL_JSON)
  find_package(nlohmann_json VERSION 3.11.3 REQUIRED)
else()
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(json)
endif()

# dependencies: fmtlib
if(TRITON_JIT_USE_EXTERNAL_FMTLIB)
  find_package(fmt VERSION 10.2.1 REQUIRED)
else()
  FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(fmt)
  # so as to build a shared library that links libfmt.a
  set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
if(TRITON_JIT_USE_EXTERNAL_PYBIND11)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  find_package(pybind11 CONFIG REQUIRED)
else()
  FetchContent_Declare(pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(pybind11)
endif()
# --------------------------- subdirectories ---------------------------
add_subdirectory(src)
if(TRITON_JIT_BUILD_EXAMPLES)
  set(INSTALL_GTEST OFF) # we do not install tests
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.16.0
  )
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(examples)
endif()

# --------------------------- tests ---------------------------
if(TRITON_JIT_BUILD_TESTS)
  message(STATUS "Building tests")
  add_subdirectory(tests)
endif()

# --------------------------- cmake package ---------------------------
if(TRITON_JIT_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)
  # generate the config file that includes the exports: TritonJITConfig.cmake
  configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
  )
  # generate the config file that includes the version: TritonJITConfigVersion.cmake
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake"
    VERSION "${TritonJIT_VERSION_MAJOR}.${TritonJIT_VERSION_MINOR}.${TritonJIT_VERSION_PATCH}"
    COMPATIBILITY SameMajorVersion
  )
  # install the generated project cmake config & version file
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
  )
  # install the FindTorch module by us
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT)
endif()
