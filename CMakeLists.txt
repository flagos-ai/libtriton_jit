cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(TritonJIT LANGUAGES CUDA CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 17) # for fold-expression
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
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
# a good practice for top-level project to control whether to install it as a dependency
option(TRITON_JIT_INSTALL "whether to install the packages" ${PROJECT_IS_TOP_LEVEL})


# --------------------------- dependencies ---------------------------
include(FetchContent)
# dependencies: cuda toolkit
find_package(CUDAToolkit REQUIRED COMPONENTS cuda_driver)
# dependencies: python: we will use it for embeddeing the interpreter
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python site-packages path: ${Python_SITELIB}")
# dependencies: torch
# This is the FindTorch.cmake then it finds the TorchConfig.cmake provided by torch
find_package(Torch MODULE REQUIRED) # Note: it sets ABI

# dependencies: json
if (TRITON_JIT_USE_EXTERNAL_JSON)
  find_package(nlohmann_json 3.11.3 REQUIRED)
else()
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
  )
  FetchContent_MakeAvailable(json)
endif()

# dependencies: fmtlib
if(TRITON_JIT_USE_EXTERNAL_FMTLIB)
  find_package(fmt 10.2.1 REQUIRED)
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
if(TRITON_JIT_INSTALL)
  include(GNUInstallDirs)
endif()
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

# --------------------------- cmake package ---------------------------
if(TRITON_JIT_INSTALL)
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
