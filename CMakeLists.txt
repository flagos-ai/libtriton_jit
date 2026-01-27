cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# ==============================================================================
# Backend Selection (must be before project())
# ==============================================================================
set(BACKEND "CUDA" CACHE STRING "Target backend: CUDA, IX, MUSA, or NPU")
set_property(CACHE BACKEND PROPERTY STRINGS "CUDA" "IX" "MUSA" "NPU")

if(NOT BACKEND MATCHES "^(CUDA|IX|MUSA|NPU)$")
    message(FATAL_ERROR "Invalid BACKEND: ${BACKEND}. Must be CUDA, IX, MUSA, or NPU")
endif()
message(STATUS "Building with backend: ${BACKEND}")

# Project definition (NPU and MUSA don't need CUDA language)
if(BACKEND STREQUAL "NPU")
    project(TritonJIT LANGUAGES CXX VERSION 0.1.0)
elseif(BACKEND STREQUAL "MUSA")
    project(TritonJIT LANGUAGES CXX VERSION 0.1.0)
else()
    project(TritonJIT LANGUAGES CUDA CXX VERSION 0.1.0)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ==============================================================================
# Compile Definitions for Backend
# ==============================================================================
if(BACKEND STREQUAL "CUDA")
    add_compile_definitions(BACKEND_CUDA)
elseif(BACKEND STREQUAL "IX")
    add_compile_definitions(BACKEND_IX)
elseif(BACKEND STREQUAL "NPU")
    add_compile_definitions(BACKEND_NPU)
elseif(BACKEND STREQUAL "MUSA")
    add_compile_definitions(BACKEND_MUSA)
endif()

# String macro for Python runtime
# Map MUSA to mtgpu for Triton backend name
if(BACKEND STREQUAL "MUSA")
    set(TRITON_BACKEND_NAME "mtgpu")
else()
    set(TRITON_BACKEND_NAME "${BACKEND}")
endif()
add_compile_definitions(BACKEND_NAME="${TRITON_BACKEND_NAME}")

# ==============================================================================
# Project-wide Settings
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT (BACKEND STREQUAL "NPU" OR BACKEND STREQUAL "MUSA"))
    set(CMAKE_CUDA_STANDARD 17)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Python_FIND_VIRTUALENV FIRST)
if(NOT DEFINED ENV{VIRTUAL_ENV})
    message(WARNING "No Python virtual environment detected. Consider using a virtual environment.")
endif()

# ==============================================================================
# Mirror Configuration (for faster download in China)
# ==============================================================================
option(USE_CHINA_MIRRORS "Use Chinese mirrors for faster download" OFF)
set(GITHUB_MIRROR_BASE "https://githubfast.com" CACHE STRING "GitHub mirror base URL")

macro(github_url OUTPUT_VAR REPO_PATH)
    if(USE_CHINA_MIRRORS)
        set(${OUTPUT_VAR} "${GITHUB_MIRROR_BASE}/${REPO_PATH}")
    else()
        set(${OUTPUT_VAR} "https://github.com/${REPO_PATH}")
    endif()
endmacro()

# ==============================================================================
# RPATH Settings
# ==============================================================================
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(_rpath_portable_origin "@loader_path")
else()
    set(_rpath_portable_origin $ORIGIN)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${_rpath_portable_origin}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ==============================================================================
# Project Options
# ==============================================================================
if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "TritonJIT is the top level project")
endif()

option(TRITON_JIT_USE_EXTERNAL_JSON "Use external json library" OFF)
option(TRITON_JIT_USE_EXTERNAL_FMTLIB "Use external fmtlib" OFF)
option(TRITON_JIT_USE_EXTERNAL_PYBIND11 "Use external pybind11 library" ON)
option(TRITON_JIT_BUILD_EXAMPLES "Build examples" ${PROJECT_IS_TOP_LEVEL})
option(TRITON_JIT_INSTALL "Install the packages" ${PROJECT_IS_TOP_LEVEL})

# ==============================================================================
# Dependencies: Python & Torch (common to all backends)
# ==============================================================================
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python: ${Python_EXECUTABLE}")
message(STATUS "Python site-packages: ${Python_SITELIB}")

find_package(Torch MODULE REQUIRED)

# Find Intel ITT Notify library (required by PyTorch)
# Get conda prefix from Python executable path
get_filename_component(PYTHON_BIN_DIR ${Python_EXECUTABLE} DIRECTORY)
get_filename_component(CONDA_ENV_ROOT ${PYTHON_BIN_DIR} DIRECTORY)

find_library(ITTNOTIFY_LIBRARY
    NAMES ittnotify libittnotify
    PATHS
        ${CONDA_ENV_ROOT}
        ${Python_SITELIB}/../..
        ENV CONDA_PREFIX
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)
# Fallback to default paths if not found
if(NOT ITTNOTIFY_LIBRARY)
    find_library(ITTNOTIFY_LIBRARY NAMES ittnotify libittnotify)
endif()
if(ITTNOTIFY_LIBRARY)
    message(STATUS "Found ittnotify library: ${ITTNOTIFY_LIBRARY}")
else()
    message(WARNING "ittnotify library not found - test executables may fail to link")
endif()

# ==============================================================================
# Backend-specific Dependencies (modular)
# ==============================================================================
if(BACKEND STREQUAL "CUDA" OR BACKEND STREQUAL "IX")
    include(BackendCUDA)
elseif(BACKEND STREQUAL "NPU")
    include(BackendNPU)
elseif(BACKEND STREQUAL "MUSA")
    include(BackendMUSA)
endif()

# ==============================================================================
# Dependencies: Third-party Libraries
# ==============================================================================
# --- nlohmann_json ---
if(TRITON_JIT_USE_EXTERNAL_JSON)
    find_package(nlohmann_json VERSION 3.11.3 REQUIRED)
else()
    github_url(JSON_URL "nlohmann/json/releases/download/v3.11.3/json.tar.xz")
    FetchContent_Declare(json URL ${JSON_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(json)
endif()

# --- fmtlib ---
if(TRITON_JIT_USE_EXTERNAL_FMTLIB)
    find_package(fmt VERSION 10.2.1 REQUIRED)
else()
    github_url(FMT_URL "fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz")
    FetchContent_Declare(fmt URL ${FMT_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(fmt)
    set_target_properties(fmt PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# --- pybind11 ---
if(TRITON_JIT_USE_EXTERNAL_PYBIND11)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_package(pybind11 CONFIG REQUIRED)
else()
    github_url(PYBIND11_URL "pybind/pybind11")
    FetchContent_Declare(pybind11 GIT_REPOSITORY ${PYBIND11_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
    FetchContent_MakeAvailable(pybind11)
endif()

# ==============================================================================
# Subdirectories
# ==============================================================================
add_subdirectory(src)

# Examples directory removed - using operators/ instead
# if(TRITON_JIT_BUILD_EXAMPLES)
#     set(INSTALL_GTEST OFF)
#     github_url(GTEST_URL "google/googletest/archive/refs/tags/v1.16.0.tar.gz")
#     FetchContent_Declare(googletest URL ${GTEST_URL} DOWNLOAD_EXTRACT_TIMESTAMP ON)
#     FetchContent_MakeAvailable(googletest)
#     add_subdirectory(examples)
# endif()

# ==============================================================================
# Operators with Test Framework
# ==============================================================================
option(TRITON_JIT_BUILD_OPERATORS "Build operators with test framework" ${PROJECT_IS_TOP_LEVEL})
if(TRITON_JIT_BUILD_OPERATORS)
    add_subdirectory(operators)
endif()

# ==============================================================================
# CMake Package Installation
# ==============================================================================
if(TRITON_JIT_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
    )

    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake"
        VERSION "${TritonJIT_VERSION_MAJOR}.${TritonJIT_VERSION_MINOR}.${TritonJIT_VERSION_PATCH}"
        COMPATIBILITY SameMajorVersion
    )

    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/TritonJITConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
    )

    install(
        FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTorch.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonJIT
    )
endif()
