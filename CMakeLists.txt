cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(TritonJIT LANGUAGES CXX VERSION 0.1.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

# ------------------------------- project-wide settings -------------------------------
set(CMAKE_CXX_STANDARD 17) # for fold-expression
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Ensures only standard-compliant C++ is used
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --------------------------- Mirror Configuration ---------------------------
option(USE_CHINA_MIRRORS "Use Chinese mirrors for faster download" OFF)
set(GITHUB_MIRROR_BASE "https://githubfast.com" CACHE STRING "GitHub mirror base URL")

# It is also recommended to install them in a Python virtual environment
set(Python_FIND_VIRTUALENV FIRST)
if(NOT DEFINED ENV{VIRTUAL_ENV})
    message(WARNING "Warning: No Python virtual environment detected. Consider using a virtual environment for better package management.")
endif()

# --------------------------- Ascend Runtime Environment ---------------------------
# Set default Ascend installation path
set(ASCEND_INSTALL_PATH "/usr/local/Ascend" CACHE PATH "Ascend installation path")
if(NOT EXISTS ${ASCEND_INSTALL_PATH})
    message(FATAL_ERROR "Ascend installation not found at ${ASCEND_INSTALL_PATH}. Please set ASCEND_INSTALL_PATH correctly.")
endif()

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

if(NOT EXISTS ${ASCEND_RUNTIME_PATH})
    message(FATAL_ERROR "Ascend runtime path not found. Checked paths: ${ASCEND_INSTALL_PATH}/ascend-toolkit/${ASCEND_CANN_VERSION}, ${ASCEND_INSTALL_PATH}/ascend-toolkit/latest, ${ASCEND_INSTALL_PATH}/nnae/latest")
endif()

message(STATUS "Using Ascend runtime path: ${ASCEND_RUNTIME_PATH}")

# Detect architecture and set appropriate paths
set(ASCEND_ARCH_PATH "${ASCEND_RUNTIME_PATH}/aarch64-linux")
if(NOT EXISTS ${ASCEND_ARCH_PATH})
    # Fallback to x86_64 if aarch64 not found
    set(ASCEND_ARCH_PATH "${ASCEND_RUNTIME_PATH}/x86_64-linux")
endif()

if(EXISTS ${ASCEND_ARCH_PATH})
    message(STATUS "Using Ascend architecture path: ${ASCEND_ARCH_PATH}")
    # Use architecture-specific paths
    set(ASCEND_INCLUDE_DIR "${ASCEND_ARCH_PATH}/include")
    set(ASCEND_LIB_DIR "${ASCEND_ARCH_PATH}/lib64")
else()
    # Use symbolic links in root directory
    set(ASCEND_INCLUDE_DIR "${ASCEND_RUNTIME_PATH}/include")
    set(ASCEND_LIB_DIR "${ASCEND_RUNTIME_PATH}/lib64")
endif()

# Add Ascend include directories
include_directories(${ASCEND_INCLUDE_DIR})
include_directories(${ASCEND_INCLUDE_DIR}/graph)
include_directories(${ASCEND_INCLUDE_DIR}/ge)
include_directories(${ASCEND_RUNTIME_PATH}/compiler/include)
include_directories(${ASCEND_RUNTIME_PATH}/fwkacllib/include)

# Add Ascend library directories
link_directories(${ASCEND_LIB_DIR})
link_directories(${ASCEND_LIB_DIR}/stub)
link_directories(${ASCEND_RUNTIME_PATH}/fwkacllib/lib64)
link_directories(${ASCEND_RUNTIME_PATH}/atc/lib64)

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
set(FETCHCONTENT_QUIET OFF)

# (CUDA dependency removed)
# dependencies: Ascend Runtime libraries
find_library(ASCEND_RUNTIME_LIB ascendcl PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64 REQUIRED)
find_library(ASCEND_ACL_OP_LIB acl_op_compiler PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64 REQUIRED)
find_library(ASCEND_GE_LIB ge_runner PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64)
find_library(ASCEND_GRAPH_LIB graph PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64)
find_library(ASCEND_RUNTIME_LIB_STUB ascendcl PATHS ${ASCEND_LIB_DIR}/stub NO_DEFAULT_PATH)

# Additional CANN 8.2 specific libraries
find_library(ASCEND_ACL_LIB acl PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64)
find_library(ASCEND_RUNTIME_API_LIB runtime PATHS ${ASCEND_LIB_DIR} ${ASCEND_RUNTIME_PATH}/runtime/lib64)
find_library(ASCEND_ATC_LIB atc PATHS ${ASCEND_RUNTIME_PATH}/atc/lib64)

# Check if required Ascend libraries are found
if(NOT ASCEND_RUNTIME_LIB)
    message(FATAL_ERROR "Ascend runtime library (ascendcl) not found in ${ASCEND_LIB_DIR} or ${ASCEND_RUNTIME_PATH}/fwkacllib/lib64")
endif()

message(STATUS "Found Ascend runtime library: ${ASCEND_RUNTIME_LIB}")
if(ASCEND_ACL_OP_LIB)
    message(STATUS "Found Ascend ACL op compiler library: ${ASCEND_ACL_OP_LIB}")
endif()
if(ASCEND_GE_LIB)
    message(STATUS "Found Ascend GE library: ${ASCEND_GE_LIB}")
endif()
if(ASCEND_GRAPH_LIB)
    message(STATUS "Found Ascend Graph library: ${ASCEND_GRAPH_LIB}")
endif()
if(ASCEND_ACL_LIB)
    message(STATUS "Found Ascend ACL library: ${ASCEND_ACL_LIB}")
endif()
if(ASCEND_RUNTIME_API_LIB)
    message(STATUS "Found Ascend Runtime API library: ${ASCEND_RUNTIME_API_LIB}")
endif()
if(ASCEND_ATC_LIB)
    message(STATUS "Found Ascend ATC library: ${ASCEND_ATC_LIB}")
endif()
if(ASCEND_RUNTIME_LIB_STUB)
    message(STATUS "Found Ascend runtime stub library: ${ASCEND_RUNTIME_LIB_STUB}")
endif()

# dependencies: python: we will use it for embeddeing the interpreter
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python site-packages path: ${Python_SITELIB}")
# dependencies: torch
# This is the FindTorch.cmake then it finds the TorchConfig.cmake provided by torch
find_package(Torch MODULE REQUIRED) # Note: it sets ABI

# Verify torch is built with Ascend support
if(TORCH_FOUND)
    message(STATUS "Found Torch version: ${TORCH_VERSION}")
    # Get ABI setting from torch
    get_target_property(TORCH_CXX_FLAGS torch INTERFACE_COMPILE_DEFINITIONS)
    message(STATUS "Torch ABI settings: ${TORCH_CXX_FLAGS}")
    
    # Try to find torch_npu
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch_npu; import os; print(os.path.dirname(torch_npu.__file__))"
        OUTPUT_VARIABLE TORCH_NPU_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
	
    set(TORCH_NPU_PATH /data/baai_user_home/chwork/compile_triton/pytorch/libtorch_npu)

    if(TORCH_NPU_PATH)
        message(STATUS "Found torch_npu at: ${TORCH_NPU_PATH}")
        
        get_filename_component(TORCH_NPU_BASE_PATH "${TORCH_NPU_PATH}" DIRECTORY)
        message(STATUS "torch_npu base path: ${TORCH_NPU_BASE_PATH}")

        # Find the main torch_npu library
        find_library(TORCH_NPU_LIB 
            NAMES torch_npu
            PATHS "${TORCH_NPU_PATH}/lib"
            NO_DEFAULT_PATH
        )
        
        # Find the torch_npu Python extension
        find_library(TORCH_NPU_C_LIB 
            NAMES _C.cpython-310-aarch64-linux-gnu
            PATHS "${TORCH_NPU_PATH}"
            NO_DEFAULT_PATH
        )
        
        if(TORCH_NPU_LIB)
            message(STATUS "Found torch_npu library: ${TORCH_NPU_LIB}")
        endif()
        
        if(TORCH_NPU_C_LIB)
            message(STATUS "Found torch_npu C extension: ${TORCH_NPU_C_LIB}")
        endif()
        
        if(NOT TORCH_NPU_LIB AND NOT TORCH_NPU_C_LIB)
            message(WARNING "torch_npu libraries not found")
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
        message(WARNING "torch_npu not found or not importable")
    endif()
endif()

# dependencies: json
if (TRITON_JIT_USE_EXTERNAL_JSON)
  find_package(nlohmann_json VERSION 3.11.3 REQUIRED)
else()
  if(USE_CHINA_MIRRORS)
    FetchContent_Declare(json
      URL ${GITHUB_MIRROR_BASE}/nlohmann/json/releases/download/v3.11.3/json.tar.xz
      DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
  else()
    FetchContent_Declare(json
      URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
      DOWNLOAD_EXTRACT_TIMESTAMP ON
    )
  endif()
  FetchContent_MakeAvailable(json)
endif()

# dependencies: fmtlib
if(TRITON_JIT_USE_EXTERNAL_FMTLIB)
  find_package(fmt VERSION 10.2.1 REQUIRED)
else()
  if(USE_CHINA_MIRRORS)
    FetchContent_Declare(fmt
      URL ${GITHUB_MIRROR_BASE}/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP ON
      SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/fmt-src)
  else()
    FetchContent_Declare(fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt
      GIT_TAG e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
  endif()
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
  if(USE_CHINA_MIRRORS)
    FetchContent_Declare(pybind11
      GIT_REPOSITORY ${GITHUB_MIRROR_BASE}/pybind/pybind11
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
  else()
    FetchContent_Declare(pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11
      DOWNLOAD_EXTRACT_TIMESTAMP ON)
  endif()
  FetchContent_MakeAvailable(pybind11)
endif()

# --------------------------- Ascend-specific compiler flags ---------------------------
# Add Ascend-specific compile definitions
add_compile_definitions(ASCEND_PLATFORM)
add_compile_definitions(ASCEND_CANN_VERSION_82)

# Note: ABI compatibility is handled by torch - do not override
# PyTorch sets _GLIBCXX_USE_CXX11_ABI=1, which should be compatible with modern Ascend
# If you encounter ABI issues, check your PyTorch and Ascend versions compatibility

# Set compiler flags for Ascend compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++17")
    # Add Ascend-specific flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DASCEND_PLATFORM -D__FILENAME__='\"$(notdir $(abspath $<))\"'")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Set environment variables for runtime
set(ENV{ASCEND_INSTALL_PATH} ${ASCEND_INSTALL_PATH})
set(ENV{ASCEND_RUNTIME_PATH} ${ASCEND_RUNTIME_PATH})
set(ENV{LD_LIBRARY_PATH} "${ASCEND_LIB_DIR}:${ASCEND_RUNTIME_PATH}/fwkacllib/lib64:$ENV{LD_LIBRARY_PATH}")

# Add a helper function to link Ascend libraries
function(target_link_ascend_libraries target_name)
    target_link_libraries(${target_name} PRIVATE ${ASCEND_RUNTIME_LIB})
    if(ASCEND_ACL_LIB)
        target_link_libraries(${target_name} PRIVATE ${ASCEND_ACL_LIB})
    endif()
    if(ASCEND_ACL_OP_LIB)
        target_link_libraries(${target_name} PRIVATE ${ASCEND_ACL_OP_LIB})
    endif()
    if(ASCEND_GE_LIB)
        target_link_libraries(${target_name} PRIVATE ${ASCEND_GE_LIB})
    endif()
    if(ASCEND_GRAPH_LIB)
        target_link_libraries(${target_name} PRIVATE ${ASCEND_GRAPH_LIB})
    endif()
    if(ASCEND_RUNTIME_API_LIB)
        target_link_libraries(${target_name} PRIVATE ${ASCEND_RUNTIME_API_LIB})
    endif()

    # Set runtime path for the target
    set_target_properties(${target_name} PROPERTIES
        INSTALL_RPATH "${ASCEND_LIB_DIR}:${ASCEND_RUNTIME_PATH}/fwkacllib/lib64"
        BUILD_WITH_INSTALL_RPATH FALSE
    )
endfunction()

# --------------------------- Fix gfortran linking issue ---------------------------
# Add torch.libs directory to library path to resolve gfortran dependencies
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), '..', 'torch.libs'))"
    OUTPUT_VARIABLE TORCH_LIBS_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_LIBS_PATH AND EXISTS "${TORCH_LIBS_PATH}")
    message(STATUS "Found torch.libs directory: ${TORCH_LIBS_PATH}")
    link_directories("${TORCH_LIBS_PATH}")

    # Add to RPATH for runtime linking
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${TORCH_LIBS_PATH}")
else()
    # Fallback to known path
    set(TORCH_LIBS_PATH "/root/miniconda3/envs/triton/lib/python3.10/site-packages/torch.libs")
    if(EXISTS "${TORCH_LIBS_PATH}")
        message(STATUS "Using fallback torch.libs directory: ${TORCH_LIBS_PATH}")
        link_directories("${TORCH_LIBS_PATH}")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${TORCH_LIBS_PATH}")
    endif()
endif()

# Find and link gfortran library to resolve openblas dependency issue
find_library(GFORTRAN_LIB
    NAMES gfortran libgfortran
    PATHS
        /usr/lib64
        /usr/lib
        /lib64
        /lib
        /root/miniconda3/envs/triton/lib
    NO_DEFAULT_PATH
)

if(GFORTRAN_LIB)
    message(STATUS "Found gfortran library: ${GFORTRAN_LIB}")
else()
    message(WARNING "System gfortran library not found")
endif()

# Add conda lib directory to link directories to help find gfortran
link_directories(/root/miniconda3/envs/triton/lib)

# --------------------------- subdirectories ---------------------------
add_subdirectory(src)
if(TRITON_JIT_BUILD_EXAMPLES)
  set(INSTALL_GTEST OFF) # we do not install tests
  if(USE_CHINA_MIRRORS)
    FetchContent_Declare(
      googletest
      URL ${GITHUB_MIRROR_BASE}/google/googletest/archive/refs/tags/v1.16.0.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP ON
      SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/googletest-src
    )
  else()
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.16.0
    )
  endif()
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(examples)
endif()

message("TRITON_JIT_INSTALL: ${TRITON_JIT_INSTALL}") # ON
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
