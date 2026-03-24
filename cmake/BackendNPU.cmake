# ==============================================================================
# NPU (Ascend) Backend Configuration
# ==============================================================================

message(STATUS "Configuring NPU (Ascend) backend...")

# ------------------------------- Ascend Toolkit -------------------------------
set(ASCEND_TOOLKIT_HOME $ENV{ASCEND_TOOLKIT_HOME})
if(NOT ASCEND_TOOLKIT_HOME)
    set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(NOT EXISTS ${ASCEND_TOOLKIT_HOME})
    message(FATAL_ERROR "Ascend toolkit not found at ${ASCEND_TOOLKIT_HOME}. "
                        "Please set ASCEND_TOOLKIT_HOME environment variable.")
endif()
message(STATUS "ASCEND_TOOLKIT_HOME: ${ASCEND_TOOLKIT_HOME}")

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()

# ------------------------------- Find Ascend Libraries ------------------------
find_library(ASCENDCL_LIBRARY ascendcl
    PATHS ${ASCEND_TOOLKIT_HOME}/lib64
    NO_DEFAULT_PATH REQUIRED
)
find_library(ASCEND_RUNTIME_LIBRARY runtime
    PATHS ${ASCEND_TOOLKIT_HOME}/lib64
    NO_DEFAULT_PATH REQUIRED
)

message(STATUS "Found AscendCL: ${ASCENDCL_LIBRARY}")
message(STATUS "Found Ascend Runtime: ${ASCEND_RUNTIME_LIBRARY}")

# ------------------------------- Ascend Include Directories -------------------
# Find CANN installation for pkg_inc headers (contains rt.h and other runtime headers)
set(CANN_HOME $ENV{CANN_HOME})
if(NOT CANN_HOME)
    # Try to find CANN installation automatically
    file(GLOB CANN_CANDIDATES "/usr/local/Ascend/cann-*")
    if(CANN_CANDIDATES)
        list(GET CANN_CANDIDATES 0 CANN_HOME)
    endif()
endif()

set(ASCEND_INCLUDE_DIRS
    "${ASCEND_TOOLKIT_HOME}/include"
    "${ASCEND_TOOLKIT_HOME}/include/aclnn"
    "${ASCEND_TOOLKIT_HOME}/include/experiment"
    "${ASCEND_TOOLKIT_HOME}/include/experiment/runtime"
    "${ASCEND_TOOLKIT_HOME}/${ASCEND_ARCH_DIR}/include"
    "${ASCEND_TOOLKIT_HOME}/${ASCEND_ARCH_DIR}/include/experiment"
    "${ASCEND_TOOLKIT_HOME}/${ASCEND_ARCH_DIR}/include/experiment/msprof"
)

# Add CANN pkg_inc path for rt.h and other runtime headers
if(CANN_HOME AND EXISTS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc")
    list(APPEND ASCEND_INCLUDE_DIRS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc")
    message(STATUS "Found CANN pkg_inc: ${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc")
    # Also add runtime/runtime for config.h and other deps
    if(EXISTS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc/runtime/runtime")
        list(APPEND ASCEND_INCLUDE_DIRS "${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc/runtime/runtime")
        message(STATUS "Found CANN runtime headers: ${CANN_HOME}/${ASCEND_ARCH_DIR}/pkg_inc/runtime/runtime")
    endif()
endif()

# ------------------------------- Create Imported Targets ----------------------
add_library(Ascend::ascendcl SHARED IMPORTED)
set_target_properties(Ascend::ascendcl PROPERTIES
    IMPORTED_LOCATION ${ASCENDCL_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_INCLUDE_DIRS}"
)

add_library(Ascend::runtime SHARED IMPORTED)
set_target_properties(Ascend::runtime PROPERTIES
    IMPORTED_LOCATION ${ASCEND_RUNTIME_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES "${ASCEND_INCLUDE_DIRS}"
)

# ------------------------------- torch_npu Integration ------------------------
# Hardcoded path to libtorch_npu
set(TORCH_NPU_PATH "/data/baai_user_home/chwork/compile_triton/pytorch/libtorch_npu")

if(TORCH_NPU_PATH AND EXISTS "${TORCH_NPU_PATH}")
    message(STATUS "Found torch_npu at: ${TORCH_NPU_PATH}")

    # Get base path (parent directory)
    get_filename_component(TORCH_NPU_BASE_PATH "${TORCH_NPU_PATH}" DIRECTORY)
    message(STATUS "torch_npu base path: ${TORCH_NPU_BASE_PATH}")

    # torch_npu include paths
    set(TORCH_NPU_INCLUDE_PATHS "")
    foreach(_subdir csrc include)
        if(EXISTS "${TORCH_NPU_PATH}/${_subdir}")
            list(APPEND TORCH_NPU_INCLUDE_PATHS "${TORCH_NPU_PATH}/${_subdir}")
        endif()
    endforeach()

    # ACL headers from torch_npu third_party
    if(EXISTS "${TORCH_NPU_PATH}/include/third_party/acl/inc")
        list(APPEND TORCH_NPU_INCLUDE_PATHS "${TORCH_NPU_PATH}/include/third_party/acl/inc")
    endif()

    # Find torch_npu library
    find_library(TORCH_NPU_LIB torch_npu
        PATHS "${TORCH_NPU_PATH}/lib"
        NO_DEFAULT_PATH
    )
    if(TORCH_NPU_LIB)
        message(STATUS "Found torch_npu library: ${TORCH_NPU_LIB}")
    endif()

    link_directories("${TORCH_NPU_PATH}/lib")

    # Cache for subdirectories
    set(TORCH_NPU_INCLUDE_PATHS "${TORCH_NPU_INCLUDE_PATHS}" CACHE INTERNAL "torch_npu include paths")
    set(TORCH_NPU_BASE_PATH "${TORCH_NPU_BASE_PATH}" CACHE INTERNAL "torch_npu base path")
else()
    message(WARNING "torch_npu not found at ${TORCH_NPU_PATH}")
endif()

# ------------------------------- torch.libs for gfortran -----------------------
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), '..', 'torch.libs'))"
    OUTPUT_VARIABLE TORCH_LIBS_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_LIBS_PATH AND EXISTS "${TORCH_LIBS_PATH}")
    message(STATUS "Found torch.libs directory: ${TORCH_LIBS_PATH}")
    link_directories("${TORCH_LIBS_PATH}")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${TORCH_LIBS_PATH}")
endif()

# ------------------------------- Helper Function ------------------------------
function(target_link_ascend_libraries target_name)
    target_link_libraries(${target_name} PRIVATE Ascend::ascendcl Ascend::runtime)
    target_include_directories(${target_name} PRIVATE ${ASCEND_INCLUDE_DIRS})

    if(TORCH_NPU_INCLUDE_PATHS)
        target_include_directories(${target_name} PRIVATE ${TORCH_NPU_INCLUDE_PATHS})
    endif()
    if(TORCH_NPU_PATH)
        target_include_directories(${target_name} PRIVATE ${TORCH_NPU_PATH})
    endif()
    if(TORCH_NPU_BASE_PATH)
        target_include_directories(${target_name} PRIVATE ${TORCH_NPU_BASE_PATH})
    endif()
    if(TORCH_NPU_LIB)
        target_link_libraries(${target_name} PRIVATE ${TORCH_NPU_LIB})
    endif()
endfunction()

message(STATUS "NPU backend configuration complete")
