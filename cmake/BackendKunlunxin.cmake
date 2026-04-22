# ==============================================================================
# Kunlunxin (Baidu XPU3) Backend Configuration
# ==============================================================================

message(STATUS "Configuring Kunlunxin backend...")

# Find Kunlunxin XPU runtime library and headers
# xpu/runtime.h provides: XPUStream, XPUFunc, xpu_strerror, xpu_current_device,
#                          KL3_BEGIN, KL4_BEGIN, KL5_BEGIN, etc.
# Honor XRE3_HOME / XPU_HOME / XPU_SDK_ROOT environment variables.
# Typical layouts:
#   $XRE3_HOME/include/xpu/runtime.h , $XRE3_HOME/so/libxpurt.so
#   $XPU_HOME/include/xpu/runtime.h  , $XPU_HOME/lib/libxpurt.so
find_path(KUNLUNXIN_RUNTIME_INCLUDE_DIR
    NAMES xpu/runtime.h
    HINTS
        ENV XRE3_HOME
        ENV XPU_HOME
        ENV XPU_SDK_ROOT
        ${CONDA_ENV_ROOT}
    PATH_SUFFIXES include
    PATHS
        /usr/include
        /usr/local/include
)

find_library(KUNLUNXIN_RUNTIME_LIBRARY
    NAMES xpurt xpu_runtime
    HINTS
        ENV XRE3_HOME
        ENV XPU_HOME
        ENV XPU_SDK_ROOT
        ${CONDA_ENV_ROOT}
    PATH_SUFFIXES so lib lib64
    PATHS
        /usr/lib
        /usr/local/lib
)

if(KUNLUNXIN_RUNTIME_INCLUDE_DIR)
    message(STATUS "Found Kunlunxin runtime headers: ${KUNLUNXIN_RUNTIME_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Kunlunxin runtime headers (xpu/runtime.h) not found. "
                        "Set XPU_HOME or XPU_SDK_ROOT environment variable.")
endif()

if(KUNLUNXIN_RUNTIME_LIBRARY)
    message(STATUS "Found Kunlunxin runtime library: ${KUNLUNXIN_RUNTIME_LIBRARY}")
else()
    message(WARNING "Kunlunxin runtime library (libxpurt) not found. "
                    "Linking may fail at build time.")
endif()

# Create imported target
if(NOT TARGET Kunlunxin::runtime)
    add_library(Kunlunxin::runtime UNKNOWN IMPORTED)
    set_target_properties(Kunlunxin::runtime PROPERTIES
        IMPORTED_LOCATION "${KUNLUNXIN_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${KUNLUNXIN_RUNTIME_INCLUDE_DIR}"
    )
endif()

message(STATUS "Kunlunxin backend configuration complete")
