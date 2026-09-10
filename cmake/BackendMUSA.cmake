# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ==============================================================================
# MUSA (Moore Threads) Backend Configuration
# ==============================================================================

message(STATUS "Configuring MUSA backend...")

# ------------------------------- MUSA Runtime Detection -----------------------
# Find MUSA Runtime (Driver/Runtime API only, no torch_musa C++ headers)
if(NOT DEFINED MUSA_HOME)
    set(MUSA_HOME "/usr/local/musa")
endif()

message(STATUS "MUSA_HOME: ${MUSA_HOME}")

# Find MUSA Runtime library
find_library(MUSA_RUNTIME_LIB musart
    PATHS "${MUSA_HOME}/lib64" "${MUSA_HOME}/lib"
    NO_DEFAULT_PATH
)

# Find MUSA Driver library
# Real driver library is usually installed by system driver (like /usr/lib/x86_64-linux-gnu)
# Stubs are only for compile-time symbol resolution
find_library(MUSA_DRIVER_LIB musa
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/lib64
        /usr/lib
        "${MUSA_HOME}/lib64"
        "${MUSA_HOME}/lib"
    NO_DEFAULT_PATH
)

# If not found, try system paths (will find stub as last resort)
if(NOT MUSA_DRIVER_LIB)
    find_library(MUSA_DRIVER_LIB musa)
endif()

# Find MUSA include directory
find_path(MUSA_INCLUDE_DIR musa_runtime.h
    PATHS "${MUSA_HOME}/include"
    NO_DEFAULT_PATH
)

if(MUSA_RUNTIME_LIB AND MUSA_DRIVER_LIB AND MUSA_INCLUDE_DIR)
    message(STATUS "Found MUSA Runtime: ${MUSA_RUNTIME_LIB}")
    message(STATUS "Found MUSA Driver: ${MUSA_DRIVER_LIB}")
    message(STATUS "Found MUSA Include: ${MUSA_INCLUDE_DIR}")

    # Create MUSA::musa_runtime imported target
    # GLOBAL so that consumers (e.g. FlagFFT) added via add_subdirectory
    # can link against it from the parent scope.
    if(NOT TARGET MUSA::musa_runtime)
        add_library(MUSA::musa_runtime INTERFACE IMPORTED GLOBAL)
        set_target_properties(MUSA::musa_runtime PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MUSA_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${MUSA_RUNTIME_LIB};${MUSA_DRIVER_LIB}"
        )
        message(STATUS "Created MUSA::musa_runtime imported target")
    endif()
else()
    message(FATAL_ERROR "MUSA Runtime not found at ${MUSA_HOME}. Please ensure MUSA toolkit is installed.")
endif()

# ------------------------------- torch_musa Runtime Check ---------------------
# Check if torch_musa Python module is available (needed at runtime for device registration)
# NOTE: We do NOT include torch_musa C++ headers to avoid conflicts with PyTorch headers
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch_musa; print('available')"
    OUTPUT_VARIABLE TORCH_MUSA_CHECK
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(TORCH_MUSA_CHECK STREQUAL "available")
    message(STATUS "torch_musa Python module detected (runtime device registration available)")
    set(HAS_TORCH_MUSA_RUNTIME TRUE CACHE INTERNAL "torch_musa runtime available")
    add_compile_definitions(HAS_TORCH_MUSA_RUNTIME)
else()
    message(STATUS "torch_musa Python module not detected")
    message(STATUS "Note: You will need 'import torch_musa' in Python to use MUSA devices")
    set(HAS_TORCH_MUSA_RUNTIME FALSE CACHE INTERNAL "torch_musa runtime not available")
endif()

# ------------------------------- Helper Function ------------------------------
function(target_link_musa_libraries target_name)
    message(STATUS "Linking target ${target_name} with MUSA libraries")
    target_link_libraries(${target_name} PRIVATE MUSA::musa_runtime)
endfunction()

message(STATUS "MUSA backend configuration complete")
