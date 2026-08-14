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
# MACA (MetaX) Backend Configuration
# ==============================================================================

message(STATUS "Configuring MACA backend...")

if(NOT EXISTS "${MACA_PATH}")
    message(FATAL_ERROR "MACA SDK not found at ${MACA_PATH}. Please set MACA_PATH.")
endif()
message(STATUS "MACA_PATH: ${MACA_PATH}")

find_path(MACA_INCLUDE_DIR
    NAMES mcr/mc_runtime.h
    PATHS "${MACA_PATH}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(MACA_RUNTIME_LIB
    NAMES mcruntime
    PATHS "${MACA_PATH}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "Found MACA include: ${MACA_INCLUDE_DIR}")
message(STATUS "Found MACA runtime: ${MACA_RUNTIME_LIB}")

if(NOT TARGET MACA::mcruntime)
    add_library(MACA::mcruntime SHARED IMPORTED)
    set_target_properties(MACA::mcruntime PROPERTIES
        IMPORTED_LOCATION "${MACA_RUNTIME_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${MACA_INCLUDE_DIR};${MACA_INCLUDE_DIR}/mcr"
    )
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib")

function(target_link_maca_libraries target_name)
    target_link_libraries(${target_name} PRIVATE MACA::mcruntime)
endfunction()

message(STATUS "MACA backend configuration complete")
