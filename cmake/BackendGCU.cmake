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
# GCU (Enflame) Backend Configuration
# ==============================================================================

message(STATUS "Configuring GCU backend...")

set(TOPS_ROOT "/opt/tops" CACHE PATH "Enflame TOPS SDK root directory")

if(NOT EXISTS ${TOPS_ROOT}/include/tops_runtime_api.h)
  message(FATAL_ERROR "TOPS SDK not found at ${TOPS_ROOT}. "
    "Please set -DTOPS_ROOT to the correct TOPS SDK installation path.")
endif()

find_library(TOPSRT_LIBRARY
  NAMES topsrt
  PATHS ${TOPS_ROOT}/lib
  NO_DEFAULT_PATH
)

if(NOT TOPSRT_LIBRARY)
  message(FATAL_ERROR "libtopsrt.so not found in ${TOPS_ROOT}/lib")
endif()

if(NOT TARGET GCU::efrt)
  add_library(GCU::efrt SHARED IMPORTED)
  set_target_properties(GCU::efrt PROPERTIES
    IMPORTED_LOCATION "${TOPSRT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TOPS_ROOT}/include"
  )
endif()

function(target_link_gcu_libraries target_name)
  target_include_directories(${target_name} PUBLIC ${TOPS_ROOT}/include)
  target_link_libraries(${target_name} PUBLIC GCU::efrt)
endfunction()

message(STATUS "GCU backend configuration complete")
message(STATUS "  TOPS SDK: ${TOPS_ROOT}")
message(STATUS "  libtopsrt: ${TOPSRT_LIBRARY}")
