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
# MLU (Cambricon MLU) Backend Configuration
# Copyright 2026, FlagOs Contributors.
# ==============================================================================

message(STATUS "Configuring MLU backend...")

# ------------------------------- Neuware SDK --------------------------------
set(NEUWARE_HOME $ENV{NEUWARE_HOME})
if(NOT NEUWARE_HOME)
  set(NEUWARE_HOME "/usr/local/neuware")
endif()
if(NOT EXISTS "${NEUWARE_HOME}/include/cn_api.h")
  message(FATAL_ERROR "Neuware SDK not found at ${NEUWARE_HOME}/include/cn_api.h. "
                      "Please set NEUWARE_HOME to your neuware install root.")
endif()

set(ENV{NEUWARE_HOME} "${NEUWARE_HOME}")
message(STATUS "NEUWARE_HOME: ${NEUWARE_HOME}")

# ------------------------------- TorchMLU -----------------------------------
if(NOT DEFINED TorchMLU_ROOT)
  execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch_mlu,os;print(os.path.dirname(torch_mlu.__file__))"
    OUTPUT_VARIABLE _TORCH_MLU_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch_mlu;print(torch_mlu.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TorchMLU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NOT EXISTS "${TorchMLU_ROOT}/TorchMLU/TorchMLUConfig.cmake"
     AND NOT EXISTS "${TorchMLU_ROOT}/TorchMLUConfig.cmake")
    if(EXISTS "${_TORCH_MLU_DIR}/csrc/share/cmake/TorchMLU/TorchMLUConfig.cmake")
      set(TorchMLU_DIR "${_TORCH_MLU_DIR}/csrc/share/cmake/TorchMLU")
      message(STATUS "TorchMLUConfig.cmake not under reported prefix; "
                     "falling back to TorchMLU_DIR=${TorchMLU_DIR}")
    endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)

find_package(TorchMLU CONFIG REQUIRED)

if(TorchMLU_FOUND)
  message(STATUS "Found MLU Runtime.")

  set(_TORCH_MLU_INCLUDE_DIRS "")
  foreach(_dir IN LISTS TORCH_MLU_INCLUDE_DIRS)
    if(IS_DIRECTORY "${_dir}")
      list(APPEND _TORCH_MLU_INCLUDE_DIRS "${_dir}")
    else()
      message(STATUS "Skipping non-existent MLU include dir: ${_dir}")
    endif()
  endforeach()
  list(APPEND _TORCH_MLU_INCLUDE_DIRS "${NEUWARE_HOME}/include")

  set(_TORCH_MLU_LIBRARIES "")
  foreach(_lib IN LISTS TORCH_MLU_LIBRARIES)
    if(_lib MATCHES "-NOTFOUND$")
      message(STATUS "Skipping missing MLU library: ${_lib}")
    else()
      list(APPEND _TORCH_MLU_LIBRARIES "${_lib}")
    endif()
  endforeach()

  if(NOT TARGET MLU::mlu_runtime)
    add_library(MLU::mlu_runtime INTERFACE IMPORTED)
    set_target_properties(MLU::mlu_runtime PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${_TORCH_MLU_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${_TORCH_MLU_LIBRARIES}"
    )
    message(STATUS "Created MLU::mlu_runtime imported target")
  endif()
else()
  message(FATAL_ERROR "MLU Runtime not found. Please ensure cntoolkit is installed.")
endif()

function(target_link_mlu_libraries target_name)
  target_include_directories(${target_name} PRIVATE ${_TORCH_MLU_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE ${_TORCH_MLU_LIBRARIES})
endfunction()

message(STATUS "MLU backend configuration complete")
