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

# dependencies: torch
# use the current python interpreter's torch installation
if (NOT DEFINED Torch_ROOT)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
  execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch;print(torch.utils.cmake_prefix_path)"
                  OUTPUT_VARIABLE Torch_ROOT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  )
endif()
message(STATUS "Torch_ROOT: ${Torch_ROOT}")
find_package(Torch CONFIG REQUIRED)

# message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
# message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
# get_target_property(TMP_TORCH_LINKED_LIBRARIES torch INTERFACE_LINK_LIBRARIES)
# message(STATUS "torch linked libraries: ${TMP_TORCH_LINKED_LIBRARIES}")


# message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
# get_target_property(TMP_TORCH_INTERFACE_INCLUDE_DIR torch INTERFACE_INCLUDE_DIRECTORIES)
# message(STATUS "torch linked libraries: ${TMP_TORCH_INTERFACE_INCLUDE_DIR}")
# depending on targets other than cmake variables has better composability
# since TritonJIT publicly depends on torch, It is better to transitive(recursively) resolve
# all the dependencies when other targets dependens on TritonJIT::triton_jit
# But cmake variable have Direcory Scope, thus, those variables "TORCH_LIBRARIES" does not propagate
# So we would depend on the target instead. Using an alias is for better naming convention to have
# targets inside some namespace
if (NOT TARGET Torch::Torch)
  add_library(Torch::Torch INTERFACE IMPORTED)
  target_include_directories(Torch::Torch INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(Torch::Torch INTERFACE ${TORCH_LIBRARIES})
  target_compile_options(Torch::Torch INTERFACE ${TORCH_CXX_FLAGS})
  message(STATUS "torch interface cxx flag ${TORCH_CXX_FLAGS}")

  # add torch_python
  add_library(Torch::Torch_Python INTERFACE IMPORTED)
  find_library(torch_python_lib
    NAMES torch_python
    PATHS "${TORCH_INSTALL_PREFIX}/lib"
    REQUIRED)
  message(STATUS "find torch_python lib: ${torch_python_lib}")
  target_include_directories(Torch::Torch_Python INTERFACE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(Torch::Torch_Python INTERFACE ${torch_python_lib})
  target_compile_options(Torch::Torch_Python INTERFACE ${TORCH_CXX_FLAGS})
endif()

add_compile_options(${TORCH_CXX_FLAGS})
message(STATUS "Using ABI for the whole project from torch: ${TORCH_CXX_FLAGS}")
