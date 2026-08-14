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
# HCU Backend Configuration
# ==============================================================================

message(STATUS "Configuring HCU backend...")

# HCU uses HIP runtime
# Find HIP package - typically provided by DTK installation
list(APPEND CMAKE_PREFIX_PATH
    "$ENV{ROCM_PATH}"
    "$ENV{HIP_PATH}"
    "/opt/rocm"
    "/opt/rocm/hip"
    "/opt/dtk"
    "/opt/dtk/hip"
)

# DTK's MIOpen depends on /usr/lib/x86_64-linux-gnu/librt.so in its CMake target,
# but on Ubuntu 22.04+ (glibc 2.34) librt was merged into libc and the stub
# symlink librt.so is no longer shipped. Create it manually if needed:
#   sudo ln -s /usr/lib/x86_64-linux-gnu/librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so
# Or opt in to automatic creation at configure time (requires write access to /usr/lib):
option(HCU_CREATE_LIBRT_STUB
    "Create /usr/lib/x86_64-linux-gnu/librt.so symlink for DTK MIOpen (requires root)"
    OFF)
if(HCU_CREATE_LIBRT_STUB
   AND NOT EXISTS "/usr/lib/x86_64-linux-gnu/librt.so"
   AND EXISTS "/usr/lib/x86_64-linux-gnu/librt.so.1")
    message(STATUS "HCU_CREATE_LIBRT_STUB=ON: creating librt.so symlink (required by DTK MIOpen)")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            /usr/lib/x86_64-linux-gnu/librt.so.1
            /usr/lib/x86_64-linux-gnu/librt.so
        RESULT_VARIABLE _librt_symlink_result
    )
    if(_librt_symlink_result EQUAL 0)
        message(STATUS "Created /usr/lib/x86_64-linux-gnu/librt.so -> librt.so.1")
    else()
        message(WARNING
            "Failed to create /usr/lib/x86_64-linux-gnu/librt.so -> librt.so.1 "
            "(exit code ${_librt_symlink_result}). "
            "Build may fail with 'No rule to make target librt.so'. "
            "Fix manually with: "
            "sudo ln -s /usr/lib/x86_64-linux-gnu/librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so")
    endif()
elseif(NOT EXISTS "/usr/lib/x86_64-linux-gnu/librt.so"
       AND EXISTS "/usr/lib/x86_64-linux-gnu/librt.so.1")
    message(WARNING
        "Missing /usr/lib/x86_64-linux-gnu/librt.so (required by DTK MIOpen on glibc 2.34+). "
        "Create it manually with: "
        "sudo ln -s /usr/lib/x86_64-linux-gnu/librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so "
        "or reconfigure with -DHCU_CREATE_LIBRT_STUB=ON")
endif()

find_package(hip REQUIRED)
message(STATUS "Found HIP: ${hip_VERSION}")

# The HIP runtime (libgalaxyhip.so) carries RUNPATH=/opt/dtk/lib which
# contains a bundled libunwind.so.8 (from hipprof_utils).  CMake warns
# about an RPATH conflict with the identical-SONAME system libunwind in
# /usr/lib/x86_64-linux-gnu.  The DTK compiler already treats several
# /opt/dtk subdirectories as implicit link dirs; extend this to the parent
# /opt/dtk/lib so CMake omits it from RPATH (the HIP library's own RUNPATH
# still provides it at runtime).
if(DEFINED ENV{ROCM_PATH})
    file(REAL_PATH "$ENV{ROCM_PATH}/lib" _rocm_lib_real)
    list(APPEND CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES
        "$ENV{ROCM_PATH}/lib" "${_rocm_lib_real}")
endif()

get_target_property(_torch_hip_opts torch_hip INTERFACE_COMPILE_OPTIONS)
if(_torch_hip_opts)
    # torch_hip sets -std=c++17 in INTERFACE_COMPILE_OPTIONS, which overrides
    # our C++20 standard. Strip it so C++20 concepts work.
    # See Caffe2Targets.cmake for more details.
    list(FILTER _torch_hip_opts EXCLUDE REGEX "-std=c\\+\\+17")
    # -Wno-duplicate-decl-specifier is valid for C/ObjC only; GCC warns when
    # passed to C++ compilation.  Strip it from the interface flags.
    # See Caffe2Targets.cmake for more details.
    list(FILTER _torch_hip_opts EXCLUDE REGEX "-Wno-duplicate-decl-specifier")
    set_property(TARGET torch_hip PROPERTY INTERFACE_COMPILE_OPTIONS ${_torch_hip_opts})
endif()

# PyTorch pip wheels bundle libibverbs inside torch.libs/ with hashed SONAMEs
# (e.g. libnl-3-04364822.so.200.26.0) that are not on the linker search path.
# The linker follows libibverbs's DT_NEEDED chain and fails to resolve them at
# link time, even though libtorch_hip.so's $ORIGIN/../../torch.libs RPATH
# resolves them correctly at runtime.  Suppress the link-time error.
add_link_options(-Wl,--allow-shlib-undefined)

# ------------------------------- Helper Function ------------------------------
function(target_link_hcu_libraries target_name)
    target_link_libraries(${target_name} PRIVATE hip::host)
endfunction()

message(STATUS "HCU backend configuration complete")
