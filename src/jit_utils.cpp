#include "triton_jit/jit_utils.h"

#include <dlfcn.h>  // dladdr
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace triton_jit {
std::filesystem::path get_path_of_this_library() {
  // This function gives the library path of this library as runtime, similar to the $ORIGIN
  // that is used for run path (RPATH), but unfortunately, for custom dependencies (instead of linking)
  // there is no build system generator to take care of this.
  static const std::filesystem::path cached_path = []() {
    Dl_info dl_info;
    if (dladdr(reinterpret_cast<void*>(&get_path_of_this_library), &dl_info) && dl_info.dli_fname) {
      return std::filesystem::canonical(dl_info.dli_fname);  // Ensure absolute, resolved path
    } else {
      throw std::runtime_error("cannot get the path of libjit_utils.so");
    }
  }();
  return cached_path;
}

std::filesystem::path get_script_dir() {
  const static std::filesystem::path script_dir = []() {
    std::filesystem::path installed_script_dir =
        get_path_of_this_library().parent_path().parent_path() / "share" / "triton_jit" / "scripts";

    if (std::filesystem::exists(installed_script_dir)) {
      return installed_script_dir;
    } else {
      std::filesystem::path source_script_dir =
          std::filesystem::path(__FILE__).parent_path().parent_path() / "scripts";
      return source_script_dir;
    }
  }();
  return script_dir;
}

std::filesystem::path get_home_directory() {
  const static std::filesystem::path home_dir = []() {
#ifdef _WIN32
    const char* home_dir_path = std::getenv("USERPROFILE");
#else
    const char* home_dir_path = std::getenv("HOME");
#endif
    return std::filesystem::path(home_dir_path);
  }();
  return home_dir;
}

#if !defined(BACKEND_NPU) && !defined(BACKEND_MUSA) && !defined(BACKEND_KUNLUNXIN)
void ensure_cuda_context() {
  CUcontext pctx;
  checkCudaErrors(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    CUdevice device_index;
    checkCudaErrors(cuDeviceGet(&device_index, /*ordinal*/ 0));
    checkCudaErrors(cuDevicePrimaryCtxRetain(&pctx, device_index));
    checkCudaErrors(cuCtxSetCurrent(pctx));
  }
}
#endif
}  // namespace triton_jit
