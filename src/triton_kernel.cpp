#include "triton_jit/triton_kernel.h"

#include <fstream>
#include <iostream>
#include <string>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "acl/acl.h"
#include "experiment/runtime/runtime/rt.h"
#include <unordered_map>

using json = nlohmann::json;

namespace triton_jit {
TritonKernel::TritonKernel(std::string_view dir, std::string_view kernel_name)
    : dir_(std::string(dir)), kernel_name_(std::string(kernel_name)), 
      mix_mode_("mix"), loaded_(false),
      bin_handle_(nullptr), fn_(nullptr) {
  std::string metadata_path = fmt::format("{}/{}.json", this->dir_, this->kernel_name_);
  std::ifstream f(metadata_path.c_str());
  
  if (!f.good()) {
    this->shared_ = 0;
    this->arch_ = 0;
    return;
  }
  
  json meta_data = json::parse(f);
  
  // Read shared memory configuration
  this->shared_ = meta_data.contains("shared") ? meta_data["shared"].get<int>() : 0;
  
  // Read mix_mode (determines binary format)
  this->mix_mode_ = meta_data.contains("mix_mode") ? meta_data["mix_mode"].get<std::string>() : "mix";
  
  this->arch_ = 0;
}

void TritonKernel::lazy_init_handle() const {
  if (this->loaded_) {
    return;
  }

  // Get current device ID
  int device_id = -1;
  aclError err = aclrtGetDevice(&device_id);
  if (err != ACL_SUCCESS) {
    device_id = 0;  // fallback
  }

  // Find kernel binary file (try .npubin, .o, .ttadapter, .bin)
  std::string rt_bin_path = fmt::format("{}/{}.npubin", this->dir_, this->kernel_name_);
  std::ifstream bin_file(rt_bin_path, std::ios::binary | std::ios::ate);
  
  if (!bin_file.good()) {
    std::vector<std::string> fallback_exts = {".o", ".ttadapter", ".bin"};
    bool file_found = false;
    
    for (const auto& ext : fallback_exts) {
      rt_bin_path = fmt::format("{}/{}{}", this->dir_, this->kernel_name_, ext);
      bin_file.open(rt_bin_path, std::ios::binary | std::ios::ate);
      if (bin_file.good()) {
        file_found = true;
        break;
      }
      bin_file.close();
      bin_file.clear();
    }
    
    if (!file_found) {
      throw std::runtime_error(fmt::format("Kernel binary not found: {}", 
                                          this->dir_ + "/" + this->kernel_name_));
    }
  }

  // Read binary file
  std::streamsize size = bin_file.tellg();
  if (size <= 0) {
    throw std::runtime_error(fmt::format("Invalid binary size: {}", rt_bin_path));
  }
  
  bin_file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  
  if (!bin_file.read(buffer.data(), size)) {
    throw std::runtime_error(fmt::format("Failed to read binary: {}", rt_bin_path));
  }
  bin_file.close();

  // Set device
  rtError_t rt_err = rtSetDevice(device_id);
  if (rt_err != RT_ERROR_NONE) {
    throw std::runtime_error(fmt::format("rtSetDevice failed for device {}, error: {}", 
                                        device_id, static_cast<int>(rt_err)));
  }

  // Register binary with RT API
  rtDevBinary_t binary;
  binary.data = buffer.data();
  binary.length = static_cast<uint32_t>(size);
  
  // Set magic value based on mix_mode
  binary.magic = (this->mix_mode_ == "aiv") ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
  binary.version = 0;

  void* rt_bin_handle = nullptr;
  rt_err = rtDevBinaryRegister(&binary, &rt_bin_handle);
  if (rt_err != RT_ERROR_NONE) {
    throw std::runtime_error(fmt::format("rtDevBinaryRegister failed: {}", 
                                        static_cast<int>(rt_err)));
  }

  // Create function stub
  static std::unordered_map<std::string, size_t> registered_names;
  static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;
  
  std::string stubName = this->kernel_name_;
  stubName += "_" + std::to_string(registered_names[this->kernel_name_]);
  registered_names[this->kernel_name_]++;
  
  auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
  void* func_stub_handle = registered.first->second.get();

  // Register function
  rt_err = rtFunctionRegister(rt_bin_handle, 
                             func_stub_handle,
                             stubName.c_str(),
                             (void*)this->kernel_name_.c_str(),
                             0);
  if (rt_err != RT_ERROR_NONE) {
    throw std::runtime_error(fmt::format("rtFunctionRegister failed: {}", 
                                        static_cast<int>(rt_err)));
  }
  
  // Store handles
  this->bin_handle_ = rt_bin_handle;
  this->fn_ = (aclrtFuncHandle)func_stub_handle;
  this->loaded_ = true;
}

void TritonKernel::launch(unsigned int grid_x,
  unsigned int grid_y,
  unsigned int grid_z,
  int num_warps,
  aclrtStream stream,
  void** args) const {
  this->lazy_init_handle();

  // Calculate block count
  uint32_t blockNum = grid_x * grid_y * grid_z;

  // Get system control address
  rtError_t ret;
  void *ffts_addr = NULL;
  uint32_t ffts_len;
  ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
  if (ret != RT_ERROR_NONE) {
    throw std::runtime_error(fmt::format("rtGetC2cCtrlAddr failed: {}", 
                                        static_cast<int>(ret)));
  }

  // Build kernel arguments structure
  struct __attribute__((packed)) KernelArgs {
    void* ffts_addr __attribute__((aligned(8)));
    void* syncBlockLock __attribute__((aligned(8)));
    void* workspace_addr __attribute__((aligned(8)));
    void* arg0 __attribute__((aligned(8)));
    void* arg1 __attribute__((aligned(8)));
    void* arg2 __attribute__((aligned(8)));
    int64_t arg3 __attribute__((aligned(8)));
    int32_t gridX __attribute__((aligned(4)));
    int32_t gridY __attribute__((aligned(4)));
    int32_t gridZ __attribute__((aligned(4)));
  };

  KernelArgs kernel_args;
  kernel_args.ffts_addr = ffts_addr;
  kernel_args.syncBlockLock = nullptr;
  kernel_args.workspace_addr = nullptr;

  if (args != nullptr) {
    kernel_args.arg0 = *reinterpret_cast<void**>(args[0]);
    kernel_args.arg1 = *reinterpret_cast<void**>(args[1]);
    kernel_args.arg2 = *reinterpret_cast<void**>(args[2]);
    kernel_args.arg3 = args[3] ? *reinterpret_cast<int64_t*>(args[3]) : 128;
  } else {
    memset(&kernel_args.arg0, 0, sizeof(void*) * 3 + sizeof(int64_t));
    kernel_args.arg3 = 128;
  }

  kernel_args.gridX = static_cast<int32_t>(grid_x);
  kernel_args.gridY = static_cast<int32_t>(grid_y);
  kernel_args.gridZ = static_cast<int32_t>(grid_z);

  // Launch kernel
  rtError_t rt_err = rtKernelLaunch(this->fn_,
                                    blockNum,
                                    static_cast<void*>(&kernel_args),
                                    sizeof(kernel_args),
                                    nullptr,
                                    stream);

  if (rt_err != RT_ERROR_NONE) {
    throw std::runtime_error(fmt::format("rtKernelLaunch failed: {}", 
                                        static_cast<int>(rt_err)));
  }
}
}  // namespace triton_jit
