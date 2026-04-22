#pragma once

// Baidu Kunlunxin (KL3/XPU3) backend.
// Aligned with Triton third_party/xpu backend (driver.c, driver.py, compiler.py).
// Uses native xpuLaunchKernel + manually constructed xpu_kernel handle.

#include <xpu/runtime.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"
#include "triton_jit/backend_policy.h"
#include "triton_jit/jit_utils.h"
#include "triton_jit/kernel_metadata.h"

namespace triton_jit {

// No xpuLaunchKernel forward declaration needed - we use the lower-level XRE3 API:
//   xpu_launch_config(nclusters, ncores, stream)
//   xpu_launch_argument_set(arg_ptr, size, offset)
//   xpu_launch_async(func)
// which are all exported by libxpurt.so.

// ---- xpu_kernel struct (exact copy of driver.c layout) ----

enum KunlunxinKernelType {
  KT_CLUSTER = 0,
  KT_SDCDNN = 1,
};

enum KunlunxinKernelPlace {
  KP_CPU = 0,
  KP_XPU = 1,
};

struct KunlunxinKernelDesc {
  uint32_t type : 16;
  uint32_t place : 16;
  uint64_t code_addr;
  uint32_t code_byte_size;
  uint32_t code_pc;
  uint32_t param_dword_size;
  uint64_t hash;
  const char* name;
  void* rt_private;
  uint64_t printf_buffer_offset;
};

// ---- Kunlunxin param type enum (aligned with driver.py generate_kernel_params) ----

enum KunlunxinParamType : int64_t {
  KLX_PARAM_UNKNOWN = 0,
  KLX_PARAM_PTR_FP32 = 1,
  KLX_PARAM_PTR_F64 = 2,
  KLX_PARAM_PTR_INT32 = 3,
  KLX_PARAM_PTR_INT64 = 4,
  KLX_PARAM_BOOL = 5,
  KLX_PARAM_PTR_FP16 = 6,
  KLX_PARAM_PTR_BF16 = 7,
  KLX_PARAM_INT32 = 8,
  KLX_PARAM_INT64 = 9,
  KLX_PARAM_INT8 = 10,
  KLX_PARAM_INT16 = 11,
};

struct KunlunxinArgInfo {
  std::string type;
  bool is_pointer;
  int64_t type_enum;
  size_t byte_size;
};

inline KunlunxinParamType sig_to_param_type(const std::string& sig_type) {
  // libtriton_jit naming: "*fp32", "*fp16", "*bf16", "*fp64", "*i32", "*i64",
  //                       "i1", "i8", "i16", "i32", "i64".
  if (sig_type == "*fp32") return KLX_PARAM_PTR_FP32;
  if (sig_type == "*fp64" || sig_type == "*f64") return KLX_PARAM_PTR_F64;
  if (sig_type == "*i32") return KLX_PARAM_PTR_INT32;
  if (sig_type == "*i64") return KLX_PARAM_PTR_INT64;
  if (sig_type == "*fp16") return KLX_PARAM_PTR_FP16;
  if (sig_type == "*bf16") return KLX_PARAM_PTR_BF16;
  if (sig_type == "i1") return KLX_PARAM_BOOL;
  if (sig_type == "i32") return KLX_PARAM_INT32;
  if (sig_type == "i64") return KLX_PARAM_INT64;
  if (sig_type == "i8") return KLX_PARAM_INT8;
  if (sig_type == "i16") return KLX_PARAM_INT16;
  return KLX_PARAM_UNKNOWN;
}

inline size_t sig_to_byte_size(const std::string& sig_type) {
  if (!sig_type.empty() && sig_type[0] == '*') return sizeof(void*);
  if (sig_type == "i1") return sizeof(bool);
  if (sig_type == "i8") return 1;
  if (sig_type == "i16") return 2;
  if (sig_type == "i32" || sig_type == "fp32") return 4;
  if (sig_type == "i64" || sig_type == "fp64" || sig_type == "f64") return 8;
  return 8;
}

// Parse "*fp32:16,*fp16,i64,1024" -> per-arg info (constexpr tokens are numeric
// literals, parse_kunlunxin_signature just records them as UNKNOWN/byte_size=8
// since they are NOT part of the runtime kernel params).
inline std::vector<KunlunxinArgInfo> parse_kunlunxin_signature(const std::string& sig) {
  std::vector<KunlunxinArgInfo> result;
  if (sig.empty()) return result;
  std::stringstream ss(sig);
  std::string token;
  while (std::getline(ss, token, ',')) {
    auto colon = token.find(':');
    std::string base = (colon != std::string::npos) ? token.substr(0, colon) : token;
    KunlunxinArgInfo info;
    info.type = base;
    info.is_pointer = (!base.empty() && base[0] == '*');
    info.type_enum = static_cast<int64_t>(sig_to_param_type(base));
    info.byte_size = sig_to_byte_size(base);
    result.push_back(info);
  }
  return result;
}

// ---- Kunlunxin metadata (superset of GpuKernelMeta) ----

struct KunlunxinKernelMetadata {
  unsigned int shared = 0;       // placeholder; xpu ignores
  unsigned int xpu_arch = 3;     // 3 = KL3, 4 = KL4, 5 = KL5
  bool is_sdnn = false;
  std::string mangled_name;      // metadata["name"] from compiler.py
  uint64_t printf_buf_offset = 0;
};

inline KunlunxinKernelMetadata load_kunlunxin_metadata(const std::string& dir,
                                                      const std::string& kernel_name) {
  KunlunxinKernelMetadata meta;
  meta.mangled_name = kernel_name;  // fallback
  std::string path = fmt::format("{}/{}.json", dir, kernel_name);
  std::ifstream f(path);
  if (!f.is_open()) {
    LOG(WARNING) << "Kunlunxin metadata not found: " << path;
    return meta;
  }
  try {
    nlohmann::json j = nlohmann::json::parse(f);
    if (j.contains("xpu_arch")) {
      meta.xpu_arch = j["xpu_arch"].get<unsigned int>();
    } else if (j.contains("target") && j["target"].contains("arch")) {
      // target.arch may be a string like "xpu3" or an int
      const auto& a = j["target"]["arch"];
      if (a.is_number_integer()) {
        meta.xpu_arch = a.get<unsigned int>();
      } else if (a.is_string()) {
        std::string s = a.get<std::string>();
        if (s.size() > 3 && s.rfind("xpu", 0) == 0) {
          try { meta.xpu_arch = std::stoul(s.substr(3)); } catch (...) {}
        }
      }
    }
    meta.is_sdnn = j.value("is_sdnn", false);
    meta.printf_buf_offset = j.value("printf_buf_offset", uint64_t{0});
    if (j.contains("name")) {
      meta.mangled_name = j["name"].get<std::string>();
    }
    if (j.contains("shared")) {
      int s = j["shared"].get<int>();
      meta.shared = (s < 0) ? 0u : static_cast<unsigned int>(s);
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << fmt::format("Failed to parse Kunlunxin metadata {}: {}", path, e.what());
  }
  return meta;
}

// Match driver.c checksum(): signed-char sum into uint32
inline uint32_t kunlunxin_checksum(const char* data, size_t len) {
  uint32_t crc = 0;
  for (size_t i = 0; i < len; ++i) {
    crc += static_cast<uint32_t>(data[i]);  // keep signed-char semantics
  }
  return crc;
}

// (num_clusters, num_cores) per (arch, is_sdnn)
inline std::pair<int, int> get_xpu_spec(unsigned int arch, bool is_sdnn) {
  switch (arch) {
    case 2: return is_sdnn ? std::make_pair(8, 8)  : std::make_pair(8, 64);
    case 3: return is_sdnn ? std::make_pair(12, 8) : std::make_pair(12, 64);
    case 4: return is_sdnn ? std::make_pair(6, 8)  : std::make_pair(12, 64);
    default:
      return std::make_pair(12, 64);
  }
}

// ---- Backend ----

struct KunlunxinBackend {
  using StreamType = XPUStream;
  using ContextType = void*;
  using KernelHandle = XPUFunc;

  // Kunlunxin has no warp concept; let block_x = num_warps * 1.
  static constexpr unsigned int WARP_SIZE = 1;

  struct LaunchOptions {
    unsigned int shared_memory = 0;
    int nclusters = 12;
    int ncores = 64;
    std::string kernel_name;  // mangled name
    std::string signature;
    size_t num_args = 0;
  };

  struct ModuleData {
    std::vector<char> binary_data;  // must outlive the XPUFunc (code_addr points here)
    XPUFunc func = nullptr;
    KunlunxinKernelMetadata metadata;
  };

  static inline std::unordered_map<std::string, ModuleData> module_cache_;
  static inline std::mutex cache_mutex_;

  static LaunchOptions prepare_launch(const std::string& dir,
                                      const std::string& name,
                                      unsigned int shared_mem,
                                      const std::string& sig,
                                      size_t num_args) {
    LaunchOptions opts;
    opts.shared_memory = shared_mem;
    opts.signature = sig;
    opts.num_args = num_args;
    opts.kernel_name = name;
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = module_cache_.find(fmt::format("{}::{}", dir, name));
    if (it != module_cache_.end()) {
      const auto& md = it->second.metadata;
      auto [ncl, ncr] = get_xpu_spec(md.xpu_arch, md.is_sdnn);
      opts.nclusters = ncl;
      opts.ncores = ncr;
      opts.kernel_name = md.mangled_name;
    } else {
      opts.nclusters = 12;
      opts.ncores = 64;
    }
    return opts;
  }

  static void launch_kernel(XPUStream stream,
                            XPUFunc kernel,
                            unsigned grid_x,
                            unsigned grid_y,
                            unsigned grid_z,
                            unsigned /*block_x*/,
                            unsigned /*block_y*/,
                            unsigned /*block_z*/,
                            void** args,
                            const LaunchOptions& opts) {
    if (grid_x == 0 || grid_y == 0 || grid_z == 0) return;

    auto layout = parse_kunlunxin_signature(opts.signature);

    // Keep only entries whose type_enum is known (skip constexpr numeric tokens).
    std::vector<size_t> kept_indices;
    kept_indices.reserve(layout.size());
    for (size_t i = 0; i < layout.size(); ++i) {
      if (layout[i].type_enum != KLX_PARAM_UNKNOWN) {
        kept_indices.push_back(i);
      }
    }
    size_t n_runtime = kept_indices.empty() ? opts.num_args : kept_indices.size();

    // Step 1: configure launch (nclusters, ncores, stream)
    int ret = xpu_launch_config(opts.nclusters, opts.ncores, stream);
    if (ret != XPU_SUCCESS) {
      const char* err = xpu_strerror(ret);
      throw std::runtime_error(
          fmt::format("xpu_launch_config failed: {} (err={})", err ? err : "?", ret));
    }

    // Step 2: set kernel arguments sequentially.
    // xpu_launch_argument_set(ptr, size, offset) copies `size` bytes from *ptr
    // into the XPU parameter block at byte offset `offset`.
    // offset must be 4-byte aligned; size is rounded up to 4 internally.
    size_t offset = 0;
    for (size_t k = 0; k < n_runtime; ++k) {
      void* arg_ptr = args[k];   // args[k] points to the argument value
      size_t byte_sz = 8;        // default: 8 bytes (pointer)
      if (!kept_indices.empty()) {
        byte_sz = layout[kept_indices[k]].byte_size;
      }
      // Align offset to at least 4 bytes, and to byte_sz for natural alignment.
      size_t align = std::max(byte_sz, size_t(4));
      offset = (offset + align - 1) & ~(align - 1);
      ret = xpu_launch_argument_set(arg_ptr, byte_sz, offset);
      if (ret != XPU_SUCCESS) {
        const char* err = xpu_strerror(ret);
        throw std::runtime_error(
            fmt::format("xpu_launch_argument_set failed at arg {}: {} (err={})",
                        k, err ? err : "?", ret));
      }
      offset += byte_sz;
    }

    // Step 3: launch asynchronously
    ret = xpu_launch_async(kernel);
    if (ret != XPU_SUCCESS) {
      const char* err = xpu_strerror(ret);
      throw std::runtime_error(
          fmt::format("xpu_launch_async failed: {} (err={})", err ? err : "?", ret));
    }
  }

  static void ensure_context() {
    // XPU runtime initializes on first API call; nothing to do here.
  }

  static int get_device_index() {
    int dev = 0;
    int ret = xpu_current_device(&dev);
    if (ret != XPU_SUCCESS) {
      const char* err = xpu_strerror(ret);
      throw std::runtime_error(
          fmt::format("xpu_current_device failed: {}", err ? err : "?"));
    }
    return dev;
  }

  static XPUFunc load_kernel(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);

    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.func;
    }

    KunlunxinKernelMetadata meta = load_kunlunxin_metadata(dir, kernel_name);

    LOG(INFO) << fmt::format(
        "[kunlunxin] loading kernel {} arch={} is_sdnn={} mangled={} "
        "printf_buf_offset=0x{:x}",
        kernel_name, meta.xpu_arch, meta.is_sdnn, meta.mangled_name,
        meta.printf_buf_offset);

    std::string bin_path = fmt::format("{}/{}.xpubin", dir, kernel_name);
    std::ifstream ifs(bin_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
      throw std::runtime_error(fmt::format("open failed: {}", bin_path));
    }
    auto sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> bin(static_cast<size_t>(sz));
    if (!ifs.read(bin.data(), sz)) {
      throw std::runtime_error(fmt::format("read failed: {}", bin_path));
    }

    // Cache first so code_addr points to a stable address.
    auto& cached = module_cache_[key];
    cached.binary_data = std::move(bin);
    cached.metadata = std::move(meta);

    auto* kern = new KunlunxinKernelDesc();
    kern->type = cached.metadata.is_sdnn ? KT_SDCDNN : KT_CLUSTER;
    kern->place = KP_CPU;
    kern->code_addr = reinterpret_cast<uint64_t>(cached.binary_data.data());
    kern->code_byte_size = static_cast<uint32_t>(cached.binary_data.size());
    kern->code_pc = 0;
    kern->param_dword_size = 0;
    kern->hash = kunlunxin_checksum(cached.binary_data.data(), cached.binary_data.size());
    kern->name = cached.metadata.mangled_name.c_str();
    kern->rt_private = nullptr;
    kern->printf_buffer_offset = cached.metadata.printf_buf_offset;

    cached.func = reinterpret_cast<XPUFunc>(kern);
    return cached.func;
  }

  static unsigned int get_shared_memory(const std::string& dir, const std::string& kernel_name) {
    std::string key = fmt::format("{}::{}", dir, kernel_name);
    std::lock_guard<std::mutex> lock(cache_mutex_);
    auto it = module_cache_.find(key);
    if (it != module_cache_.end()) {
      return it->second.metadata.shared;
    }
    return load_shared_memory(dir, kernel_name);
  }
};

static_assert(BackendPolicy<KunlunxinBackend>,
              "KunlunxinBackend must satisfy BackendPolicy concept");

}  // namespace triton_jit
