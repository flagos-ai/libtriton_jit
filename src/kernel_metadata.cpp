// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "triton_jit/kernel_metadata.h"

#include <fstream>
#include <string>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"

namespace triton_jit {

GpuKernelMeta load_gpu_metadata(const std::string& dir, const std::string& kernel_name) {
  std::string path = fmt::format("{}/{}.json", dir, kernel_name);
  std::ifstream f(path);
  GpuKernelMeta meta;
  if (!f.is_open()) {
    return meta;
  }
  try {
    nlohmann::json j = nlohmann::json::parse(f);
    meta.shared = j.value("shared", 0u);
    if (j.contains("target") && j["target"].contains("arch")) {
      meta.arch = j["target"]["arch"].get<unsigned int>();
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << fmt::format("Failed to parse GPU metadata {}: {}", path, e.what());
  }
  return meta;
}

NpuKernelMetadata load_npu_metadata(const std::string& dir, const std::string& kernel_name) {
  std::string path = fmt::format("{}/{}.json", dir, kernel_name);
  std::ifstream f(path);
  NpuKernelMetadata meta;
  meta.shared = 0;
  meta.mix_mode = "mix";

  if (!f.is_open()) {
    return meta;
  }

  try {
    nlohmann::json j = nlohmann::json::parse(f);
    meta.shared = j.value("shared", 0u);
    meta.mix_mode = j.value("mix_mode", std::string("mix"));

    if (j.contains("workspace_size")) {
      meta.workspace_size = j["workspace_size"].get<size_t>();
      LOG(INFO) << fmt::format("Loaded workspace_size={} from metadata", meta.workspace_size);
    }

    if (j.contains("arg_layout") && j["arg_layout"].is_array()) {
      for (const auto& arg : j["arg_layout"]) {
        if (!arg.contains("type")) continue;
        std::string type_str = arg["type"].get<std::string>();
        if (type_str == "constexpr") continue;

        NpuArgInfo info;
        if (type_str == "ptr" || type_str == "pointer") {
          info.type = NpuArgType::POINTER;
        } else if (type_str == "i64" || type_str == "u64") {
          info.type = NpuArgType::I64;
        } else if (type_str == "i32" || type_str == "u32") {
          info.type = NpuArgType::I32;
        } else if (type_str == "fp64" || type_str == "f64") {
          info.type = NpuArgType::F64;
        } else if (type_str == "fp32" || type_str == "f32") {
          info.type = NpuArgType::F32;
        } else {
          LOG(WARNING) << "Unknown arg type in metadata: " << type_str;
          info.type = NpuArgType::I64;
        }
        meta.arg_layout.push_back(info);
      }
      LOG(INFO) << fmt::format("Loaded arg_layout from JSON with {} args", meta.arg_layout.size());
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << fmt::format("Failed to parse NPU metadata {}: {}", path, e.what());
  }

  return meta;
}

HcuKernelMetadata load_hcu_metadata(const std::string& dir, const std::string& kernel_name) {
  std::string path = fmt::format("{}/{}.json", dir, kernel_name);
  std::ifstream f(path);
  HcuKernelMetadata meta;
  if (!f.is_open()) {
    return meta;
  }

  try {
    nlohmann::json j = nlohmann::json::parse(f);
    meta.shared = j.value("shared", 0u);
    if (j.contains("target") && j["target"].contains("arch")) {
      meta.arch = j["target"]["arch"].get<std::string>();
    }
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << fmt::format("Failed to parse HCU metadata {}: {}", path, e.what());
  }
  return meta;
}

MluKernelMetadata load_mlu_metadata(const std::string& dir, const std::string& kernel_name) {
  std::string path = fmt::format("{}/{}.json", dir, kernel_name);
  std::ifstream f(path);
  MluKernelMetadata meta;
  if (!f.is_open()) {
    return meta;
  }
  nlohmann::json j = nlohmann::json::parse(f);
  meta.shared = j.value("shared", 0u);
  meta.num_warps = j.value("num_warps", 1);
  meta.promote_shared = j.value("promote_shared", false);
  if (j.contains("target") && j["target"].contains("arch")) {
    meta.arch = j["target"]["arch"].get<unsigned int>();
  }
  return meta;
}

unsigned int load_shared_memory(const std::string& dir, const std::string& kernel_name) {
  std::string path = fmt::format("{}/{}.json", dir, kernel_name);
  std::ifstream f(path);
  if (!f.is_open()) {
    return 0;
  }
  try {
    nlohmann::json j = nlohmann::json::parse(f);
    return j.value("shared", 0u);
  } catch (const nlohmann::json::exception& e) {
    LOG(WARNING) << fmt::format("Failed to parse metadata {}: {}", path, e.what());
    return 0;
  }
}

}  // namespace triton_jit
