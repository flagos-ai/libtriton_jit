#include "triton_jit/kernel_metadata.h"

#include <fstream>
#include <string>

#include "c10/util/Logging.h"
#include "fmt/core.h"
#include "nlohmann/json.hpp"

namespace triton_jit {

GpuKernelMeta load_gpu_metadata(const std::string& dir,
                                const std::string& kernel_name) {
    std::string path = fmt::format("{}/{}.json", dir, kernel_name);
    std::ifstream f(path);
    GpuKernelMeta meta;
    if (!f.is_open()) {
        return meta;
    }
    nlohmann::json j = nlohmann::json::parse(f);
    meta.shared = j.value("shared", 0u);
    if (j.contains("target") && j["target"].contains("arch")) {
        meta.arch = j["target"]["arch"].get<unsigned int>();
    }
    return meta;
}

NpuKernelMetadata load_npu_metadata(const std::string& dir,
                                    const std::string& kernel_name) {
    std::string path = fmt::format("{}/{}.json", dir, kernel_name);
    std::ifstream f(path);
    NpuKernelMetadata meta;
    meta.shared = 0;
    meta.mix_mode = "mix";

    if (!f.is_open()) {
        return meta;
    }

    nlohmann::json j = nlohmann::json::parse(f);
    meta.shared = j.value("shared", 0u);
    meta.mix_mode = j.value("mix_mode", std::string("mix"));

    if (j.contains("workspace_size")) {
        meta.workspace_size = j["workspace_size"].get<size_t>();
        LOG(INFO) << fmt::format("Loaded workspace_size={} from metadata",
                                 meta.workspace_size);
    }

    if (j.contains("arg_layout") && j["arg_layout"].is_array()) {
        for (const auto& arg : j["arg_layout"]) {
            if (!arg.contains("type")) continue;
            std::string type_str = arg["type"].get<std::string>();
            if (type_str == "constexpr") continue;

            NpuArgInfo info;
            if (type_str == "ptr" || type_str == "pointer") {
                info.type = NpuArgType::POINTER;
                info.size = sizeof(void*);
            } else if (type_str == "i64" || type_str == "u64") {
                info.type = NpuArgType::I64;
                info.size = sizeof(int64_t);
            } else if (type_str == "i32" || type_str == "u32") {
                info.type = NpuArgType::I32;
                info.size = sizeof(int32_t);
            } else if (type_str == "fp64" || type_str == "f64") {
                info.type = NpuArgType::F64;
                info.size = sizeof(double);
            } else if (type_str == "fp32" || type_str == "f32") {
                info.type = NpuArgType::F32;
                info.size = sizeof(float);
            } else {
                LOG(WARNING) << "Unknown arg type in metadata: " << type_str;
                info.type = NpuArgType::I64;
                info.size = sizeof(int64_t);
            }
            meta.arg_layout.push_back(info);
        }
        LOG(INFO) << fmt::format("Loaded arg_layout from JSON with {} args",
                                 meta.arg_layout.size());
    }

    return meta;
}

unsigned int load_shared_memory(const std::string& dir,
                                const std::string& kernel_name) {
    std::string path = fmt::format("{}/{}.json", dir, kernel_name);
    std::ifstream f(path);
    if (!f.is_open()) {
        return 0;
    }
    nlohmann::json j = nlohmann::json::parse(f);
    return j.value("shared", 0u);
}

} // namespace triton_jit
