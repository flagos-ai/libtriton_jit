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

#include "triton_jit/tuned_config.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "nlohmann/json.hpp"
#include "triton_jit/freeze.h"

#if defined(BACKEND_CUDA) || defined(BACKEND_IX)
#include <cuda.h>
#elif defined(BACKEND_HCU)
#include <hip/hip_runtime_api.h>
#elif defined(BACKEND_NPU)
#include "acl/acl.h"
#endif

namespace triton_jit {

namespace {

  bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
  }

  bool verbose_logging() {
    static const bool enabled = env_flag("TRITON_JIT_LOG_TUNED");
    return enabled;
  }

  void log_info(const std::string& message) {
    if (verbose_logging()) {
      std::cerr << "[triton_jit::tuned] " << message << '\n';
    }
  }

  void log_warning(const std::string& message) {
    std::cerr << "[triton_jit::tuned] warning: " << message << '\n';
  }

  [[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("tuned table " + path.string() + ": " + what);
  }

  // 2 ** ceil(log2(value)) for value >= 1, computed with integer arithmetic so
  // exact powers of two round-trip exactly like Python's math.log2 does.
  int64_t pow2_ceil(int64_t value) noexcept {
    if (value <= 1) {
      return 1;
    }
    const uint64_t below = static_cast<uint64_t>(value - 1);
    const int bits = 64 - __builtin_clzll(below);
    if (bits >= 63) {
      return value;
    }
    return int64_t {1} << bits;
  }

  int64_t ceil_div_32(int64_t value) noexcept {
    // Python: math.ceil(value / 32); integer division truncates toward zero,
    // which is already ceil for negatives.
    return value >= 0 ? (value + 31) / 32 : -((-value) / 32);
  }

  constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;

  inline uint64_t fnv_bytes(uint64_t hash, const void* data, size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= kFnvPrime;
    }
    return hash;
  }

  struct TransparentStringHash {
    using is_transparent = void;
    size_t operator()(std::string_view value) const noexcept {
      return std::hash<std::string_view> {}(value);
    }
    size_t operator()(const std::string& value) const noexcept {
      return std::hash<std::string_view> {}(value);
    }
    size_t operator()(const char* value) const noexcept {
      return std::hash<std::string_view> {}(value);
    }
  };

  using OrderedJson = nlohmann::ordered_json;

  TunedValue parse_value(const std::filesystem::path& path,
                         const std::string& kernel_id,
                         const std::string& name,
                         const OrderedJson& value) {
    if (value.is_boolean()) {
      return value.get<bool>();
    }
    if (value.is_number_integer()) {
      return value.get<int64_t>();
    }
    if (value.is_number_float()) {
      return value.get<double>();
    }
    if (value.is_string()) {
      return value.get<std::string>();
    }
    fail(path,
         "kernel '" + kernel_id + "' kwarg '" + name + "' has unsupported JSON type " +
             std::string(value.type_name()));
  }

  std::string require_string(const std::filesystem::path& path,
                             const OrderedJson& object,
                             const char* field,
                             const std::string& context) {
    auto it = object.find(field);
    if (it == object.end() || !it->is_string()) {
      fail(path, context + ": missing string field '" + field + "'");
    }
    return it->get<std::string>();
  }

  std::string optional_string(const OrderedJson& object, const char* field) {
    auto it = object.find(field);
    if (it == object.end() || it->is_null()) {
      return {};
    }
    if (it->is_string()) {
      return it->get<std::string>();
    }
    return it->dump();
  }

  BackendFingerprint parse_fingerprint(const std::filesystem::path& path, const OrderedJson& root) {
    auto it = root.find("fingerprint");
    if (it == root.end() || !it->is_object()) {
      fail(path, "missing 'fingerprint' object");
    }
    BackendFingerprint fp;
    fp.backend = require_string(path, *it, "backend", "fingerprint");
    fp.vendor = optional_string(*it, "vendor");
    fp.device_name = optional_string(*it, "device_name");
    fp.arch = optional_string(*it, "arch");
    fp.triton_version = optional_string(*it, "triton_version");
    fp.libtriton_jit_version = optional_string(*it, "libtriton_jit_version");
    return fp;
  }

  void check_fingerprint(const std::filesystem::path& path,
                         const BackendFingerprint& table,
                         const BackendFingerprint& runtime) {
    if (table.backend.empty()) {
      fail(path, "fingerprint.backend is empty");
    }
    if (runtime.backend.empty()) {
      log_warning(path.string() + ": running backend is unknown; accepting table for backend '" +
                  table.backend + "'");
    } else if (table.backend != runtime.backend) {
      fail(path, "tuned for backend '" + table.backend + "' but running on '" + runtime.backend + "'");
    }
    if (table.device_name.empty()) {
      log_warning(path.string() + ": table carries no device_name; cannot verify it matches this device");
    } else if (runtime.device_name.empty()) {
      log_warning(path.string() + ": device name of this process is unknown; accepting table for '" +
                  table.device_name + "' unverified");
    } else if (table.device_name != runtime.device_name) {
      if (env_flag("TRITON_JIT_TUNED_IGNORE_DEVICE")) {
        log_warning(path.string() + ": tuned for '" + table.device_name + "' but running on '" +
                    runtime.device_name + "' (accepted because TRITON_JIT_TUNED_IGNORE_DEVICE is set)");
      } else {
        fail(path,
             "tuned for device '" + table.device_name + "' but running on '" + runtime.device_name +
                 "' (set TRITON_JIT_TUNED_IGNORE_DEVICE=1 to override)");
      }
    }
  }

}  // namespace

// ----------------------------------------------------------------------------
// TunedConfig

bool TunedConfig::has(std::string_view name) const noexcept {
  for (const auto& [key, value] : kwargs) {
    if (key == name) {
      return true;
    }
  }
  return false;
}

int64_t TunedConfig::get_i64(std::string_view name, int64_t fallback) const noexcept {
  for (const auto& [key, value] : kwargs) {
    if (key != name) {
      continue;
    }
    if (const auto* i = std::get_if<int64_t>(&value)) {
      return *i;
    }
    if (const auto* b = std::get_if<bool>(&value)) {
      return *b ? 1 : 0;
    }
    if (const auto* d = std::get_if<double>(&value)) {
      return static_cast<int64_t>(*d);
    }
    return fallback;
  }
  return fallback;
}

bool TunedConfig::get_bool(std::string_view name, bool fallback) const noexcept {
  for (const auto& [key, value] : kwargs) {
    if (key != name) {
      continue;
    }
    if (const auto* b = std::get_if<bool>(&value)) {
      return *b;
    }
    if (const auto* i = std::get_if<int64_t>(&value)) {
      return *i != 0;
    }
    return fallback;
  }
  return fallback;
}

double TunedConfig::get_f64(std::string_view name, double fallback) const noexcept {
  for (const auto& [key, value] : kwargs) {
    if (key != name) {
      continue;
    }
    if (const auto* d = std::get_if<double>(&value)) {
      return *d;
    }
    if (const auto* i = std::get_if<int64_t>(&value)) {
      return static_cast<double>(*i);
    }
    if (const auto* b = std::get_if<bool>(&value)) {
      return *b ? 1.0 : 0.0;
    }
    return fallback;
  }
  return fallback;
}

const std::string* TunedConfig::get_str(std::string_view name) const noexcept {
  for (const auto& [key, value] : kwargs) {
    if (key == name) {
      return std::get_if<std::string>(&value);
    }
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
// Key strategies and dtype names

std::optional<KeyStrategy> parse_key_strategy(std::string_view name) noexcept {
  if (name.empty() || name == "default" || name == "None") {
    return KeyStrategy::kDefault;
  }
  if (name == "log") {
    return KeyStrategy::kLog;
  }
  if (name == "align32") {
    return KeyStrategy::kAlign32;
  }
  if (name == "align32_ceil") {
    return KeyStrategy::kAlign32Ceil;
  }
  return std::nullopt;
}

const char* key_strategy_name(KeyStrategy strategy) noexcept {
  switch (strategy) {
    case KeyStrategy::kDefault:
      return "default";
    case KeyStrategy::kLog:
      return "log";
    case KeyStrategy::kAlign32:
      return "align32";
    case KeyStrategy::kAlign32Ceil:
      return "align32_ceil";
  }
  return "default";
}

int64_t normalize_dim(KeyStrategy strategy, int64_t value) noexcept {
  switch (strategy) {
    case KeyStrategy::kDefault:
      return value;
    case KeyStrategy::kLog:
      return value <= 0 ? value : pow2_ceil(value);
    case KeyStrategy::kAlign32:
      if (value == 0) {
        return 0;
      }
      if (value < 0) {
        return value;
      }
      if (value < 32) {
        return pow2_ceil(value);
      }
      return ceil_div_32(value) * 32;
    case KeyStrategy::kAlign32Ceil:
      return ceil_div_32(value) * 32;
  }
  return value;
}

const char* torch_dtype_name(std::string_view triton_dtype) noexcept {
  struct Pair {
    const char* triton;
    const char* torch;
  };
  static constexpr Pair kTable[] = {
      {    "fp16",         "torch.float16"},
      {    "bf16",        "torch.bfloat16"},
      {    "fp32",         "torch.float32"},
      {    "fp64",         "torch.float64"},
      {      "i1",            "torch.bool"},
      {      "i8",            "torch.int8"},
      {     "i16",           "torch.int16"},
      {     "i32",           "torch.int32"},
      {     "i64",           "torch.int64"},
      {      "u8",           "torch.uint8"},
      {     "u16",          "torch.uint16"},
      {     "u32",          "torch.uint32"},
      {     "u64",          "torch.uint64"},
      { "fp8e4nv",   "torch.float8_e4m3fn"},
      {   "fp8e5",     "torch.float8_e5m2"},
      {"fp8e4b15", "torch.float8_e4m3fnuz"},
      {"fp8e5b16", "torch.float8_e5m2fnuz"},
  };
  for (const auto& pair : kTable) {
    if (triton_dtype == pair.triton) {
      return pair.torch;
    }
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
// Fingerprint detection

BackendFingerprint detect_backend_fingerprint(int device_index) {
  BackendFingerprint fp;
#ifdef BACKEND_NAME
  fp.backend = BACKEND_NAME;
#endif
  if (const char* forced = std::getenv("TRITON_JIT_TUNED_DEVICE_NAME");
      forced != nullptr && *forced != '\0') {
    fp.device_name = forced;
    return fp;
  }
#if defined(BACKEND_CUDA) || defined(BACKEND_IX)
  if (cuInit(0) == CUDA_SUCCESS) {
    CUdevice device = 0;
    if (cuDeviceGet(&device, device_index) == CUDA_SUCCESS) {
      char name[256] = {0};
      if (cuDeviceGetName(name, sizeof(name), device) == CUDA_SUCCESS) {
        fp.device_name = name;
      }
      int major = 0;
      int minor = 0;
      if (cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) ==
              CUDA_SUCCESS &&
          cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) ==
              CUDA_SUCCESS) {
        fp.arch = "sm_" + std::to_string(major) + std::to_string(minor);
      }
    }
  }
#elif defined(BACKEND_HCU)
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, device_index) == hipSuccess) {
    fp.device_name = props.name;
    std::string arch(props.gcnArchName);
    if (auto colon = arch.find(':'); colon != std::string::npos) {
      arch = arch.substr(0, colon);
    }
    fp.arch = arch;
  }
#elif defined(BACKEND_NPU)
  (void)device_index;
  if (const char* soc = aclrtGetSocName(); soc != nullptr) {
    fp.device_name = soc;
    fp.arch = soc;
  }
#else
  (void)device_index;
#endif
  return fp;
}

// ----------------------------------------------------------------------------
// TunedKernelTable

uint64_t TunedKernelTable::hash_key(const int64_t* dims, const char* const* dtypes) const noexcept {
  uint64_t hash = kFnvOffset;
  for (size_t i = 0; i < info_.key_columns.size(); ++i) {
    hash = fnv_bytes(hash, &dims[i], sizeof(int64_t));
  }
  for (size_t i = 0; i < info_.ndtype_keys; ++i) {
    const char* dtype = dtypes[i] == nullptr ? "" : dtypes[i];
    hash = fnv_bytes(hash, dtype, std::strlen(dtype));
    const unsigned char separator = 0;
    hash = fnv_bytes(hash, &separator, 1);
  }
  return hash;
}

// Append-only online nodes have stable addresses. Readers use acquire loads;
// insert_resolved serializes writers. A cached table handle sees later online
// inserts, while its info() describes the publication at which it was obtained.
struct TunedKernelTable::OnlineStorage {
  struct Node {
    Entry entry;
    const Node* next = nullptr;
  };
  static constexpr size_t kBuckets = 1024;
  std::atomic<const Node*> buckets[kBuckets] {};
  std::vector<std::unique_ptr<Node>> owned;
};

std::string scoped_kernel_id(std::string_view kernel_id, std::string_view cache_namespace) {
  if (cache_namespace.empty()) return std::string(kernel_id);
  return "@" + std::to_string(cache_namespace.size()) + ":" + std::string(cache_namespace) + ":" +
         std::string(kernel_id);
}

void TunedKernelTable::build_index() {
  size_t bucket_count = 16;
  while (bucket_count < entries_.size() * 2) {
    bucket_count *= 2;
  }
  buckets_.assign(bucket_count, UINT32_MAX);
  std::vector<const char*> dtype_ptrs;
  for (uint32_t index = 0; index < entries_.size(); ++index) {
    Entry& entry = entries_[index];
    dtype_ptrs.clear();
    for (const auto& dtype : entry.dtypes) {
      dtype_ptrs.push_back(dtype.c_str());
    }
    const uint64_t hash = hash_key(entry.dims.data(), dtype_ptrs.data());
    const size_t bucket = hash & (bucket_count - 1);
    entry.next = buckets_[bucket];
    buckets_[bucket] = index;
  }
}

const TunedConfig* TunedKernelTable::find(TuneKeyView key) const noexcept {
  constexpr size_t kMaxColumns = 32;
  const size_t ncols = info_.key_columns.size();
  if (key.ndims != ncols || key.ndtypes != info_.ndtype_keys || ncols > kMaxColumns) {
    return nullptr;
  }
  if ((ncols > 0 && key.dims == nullptr) || (info_.ndtype_keys > 0 && key.dtypes == nullptr)) {
    return nullptr;
  }
  int64_t normalized[kMaxColumns];
  for (size_t i = 0; i < ncols; ++i) {
    normalized[i] = normalize_dim(info_.key_columns[i].second, key.dims[i]);
  }
  const uint64_t hash = hash_key(normalized, key.dtypes);
  auto matches = [&](const Entry& entry) {
    bool equal = ncols == 0 || std::memcmp(entry.dims.data(), normalized, ncols * sizeof(int64_t)) == 0;
    for (size_t i = 0; equal && i < info_.ndtype_keys; ++i) {
      equal = key.dtypes[i] != nullptr && entry.dtypes[i] == key.dtypes[i];
    }
    return equal;
  };
  if (online_) {
    auto* node = online_->buckets[hash & (OnlineStorage::kBuckets - 1)].load(std::memory_order_acquire);
    for (; node; node = node->next)
      if (matches(node->entry)) return &node->entry.config;
  }
  if (offline_) return offline_->find(key);
  if (entries_.empty()) return nullptr;
  uint32_t index = buckets_[hash & (buckets_.size() - 1)];
  while (index != UINT32_MAX) {
    const Entry& entry = entries_[index];
    if (matches(entry)) {
      return &entry.config;
    }
    index = entry.next;
  }
  return nullptr;
}

// ----------------------------------------------------------------------------
// TunedTable

struct TunedTable::Snapshot {
  using KernelMap = std::unordered_map<std::string,
                                       std::shared_ptr<const TunedKernelTable>,
                                       TransparentStringHash,
                                       std::equal_to<>>;
  std::vector<KernelMap> devices;  // indexed by device_index
};

TunedTable& TunedTable::instance() {
  static TunedTable table;
  return table;
}

TunedTable::LoadReport TunedTable::load(const std::filesystem::path& path, int device_index) {
  return load(path, device_index, detect_backend_fingerprint(device_index));
}

TunedTable::LoadReport TunedTable::load(const std::filesystem::path& path,
                                        int device_index,
                                        const BackendFingerprint& runtime) {
  if (device_index < 0) {
    fail(path, "negative device index " + std::to_string(device_index));
  }
  std::ifstream file(path);
  if (!file) {
    fail(path, "cannot open file");
  }
  OrderedJson root;
  try {
    root = OrderedJson::parse(file);
  } catch (const nlohmann::json::exception& error) {
    fail(path, std::string("invalid JSON: ") + error.what());
  }
  if (!root.is_object()) {
    fail(path, "top level is not an object");
  }
  const auto version_it = root.find("format_version");
  if (version_it == root.end() || !version_it->is_number_integer() ||
      (version_it->get<int>() != 1 && version_it->get<int>() != 2)) {
    fail(path, "unsupported or missing format_version (expected 1 or 2)");
  }
  const BackendFingerprint table_fp = parse_fingerprint(path, root);
  check_fingerprint(path, table_fp, runtime);

  const auto kernels_it = root.find("kernels");
  if (kernels_it == root.end() || !kernels_it->is_array()) {
    fail(path, "missing 'kernels' array");
  }

  std::vector<std::shared_ptr<const TunedKernelTable>> loaded;
  size_t total_entries = 0;
  std::unordered_set<std::string> identities;
  for (const OrderedJson& kernel_json : *kernels_it) {
    if (!kernel_json.is_object()) {
      fail(path, "kernel entry is not an object");
    }
    auto table = std::make_shared<TunedKernelTable>();
    TunedKernelInfo& info = table->info_;
    info.kernel_id = require_string(path, kernel_json, "kernel_id", "kernel");
    const std::string context = "kernel '" + info.kernel_id + "'";
    info.cache_namespace = optional_string(kernel_json, "cache_namespace");
    if (version_it->get<int>() == 2 && info.cache_namespace.empty())
      fail(path, context + ": format_version 2 requires a non-empty cache_namespace");
    if (version_it->get<int>() == 1 && !info.cache_namespace.empty())
      fail(path, context + ": cache_namespace requires format_version 2");
    table->storage_id_ = scoped_kernel_id(info.kernel_id, info.cache_namespace);
    if (!identities.insert(table->storage_id_).second) fail(path, context + ": duplicate kernel identity");
    info.op_name = optional_string(kernel_json, "op_name");
    info.config_table_name = optional_string(kernel_json, "config_table_name");
    info.source_sha256 = optional_string(kernel_json, "source_sha256");
    info.candidate_set_hash = optional_string(kernel_json, "candidate_set_hash");
    info.unsupported = optional_string(kernel_json, "unsupported");

    if (auto columns = kernel_json.find("key_columns"); columns != kernel_json.end()) {
      if (!columns->is_array()) {
        fail(path, context + ": 'key_columns' is not an array");
      }
      for (const OrderedJson& column : *columns) {
        if (!column.is_object()) {
          fail(path, context + ": key column is not an object");
        }
        const std::string name = require_string(path, column, "name", context + " key column");
        const std::string strategy_name = optional_string(column, "strategy");
        const auto strategy = parse_key_strategy(strategy_name);
        if (!strategy) {
          fail(path, context + ": unknown key strategy '" + strategy_name + "' for column '" + name + "'");
        }
        info.key_columns.emplace_back(name, *strategy);
      }
    }
    if (auto dtype_keys = kernel_json.find("dtype_keys"); dtype_keys != kernel_json.end()) {
      if (!dtype_keys->is_number_integer() || dtype_keys->get<int64_t>() < 0) {
        fail(path, context + ": 'dtype_keys' must be a non-negative integer");
      }
      info.ndtype_keys = static_cast<size_t>(dtype_keys->get<int64_t>());
    }

    const auto entries_it = kernel_json.find("entries");
    if (entries_it != kernel_json.end() && !entries_it->is_array()) {
      fail(path, context + ": 'entries' is not an array");
    }
    if (entries_it != kernel_json.end() && !info.unsupported.empty() && !entries_it->empty()) {
      fail(path, context + ": marked unsupported but carries entries");
    }
    const size_t ncols = info.key_columns.size();
    if (entries_it != kernel_json.end()) {
      for (const OrderedJson& entry_json : *entries_it) {
        if (!entry_json.is_object()) {
          fail(path, context + ": entry is not an object");
        }
        TunedKernelTable::Entry entry;
        const auto key_it = entry_json.find("key");
        if (key_it == entry_json.end() || !key_it->is_array() || key_it->size() != ncols + info.ndtype_keys) {
          fail(path,
               context + ": entry 'key' must be an array of " + std::to_string(ncols + info.ndtype_keys) +
                   " elements (" + std::to_string(ncols) + " key columns + " +
                   std::to_string(info.ndtype_keys) + " dtypes)");
        }
        for (size_t i = 0; i < ncols; ++i) {
          const OrderedJson& element = (*key_it)[i];
          if (!element.is_number_integer()) {
            fail(path, context + ": key column '" + info.key_columns[i].first + "' is not an integer");
          }
          entry.dims.push_back(normalize_dim(info.key_columns[i].second, element.get<int64_t>()));
        }
        for (size_t i = 0; i < info.ndtype_keys; ++i) {
          const OrderedJson& element = (*key_it)[ncols + i];
          if (!element.is_string()) {
            fail(path, context + ": dtype key #" + std::to_string(i) + " is not a string");
          }
          entry.dtypes.push_back(element.get<std::string>());
        }
        if (auto nw = entry_json.find("num_warps"); nw != entry_json.end()) {
          if (!nw->is_number_integer()) {
            fail(path, context + ": 'num_warps' is not an integer");
          }
          entry.config.num_warps = nw->get<int>();
        }
        if (auto ns = entry_json.find("num_stages"); ns != entry_json.end()) {
          if (!ns->is_number_integer()) {
            fail(path, context + ": 'num_stages' is not an integer");
          }
          entry.config.num_stages = ns->get<int>();
        }
        if (auto extra = entry_json.find("extra"); extra != entry_json.end() && !extra->is_null()) {
          if (!extra->is_object()) {
            fail(path, context + ": 'extra' is not an object");
          }
          for (const auto& [name, value] : extra->items()) {
            if (name == "num_warps" || name == "num_stages") {
              fail(path, context + ": 'extra' must not carry reserved option '" + name + "'");
            }
            entry.config.extra[name] = value.is_string() ? value.get<std::string>() : value.dump();
          }
        }
        if (auto kwargs = entry_json.find("kwargs"); kwargs != entry_json.end() && !kwargs->is_null()) {
          if (kwargs->is_array()) {
            for (const OrderedJson& pair : *kwargs) {
              if (!pair.is_array() || pair.size() != 2 || !pair[0].is_string()) {
                fail(path, context + ": each kwargs element must be a [name, value] pair");
              }
              const std::string name = pair[0].get<std::string>();
              entry.config.kwargs.emplace_back(name, parse_value(path, info.kernel_id, name, pair[1]));
            }
          } else if (kwargs->is_object()) {
            for (const auto& [name, value] : kwargs->items()) {
              entry.config.kwargs.emplace_back(name, parse_value(path, info.kernel_id, name, value));
            }
          } else {
            fail(path, context + ": 'kwargs' must be an array of pairs or an object");
          }
        }
        table->entries_.push_back(std::move(entry));
      }
    }
    table->build_index();
    // Reject duplicate keys: the exporter must have produced exactly one row per key.
    for (const auto& entry : table->entries_) {
      std::vector<const char*> dtype_ptrs;
      for (const auto& dtype : entry.dtypes) {
        dtype_ptrs.push_back(dtype.c_str());
      }
      const TunedConfig* found = table->find(
          TuneKeyView {entry.dims.data(), entry.dims.size(), dtype_ptrs.data(), dtype_ptrs.size()});
      if (found != &entry.config) {
        std::ostringstream key_text;
        for (size_t i = 0; i < entry.dims.size(); ++i) {
          key_text << (i ? ", " : "") << entry.dims[i];
        }
        for (const auto& dtype : entry.dtypes) {
          key_text << ", " << dtype;
        }
        fail(path, context + ": duplicate key (" + key_text.str() + ")");
      }
    }
    info.entry_count = table->entries_.size();
    total_entries += info.entry_count;
    loaded.push_back(std::move(table));
  }

  for (auto& table : loaded) {
    publish(device_index, table);
  }

  LoadReport report;
  report.path = path.string();
  report.table_fingerprint = table_fp;
  report.kernels = loaded.size();
  report.entries = total_entries;
  log_info("loaded " + path.string() + " for device " + std::to_string(device_index) + ": " +
           std::to_string(report.kernels) + " kernels, " + std::to_string(report.entries) +
           " entries (tuned on '" + table_fp.device_name + "', backend " + table_fp.backend + ")");
  return report;
}

// Copy-on-write of the snapshot so readers never see a partial table. The
// previous snapshot is retired into history_, not freed, so a reader that
// loaded the old pointer a moment ago is still on valid memory.
void TunedTable::publish(int device_index, std::shared_ptr<const TunedKernelTable> table) {
  std::lock_guard<std::mutex> lock(load_mu_);
  publish_locked(device_index, std::move(table));
}

void TunedTable::publish_locked(int device_index, std::shared_ptr<const TunedKernelTable> table) {
  const Snapshot* current = snap_.load(std::memory_order_acquire);
  auto next = std::make_unique<Snapshot>();
  if (current != nullptr) {
    next->devices = current->devices;
  }
  if (next->devices.size() <= static_cast<size_t>(device_index)) {
    next->devices.resize(static_cast<size_t>(device_index) + 1);
  }
  next->devices[static_cast<size_t>(device_index)][table->storage_id_] = std::move(table);
  const Snapshot* published = next.get();
  history_.push_back(std::move(next));
  snap_.store(published, std::memory_order_release);
}

size_t TunedTable::load_from_env(int device_index) noexcept {
  const char* value = std::getenv("TRITON_JIT_TUNED_TABLE");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  size_t loaded = 0;
  std::string paths(value);
  size_t begin = 0;
  while (begin <= paths.size()) {
    size_t end = paths.find(':', begin);
    if (end == std::string::npos) {
      end = paths.size();
    }
    const std::string entry = paths.substr(begin, end - begin);
    begin = end + 1;
    if (entry.empty()) {
      continue;
    }
    try {
      load(std::filesystem::path(entry), device_index);
      ++loaded;
    } catch (const std::exception& error) {
      log_warning(std::string("skipping TRITON_JIT_TUNED_TABLE entry: ") + error.what());
    }
  }
  return loaded;
}

const TunedKernelTable* TunedTable::kernel(std::string_view kernel_id, int device_index) const noexcept {
  const Snapshot* snap = snap_.load(std::memory_order_acquire);
  if (snap == nullptr || device_index < 0 || static_cast<size_t>(device_index) >= snap->devices.size()) {
    return nullptr;
  }
  const auto& kernels = snap->devices[static_cast<size_t>(device_index)];
  const auto it = kernels.find(kernel_id);
  return it == kernels.end() ? nullptr : it->second.get();
}

const TunedConfig* TunedTable::find(std::string_view kernel_id,
                                    int device_index,
                                    TuneKeyView key) const noexcept {
  const TunedKernelTable* table = kernel(kernel_id, device_index);
  return table == nullptr ? nullptr : table->find(key);
}

std::vector<TunedKernelInfo> TunedTable::kernels(int device_index) const {
  std::vector<TunedKernelInfo> result;
  const Snapshot* snap = snap_.load(std::memory_order_acquire);
  if (snap == nullptr || device_index < 0 || static_cast<size_t>(device_index) >= snap->devices.size()) {
    return result;
  }
  for (const auto& [id, table] : snap->devices[static_cast<size_t>(device_index)]) {
    result.push_back(table->info());
  }
  return result;
}

void TunedTable::clear() {
  std::lock_guard<std::mutex> lock(load_mu_);
  snap_.store(nullptr, std::memory_order_release);
  history_.clear();
}

// ----------------------------------------------------------------------------
// Online resolver

void TunedTable::set_resolver(Resolver resolver) {
  auto next =
      resolver ? std::make_shared<const Resolver>(std::move(resolver)) : std::shared_ptr<const Resolver>();
  std::atomic_store_explicit(&resolver_, std::move(next), std::memory_order_release);
}

bool TunedTable::has_resolver() const noexcept {
  return std::atomic_load_explicit(&resolver_, std::memory_order_acquire) != nullptr;
}

namespace {

  std::string inflight_key(std::string_view kernel_id, int device_index, TuneKeyView key) {
    std::string text(kernel_id);
    text += '@';
    text += std::to_string(device_index);
    for (size_t i = 0; i < key.ndims; ++i) {
      text += ',';
      text += std::to_string(key.dims[i]);
    }
    for (size_t i = 0; i < key.ndtypes; ++i) {
      text += ',';
      text += key.dtypes[i] == nullptr ? "" : key.dtypes[i];
    }
    return text;
  }

}  // namespace

const TunedConfig* TunedTable::insert_resolved(std::string_view kernel_id,
                                               std::string_view storage_id,
                                               int device_index,
                                               TuneKeyView raw_key,
                                               const ResolvedEntry& resolved) {
  if (resolved.key_columns.size() != raw_key.ndims || raw_key.ndims > 32) {
    throw std::runtime_error("tuned resolver returned an incompatible key width");
  }
  if (!resolved.cache_namespace.empty() &&
      scoped_kernel_id(kernel_id, resolved.cache_namespace) != storage_id)
    throw std::runtime_error("tuned resolver returned a different source namespace");
  std::lock_guard<std::mutex> lock(load_mu_);
  auto table = std::make_shared<TunedKernelTable>();
  table->storage_id_ = storage_id;
  const auto* snapshot = snap_.load(std::memory_order_acquire);
  std::shared_ptr<const TunedKernelTable> current;
  if (snapshot && static_cast<size_t>(device_index) < snapshot->devices.size()) {
    const auto& map = snapshot->devices[device_index];
    auto it = map.find(storage_id);
    if (it != map.end()) current = it->second;
  }
  if (current) {
    if (current->info_.key_columns != resolved.key_columns || current->info_.ndtype_keys != raw_key.ndtypes) {
      throw std::runtime_error("tuned resolver for '" + std::string(kernel_id) +
                               "' disagrees with the bound key schema");
    }
    if (auto* hit = current->find(raw_key)) return hit;
    table->info_ = current->info_;
    table->online_ = current->online_;
    table->offline_ = current->online_ ? current->offline_ : current;
  } else {
    table->info_.kernel_id = kernel_id;
    table->info_.cache_namespace = resolved.cache_namespace;
    table->info_.key_columns = resolved.key_columns;
    table->info_.ndtype_keys = raw_key.ndtypes;
  }
  table->info_.unsupported.clear();
  if (!table->online_) table->online_ = std::make_shared<TunedKernelTable::OnlineStorage>();
  auto node = std::make_unique<TunedKernelTable::OnlineStorage::Node>();
  for (size_t i = 0; i < raw_key.ndims; ++i)
    node->entry.dims.push_back(normalize_dim(resolved.key_columns[i].second, raw_key.dims[i]));
  std::vector<const char*> dtypes;
  for (size_t i = 0; i < raw_key.ndtypes; ++i) {
    node->entry.dtypes.emplace_back(raw_key.dtypes[i]);
    dtypes.push_back(raw_key.dtypes[i]);
  }
  node->entry.config = resolved.config;
  auto& bucket = table->online_->buckets[table->hash_key(node->entry.dims.data(), dtypes.data()) &
                                         (TunedKernelTable::OnlineStorage::kBuckets - 1)];
  node->next = bucket.load(std::memory_order_relaxed);
  auto* saved = node.get();
  table->online_->owned.push_back(std::move(node));
  bucket.store(saved, std::memory_order_release);
  ++table->info_.entry_count;
  publish_locked(device_index, std::move(table));
  return &saved->entry.config;
}

const TunedConfig* TunedTable::find_for_context(std::string_view kernel_id,
                                                int device_index,
                                                TuneKeyView key,
                                                const void* context) const {
  const auto resolver = std::atomic_load_explicit(&resolver_, std::memory_order_acquire);
  if (resolver && resolver->identity) return find(resolver->identity(kernel_id, context), device_index, key);
  return find(kernel_id, device_index, key);
}

const TunedConfig* TunedTable::resolve(
    std::string_view kernel_id, int device_index, TuneKeyView key, const void* context, void* stream) {
  if (device_index < 0 || key.ndims > 32 || (key.ndims && !key.dims) || (key.ndtypes && !key.dtypes))
    throw std::invalid_argument("tuned resolve: invalid device or key");
  for (size_t i = 0; i < key.ndtypes; ++i)
    if (!key.dtypes[i]) throw std::invalid_argument("tuned resolve: null dtype");
  const auto resolver = std::atomic_load_explicit(&resolver_, std::memory_order_acquire);
  const std::string storage_id =
      resolver && resolver->identity ? resolver->identity(kernel_id, context) : std::string(kernel_id);
  if (const TunedConfig* hit = find(storage_id, device_index, key)) return hit;
  if (!resolver) {
    return nullptr;
  }
  // Fail fast before any work that could enter Python.
  detail::refuse_if_frozen(stream,
                           "resolve tuned config for '" + storage_id + "' on device " +
                               std::to_string(device_index) + " key=" + inflight_key("", device_index, key),
                           ColdWork::kConfig);

  std::vector<int64_t> normalized;
  TuneKeyView flight_key = key;
  if (auto* bound = kernel(storage_id, device_index)) {
    if (bound->info().key_columns.size() != key.ndims || bound->info().ndtype_keys != key.ndtypes)
      throw std::runtime_error("tuned resolve: key schema mismatch");
    for (size_t i = 0; i < key.ndims; ++i)
      normalized.push_back(normalize_dim(bound->info().key_columns[i].second, key.dims[i]));
    flight_key.dims = normalized.data();
  }
  const std::string token = inflight_key(storage_id, device_index, flight_key);
  std::shared_future<const TunedConfig*> future;
  std::promise<const TunedConfig*> promise;
  bool owner = false;
  {
    std::lock_guard<std::mutex> lock(resolve_mu_);
    auto it = inflight_.find(token);
    if (it != inflight_.end()) {
      future = it->second;
    } else {
      future = promise.get_future().share();
      inflight_.emplace(token, future);
      owner = true;
    }
  }
  if (!owner) {
    return resolver->wait ? resolver->wait(future)
                          : future.get();  // waits outside every lock; rethrows the owner's exception
  }
  // Owner: run the resolver with no runtime lock held, publish, then release
  // the waiters. The key views point at the caller's storage, which outlives
  // this call.
  const TunedConfig* result = nullptr;
  try {
    // A previous owner can finish between the initial find and acquiring resolve_mu_.
    if (const auto* hit = find(storage_id, device_index, key)) {
      result = hit;
    } else {
      std::optional<ResolvedEntry> resolved =
          resolver->with_stream ? resolver->with_stream(kernel_id, device_index, key, context, stream)
                                : (*resolver)(kernel_id, device_index, key, context);
      if (resolved) {
        result = insert_resolved(kernel_id, storage_id, device_index, key, *resolved);
      }
    }
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(resolve_mu_);
      inflight_.erase(token);
    }
    promise.set_exception(std::current_exception());
    throw;
  }
  {
    std::lock_guard<std::mutex> lock(resolve_mu_);
    inflight_.erase(token);
  }
  promise.set_value(result);
  return result;
}

}  // namespace triton_jit
