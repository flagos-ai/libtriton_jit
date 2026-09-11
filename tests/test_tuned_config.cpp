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

// Unit tests for the offline tuned-config table reader (tuned_config.h).
// Runs without a device: every fingerprint is supplied explicitly, except the
// one detection test that only checks the build's backend name.

#include "triton_jit/freeze.h"
#include "triton_jit/tuned_config.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                                        \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      ++failures;                                                                          \
      std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": " << #cond << std::endl; \
    }                                                                                      \
  } while (0)

#define CHECK_THROWS(expr, needle)                                                                    \
  do {                                                                                                \
    bool thrown = false;                                                                              \
    try {                                                                                             \
      (void)(expr);                                                                                   \
    } catch (const std::runtime_error& error) {                                                       \
      thrown = true;                                                                                  \
      if (std::string(error.what()).find(needle) == std::string::npos) {                              \
        ++failures;                                                                                   \
        std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": exception text '" << error.what() \
                  << "' does not contain '" << needle << "'" << std::endl;                            \
      }                                                                                               \
    }                                                                                                 \
    if (!thrown) {                                                                                    \
      ++failures;                                                                                     \
      std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << ": expected exception: " << #expr      \
                << std::endl;                                                                         \
    }                                                                                                 \
  } while (0)

using triton_jit::BackendFingerprint;
using triton_jit::KeyStrategy;
using triton_jit::TunedConfig;
using triton_jit::TunedTable;
using triton_jit::TuneKeyView;

const std::filesystem::path kFixture =
    std::filesystem::path(TRITON_JIT_TEST_SOURCE_DIR) / "fixtures" / "tuned_table_h800.json";

BackendFingerprint h800() {
  BackendFingerprint fp;
  fp.backend = "CUDA";
  fp.device_name = "NVIDIA H800";
  return fp;
}

std::filesystem::path write_temp(const std::string& name, const std::string& content) {
  const auto path = std::filesystem::temp_directory_path() / ("tuned_config_test_" + name + ".json");
  std::ofstream out(path);
  out << content;
  return path;
}

// A stand-in for triton_jit::CompileOptions (jit_utils.h pulls in torch).
struct FakeCompileOptions {
  int num_warps = 4;
  int num_stages = 3;
  std::map<std::string, std::string> extra;
};

void test_strategies() {
  CHECK(triton_jit::normalize_dim(KeyStrategy::kDefault, 1000) == 1000);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kDefault, -7) == -7);
  // log: 2 ** ceil(log2(x))
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 1) == 1);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 2) == 2);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 3) == 4);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 1000) == 1024);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 1024) == 1024);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 1025) == 2048);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, 0) == 0);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, -5) == -5);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kLog, int64_t {1} << 40) == (int64_t {1} << 40));
  // align32 (FlagGems dialect)
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 0) == 0);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 1) == 1);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 5) == 8);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 31) == 32);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 32) == 32);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 33) == 64);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 4090) == 4096);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, 4097) == 4128);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32, -5) == -5);
  // align32_ceil (FlagBLAS dialect)
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32Ceil, 0) == 0);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32Ceil, 5) == 32);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32Ceil, 32) == 32);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32Ceil, 33) == 64);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32Ceil, -5) == 0);
  CHECK(triton_jit::normalize_dim(KeyStrategy::kAlign32Ceil, -33) == -32);
  // names round-trip
  CHECK(triton_jit::parse_key_strategy("default") == KeyStrategy::kDefault);
  CHECK(triton_jit::parse_key_strategy("") == KeyStrategy::kDefault);
  CHECK(triton_jit::parse_key_strategy("log") == KeyStrategy::kLog);
  CHECK(triton_jit::parse_key_strategy("align32") == KeyStrategy::kAlign32);
  CHECK(triton_jit::parse_key_strategy("align32_ceil") == KeyStrategy::kAlign32Ceil);
  CHECK(!triton_jit::parse_key_strategy("nearest").has_value());
  for (auto s :
       {KeyStrategy::kDefault, KeyStrategy::kLog, KeyStrategy::kAlign32, KeyStrategy::kAlign32Ceil}) {
    CHECK(triton_jit::parse_key_strategy(triton_jit::key_strategy_name(s)) == s);
  }
  // dtype names
  CHECK(std::string(triton_jit::torch_dtype_name("fp32")) == "torch.float32");
  CHECK(std::string(triton_jit::torch_dtype_name("bf16")) == "torch.bfloat16");
  CHECK(triton_jit::torch_dtype_name("complex") == nullptr);
}

void test_load_and_find() {
  auto& table = TunedTable::instance();
  table.clear();
  const auto report = table.load(kFixture, 0, h800());
  CHECK(report.kernels == 3);
  CHECK(report.entries == 4);
  CHECK(report.table_fingerprint.device_name == "NVIDIA H800");
  CHECK(report.table_fingerprint.arch == "sm_90");

  const int64_t dims[] = {4096, 4096};
  const char* fp32[] = {"torch.float32"};
  const auto* cfg = table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp32, 1});
  CHECK(cfg != nullptr);
  if (cfg) {
    CHECK(cfg->num_warps == 4);
    CHECK(cfg->num_stages == 4);
    CHECK(cfg->get_i64("BLOCK_M", -1) == 32);
    CHECK(cfg->get_i64("BLOCK_K", -1) == 128);
    CHECK(cfg->get_i64("MISSING", 77) == 77);
    CHECK(cfg->has("BLOCK_M"));
    CHECK(!cfg->has("BLOCK_N"));
    CHECK(cfg->kwargs.size() == 2 && cfg->kwargs[0].first == "BLOCK_M" && cfg->kwargs[1].first == "BLOCK_K");
    const auto copts = cfg->to_compile_options<FakeCompileOptions>();
    CHECK(copts.num_warps == 4 && copts.num_stages == 4 && copts.extra.empty());
  }
  // dtype participates in the key
  const char* fp64[] = {"torch.float64"};
  const auto* cfg64 = table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp64, 1});
  CHECK(cfg64 != nullptr && cfg64->get_i64("BLOCK_M", -1) == 16);
  const char* fp16[] = {"torch.float16"};
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp16, 1}) == nullptr);
  // misses: different shape, wrong arity, unknown kernel, unbound device
  const int64_t other[] = {4096, 4095};
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {other, 2, fp32, 1}) == nullptr);
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 1, fp32, 1}) == nullptr);
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp32, 0}) == nullptr);
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, nullptr, 1}) == nullptr);
  CHECK(table.find("no_such_kernel", 0, TuneKeyView {dims, 2, fp32, 1}) == nullptr);
  CHECK(table.find("sgemv_n_kernel", 1, TuneKeyView {dims, 2, fp32, 1}) == nullptr);
  CHECK(table.find("sgemv_n_kernel", -1, TuneKeyView {dims, 2, fp32, 1}) == nullptr);
  // the skinny decode shape
  const int64_t skinny[] = {1, 8192};
  const auto* decode = table.find("sgemv_n_kernel", 0, TuneKeyView {skinny, 2, fp32, 1});
  CHECK(decode != nullptr && decode->num_warps == 8 && decode->get_i64("BLOCK_K", -1) == 256);

  // kernel handle reuse
  const auto* handle = table.kernel("sgemv_n_kernel", 0);
  CHECK(handle != nullptr);
  if (handle) {
    CHECK(handle->find(TuneKeyView {dims, 2, fp32, 1}) == cfg);
    CHECK(handle->info().entry_count == 3);
    CHECK(handle->info().op_name == "sgemv_n");
    CHECK(handle->info().key_columns.size() == 2 && handle->info().key_columns[0].first == "m");
    CHECK(handle->info().ndtype_keys == 1);
  }
}

void test_strategy_lookup_and_value_types() {
  auto& table = TunedTable::instance();
  table.clear();
  table.load(kFixture, 0, h800());
  const char* halves[] = {"torch.float16", "torch.float16"};
  // raw runtime shape normalises onto the stored (log, log, align32) row
  const int64_t raw[] = {1000, 4000, 4090};
  const auto* cfg = table.find("mm_kernel", 0, TuneKeyView {raw, 3, halves, 2});
  CHECK(cfg != nullptr);
  if (cfg) {
    CHECK(cfg->num_warps == 8);
    CHECK(cfg->extra.size() == 1 && cfg->extra.at("maxnreg") == "255");
    CHECK(cfg->get_i64("BLOCK_M", -1) == 128);
    CHECK(cfg->get_bool("EVEN_K", false) == true);
    CHECK(cfg->get_i64("EVEN_K", -1) == 1);
    CHECK(cfg->get_f64("SCALE", 0.0) == 0.5);
    CHECK(cfg->get_i64("SCALE", -1) == 0);
    CHECK(cfg->get_str("MODE") != nullptr && *cfg->get_str("MODE") == "fast");
    CHECK(cfg->get_str("BLOCK_M") == nullptr);
    // object-form kwargs keep their insertion order
    CHECK(cfg->kwargs.size() == 7 && cfg->kwargs[0].first == "BLOCK_M" && cfg->kwargs[6].first == "MODE");
    const auto copts = cfg->to_compile_options<FakeCompileOptions>();
    CHECK(copts.extra.at("maxnreg") == "255");
  }
  const int64_t exact[] = {1024, 4096, 4096};
  CHECK(table.find("mm_kernel", 0, TuneKeyView {exact, 3, halves, 2}) == cfg);
  const int64_t past_bucket[] = {1025, 4096, 4096};  // log(1025) = 2048 -> miss
  CHECK(table.find("mm_kernel", 0, TuneKeyView {past_bucket, 3, halves, 2}) == nullptr);
  const int64_t k_off[] = {1024, 4096, 4097};  // align32(4097) = 4128 -> miss
  CHECK(table.find("mm_kernel", 0, TuneKeyView {k_off, 3, halves, 2}) == nullptr);
}

void test_unsupported_kernel() {
  auto& table = TunedTable::instance();
  table.clear();
  table.load(kFixture, 0, h800());
  const auto* handle = table.kernel("softmax_kernel", 0);
  CHECK(handle != nullptr);
  if (handle) {
    CHECK(!handle->info().unsupported.empty());
    CHECK(handle->info().entry_count == 0);
    CHECK(handle->find(TuneKeyView {}) == nullptr);
  }
  const auto infos = table.kernels(0);
  CHECK(infos.size() == 3);
  CHECK(table.kernels(5).empty());
}

void test_fingerprint_rules() {
  auto& table = TunedTable::instance();
  table.clear();
  BackendFingerprint npu;
  npu.backend = "NPU";
  npu.device_name = "Ascend910C";
  CHECK_THROWS(table.load(kFixture, 0, npu), "tuned for backend 'CUDA' but running on 'NPU'");
  BackendFingerprint h100;
  h100.backend = "CUDA";
  h100.device_name = "NVIDIA H100 80GB HBM3";
  CHECK_THROWS(table.load(kFixture, 0, h100),
               "tuned for device 'NVIDIA H800' but running on 'NVIDIA H100 80GB HBM3'");
  CHECK(table.kernel("sgemv_n_kernel", 0) == nullptr);  // nothing published on failure
  setenv("TRITON_JIT_TUNED_IGNORE_DEVICE", "1", 1);
  CHECK(table.load(kFixture, 0, h100).kernels == 3);
  unsetenv("TRITON_JIT_TUNED_IGNORE_DEVICE");
  table.clear();
  BackendFingerprint unknown_device;
  unknown_device.backend = "CUDA";
  CHECK(table.load(kFixture, 0, unknown_device).kernels == 3);  // accepted with a warning
  table.clear();
  CHECK_THROWS(table.load(kFixture, -1, h800()), "negative device index");
  // detection: only the backend name is checked, the device name is forced.
  setenv("TRITON_JIT_TUNED_DEVICE_NAME", "NVIDIA H800", 1);
  const auto detected = triton_jit::detect_backend_fingerprint(0);
  CHECK(detected.device_name == "NVIDIA H800");
  if (detected.backend == "CUDA") {
    CHECK(table.load(kFixture, 0).kernels == 3);
  } else {
    std::cerr << "note: build backend is '" << detected.backend << "', skipping detection-based load"
              << std::endl;
  }
  unsetenv("TRITON_JIT_TUNED_DEVICE_NAME");
}

void test_scoped_tables() {
  auto& table = TunedTable::instance();
  table.clear();
  const auto path = write_temp("scoped", R"({"format_version":2,"fingerprint":{"backend":"CUDA"},"kernels":[
    {"kernel_id":"same","cache_namespace":"pkg/a@v1","key_columns":[{"name":"M","strategy":"default"}],
     "dtype_keys":0,"entries":[{"key":[8],"kwargs":[["BLOCK",64]]}]},
    {"kernel_id":"same","cache_namespace":"pkg/b@v1","key_columns":[{"name":"M","strategy":"default"}],
     "dtype_keys":0,"entries":[{"key":[8],"kwargs":[["BLOCK",128]]}]}
  ]})");
  CHECK(table.load(path, 0, h800()).kernels == 2);
  const int64_t dim = 8;
  const TuneKeyView key {&dim, 1, nullptr, 0};
  const auto* a = table.find(triton_jit::scoped_kernel_id("same", "pkg/a@v1"), 0, key);
  const auto* b = table.find(triton_jit::scoped_kernel_id("same", "pkg/b@v1"), 0, key);
  CHECK(a && b && a->get_i64("BLOCK", 0) == 64 && b->get_i64("BLOCK", 0) == 128);
  CHECK(table.find("same", 0, key) == nullptr);
  CHECK(table.find(triton_jit::scoped_kernel_id("same", "pkg/a@v2"), 0, key) == nullptr);
}

void test_malformed_tables() {
  auto& table = TunedTable::instance();
  table.clear();
  CHECK_THROWS(table.load(std::filesystem::path("/nonexistent/tuned.json"), 0, h800()), "cannot open file");
  CHECK_THROWS(table.load(write_temp("syntax", "{"), 0, h800()), "invalid JSON");
  CHECK_THROWS(table.load(write_temp("array", "[]"), 0, h800()), "top level is not an object");
  CHECK_THROWS(
      table.load(write_temp("version",
                            R"({"format_version": 99, "fingerprint": {"backend": "CUDA"}, "kernels": []})"),
                 0,
                 h800()),
      "format_version");
  CHECK_THROWS(table.load(write_temp("nofp", R"({"format_version": 1, "kernels": []})"), 0, h800()),
               "missing 'fingerprint'");
  const std::string head =
      R"({"format_version": 1, "fingerprint": {"backend": "CUDA", "device_name": "NVIDIA H800"}, "kernels": [)";
  CHECK_THROWS(
      table.load(
          write_temp("arity", head + R"({"kernel_id": "k", "key_columns": [{"name": "m"}], "dtype_keys": 1,
        "entries": [{"key": [1], "kwargs": {}}]}]})"),
          0,
          h800()),
      "entry 'key' must be an array of 2 elements");
  CHECK_THROWS(
      table.load(
          write_temp("strategy",
                     head + R"({"kernel_id": "k", "key_columns": [{"name": "m", "strategy": "nearest"}],
        "entries": []}]})"),
          0,
          h800()),
      "unknown key strategy 'nearest'");
  CHECK_THROWS(
      table.load(
          write_temp("dup", head + R"({"kernel_id": "k", "key_columns": [{"name": "m"}], "dtype_keys": 0,
        "entries": [{"key": [8], "kwargs": {"B": 1}}, {"key": [8], "kwargs": {"B": 2}}]}]})"),
          0,
          h800()),
      "duplicate key (8)");
  CHECK_THROWS(
      table.load(write_temp("dup_after_norm",
                            head + R"({"kernel_id": "k", "key_columns": [{"name": "m", "strategy": "log"}],
        "entries": [{"key": [1000], "kwargs": {"B": 1}}, {"key": [1024], "kwargs": {"B": 2}}]}]})"),
                 0,
                 h800()),
      "duplicate key (1024)");
  CHECK_THROWS(table.load(write_temp("reserved", head + R"({"kernel_id": "k", "key_columns": [],
        "entries": [{"key": [], "extra": {"num_warps": "8"}}]}]})"),
                          0,
                          h800()),
               "reserved option 'num_warps'");
  CHECK_THROWS(table.load(write_temp("kwtype", head + R"({"kernel_id": "k", "key_columns": [],
        "entries": [{"key": [], "kwargs": {"B": [1, 2]}}]}]})"),
                          0,
                          h800()),
               "unsupported JSON type");
  CHECK_THROWS(
      table.load(
          write_temp("unsup", head + R"({"kernel_id": "k", "unsupported": "pre_hook", "key_columns": [],
        "entries": [{"key": [], "kwargs": {}}]}]})"),
          0,
          h800()),
      "marked unsupported but carries entries");
  CHECK(table.kernel("k", 0) == nullptr);
}

void test_additive_load_and_device_isolation() {
  auto& table = TunedTable::instance();
  table.clear();
  table.load(kFixture, 0, h800());
  const std::string second = R"({"format_version": 1,
    "fingerprint": {"backend": "CUDA", "device_name": "NVIDIA H800"},
    "kernels": [{"kernel_id": "sgemv_n_kernel", "key_columns": [{"name": "m"}, {"name": "n"}], "dtype_keys": 1,
                 "entries": [{"key": [64, 64, "torch.float32"], "num_warps": 1, "kwargs": {"BLOCK_M": 4}}]}]})";
  const auto path = write_temp("second", second);
  table.load(path, 0, h800());
  const int64_t old_dims[] = {4096, 4096};
  const int64_t new_dims[] = {64, 64};
  const char* fp32[] = {"torch.float32"};
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {old_dims, 2, fp32, 1}) == nullptr);  // shadowed
  const auto* cfg = table.find("sgemv_n_kernel", 0, TuneKeyView {new_dims, 2, fp32, 1});
  CHECK(cfg != nullptr && cfg->get_i64("BLOCK_M", -1) == 4 && cfg->num_warps == 1);
  CHECK(table.kernel("mm_kernel", 0) != nullptr);  // untouched kernels survive
  // device 3 gets the fixture; device 0 keeps the second table
  table.load(kFixture, 3, h800());
  CHECK(table.find("sgemv_n_kernel", 3, TuneKeyView {old_dims, 2, fp32, 1}) != nullptr);
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {old_dims, 2, fp32, 1}) == nullptr);
  CHECK(table.kernels(1).empty());
  CHECK(table.kernels(3).size() == 3);
  // env loading: one good path, one bad, one empty segment
  setenv("TRITON_JIT_TUNED_TABLE", (kFixture.string() + "::/nonexistent/x.json").c_str(), 1);
  setenv("TRITON_JIT_TUNED_DEVICE_NAME", "NVIDIA H800", 1);
  table.clear();
  const auto detected = triton_jit::detect_backend_fingerprint(0);
  if (detected.backend == "CUDA") {
    CHECK(table.load_from_env(0) == 1);
    CHECK(table.kernel("mm_kernel", 0) != nullptr);
  }
  unsetenv("TRITON_JIT_TUNED_TABLE");
  unsetenv("TRITON_JIT_TUNED_DEVICE_NAME");
  CHECK(table.load_from_env(0) == 0);
}

// The JSON that scripts/export_tuned_table.py writes (fixture generated by
// tests/test_export_tuned_table.py --write-fixture) must load unchanged.
void test_exported_fixture_round_trip() {
  auto& table = TunedTable::instance();
  table.clear();
  const auto exported =
      std::filesystem::path(TRITON_JIT_TEST_SOURCE_DIR) / "fixtures" / "tuned_table_exported.json";
  const auto report = table.load(exported, 0, h800());
  CHECK(report.kernels == 3);
  CHECK(report.entries == 4);
  CHECK(report.table_fingerprint.triton_version == "3.6.0");
  const std::string source_namespace = "tests/test_export_tuned_table.py";
  const auto sgemv_id = triton_jit::scoped_kernel_id("sgemv_n_kernel", source_namespace);
  const auto mm_id = triton_jit::scoped_kernel_id("mm_kernel", source_namespace);
  const auto softmax_id = triton_jit::scoped_kernel_id("softmax_kernel", source_namespace);
  // sgemv: default strategies, three dtype keys (A, x, y)
  const int64_t dims[] = {1, 8192};
  const char* fp32x3[] = {"torch.float32", "torch.float32", "torch.float32"};
  const auto* decode = table.find(sgemv_id, 0, TuneKeyView {dims, 2, fp32x3, 3});
  CHECK(decode != nullptr);
  CHECK(table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp32x3, 3}) == nullptr);
  CHECK(table.find(triton_jit::scoped_kernel_id("sgemv_n_kernel", "other/source.py"),
                   0,
                   TuneKeyView {dims, 2, fp32x3, 3}) == nullptr);
  if (decode) {
    CHECK(decode->num_warps == 8 && decode->num_stages == 2);
    CHECK(decode->get_i64("BLOCK_M", -1) == 8 && decode->get_i64("BLOCK_K", -1) == 256);
    CHECK(decode->extra.empty());  // num_ctas == 1 was dropped by the exporter
  }
  // mm: log/log/align32 with a raw shape, bool recovered, extras carried
  const int64_t raw[] = {1000, 4000, 4090};
  const char* halves[] = {"torch.float16", "torch.float16"};
  const auto* big = table.find(mm_id, 0, TuneKeyView {raw, 3, halves, 2});
  CHECK(big != nullptr);
  if (big) {
    CHECK(big->get_bool("EVEN_K", false) == true);
    CHECK(big->get_i64("SPLIT_K", -1) == 1);
    CHECK(big->get_f64("SCALE", 0.0) == 0.5);
    CHECK(big->extra.size() == 2 && big->extra.at("num_ctas") == "2" && big->extra.at("maxnreg") == "255");
    CHECK(big->kwargs.size() == 6 && big->kwargs[0].first == "BLOCK_M" && big->kwargs[5].first == "SCALE");
    const auto copts = big->to_compile_options<FakeCompileOptions>();
    CHECK(copts.num_warps == 8 && copts.extra.at("num_ctas") == "2");
  }
  const int64_t small_raw[] = {33, 64, 40};  // log(33)=64, log(64)=64, align32(40)=64
  const char* bf16s[] = {"torch.bfloat16", "torch.bfloat16"};
  const auto* small = table.find(mm_id, 0, TuneKeyView {small_raw, 3, bf16s, 2});
  CHECK(small != nullptr && small->get_bool("EVEN_K", true) == false && small->get_i64("SPLIT_K", -1) == 4);
  const auto* handle = table.kernel(mm_id, 0);
  CHECK(handle != nullptr && handle->info().cache_namespace == source_namespace);
  CHECK(handle != nullptr && handle->info().candidate_set_hash.size() == 32);
  CHECK(handle != nullptr && !handle->info().config_table_name.empty());
  // the refused kernel is listed with its reason and never hits
  const auto* refused = table.kernel(softmax_id, 0);
  CHECK(refused != nullptr && refused->info().unsupported.find("heuristics") != std::string::npos);
}

void test_resolver() {
  auto& table = TunedTable::instance();
  table.clear();
  table.set_resolver(nullptr);
  CHECK(!table.has_resolver());
  const int64_t dims[] = {1000, 4000};
  const char* fp32[] = {"torch.float32"};
  CHECK(table.resolve("r_kernel", 0, TuneKeyView {dims, 2, fp32, 1}) == nullptr);  // no resolver: plain miss

  std::atomic<int> calls {0};
  std::atomic<bool> decline {false};
  std::atomic<bool> explode {false};
  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool gate_open = true;
  table.set_resolver([&](std::string_view kernel_id, int device, TuneKeyView key, const void* context)
                         -> std::optional<TunedTable::ResolvedEntry> {
    CHECK(context == &calls);  // whatever the caller passed arrives untouched
    calls.fetch_add(1);
    {
      std::unique_lock<std::mutex> lock(gate_mu);
      gate_cv.wait(lock, [&] { return gate_open; });
    }
    if (explode.load()) {
      throw std::runtime_error("benchmark failed");
    }
    if (decline.load()) {
      return std::nullopt;
    }
    TunedTable::ResolvedEntry entry;
    entry.key_columns = {
        {"m", KeyStrategy::kLog},
        {"n", KeyStrategy::kLog}
    };
    entry.config.num_warps = 8;
    entry.config.kwargs.emplace_back("BLOCK",
                                     int64_t {key.dims[0] + device + (kernel_id == "r_kernel" ? 0 : 1000)});
    return entry;
  });
  CHECK(table.has_resolver());

  // first resolve calls out, stores under the normalised key, later finds hit
  const auto* cfg = table.resolve("r_kernel", 0, TuneKeyView {dims, 2, fp32, 1}, &calls);
  CHECK(cfg != nullptr && cfg->num_warps == 8 && cfg->get_i64("BLOCK", -1) == 1000);
  CHECK(calls.load() == 1);
  const int64_t same_bucket[] = {1024, 4096};  // log(1000)=1024, log(4000)=4096
  CHECK(table.find("r_kernel", 0, TuneKeyView {same_bucket, 2, fp32, 1}) == cfg);
  CHECK(table.resolve("r_kernel", 0, TuneKeyView {same_bucket, 2, fp32, 1}, &calls) == cfg);
  CHECK(calls.load() == 1);
  const auto* handle = table.kernel("r_kernel", 0);
  CHECK(handle != nullptr && handle->info().entry_count == 1 &&
        handle->info().key_columns[0].second == KeyStrategy::kLog);
  // a second key extends the same kernel table; the first row survives
  const int64_t other[] = {8, 8};
  const auto* cfg2 = table.resolve("r_kernel", 0, TuneKeyView {other, 2, fp32, 1}, &calls);
  CHECK(cfg2 != nullptr && cfg2->get_i64("BLOCK", -1) == 8);
  CHECK(table.kernel("r_kernel", 0)->info().entry_count == 2);
  CHECK(table.find("r_kernel", 0, TuneKeyView {dims, 2, fp32, 1}) != nullptr);
  // devices are separate
  CHECK(table.resolve("r_kernel", 1, TuneKeyView {dims, 2, fp32, 1}, &calls)->get_i64("BLOCK", -1) == 1001);
  // a declined key is not cached
  decline.store(true);
  const int64_t declined[] = {3, 3};
  CHECK(table.resolve("r_kernel", 0, TuneKeyView {declined, 2, fp32, 1}, &calls) == nullptr);
  CHECK(table.resolve("r_kernel", 0, TuneKeyView {declined, 2, fp32, 1}, &calls) == nullptr);
  CHECK(calls.load() == 5);
  decline.store(false);
  // frozen: refused before the resolver runs; loaded rows still answer
  {
    triton_jit::ScopedFreeze guard;
    const int64_t fresh[] = {40, 40};  // log -> {64, 64}: a bucket nobody resolved yet
    bool refused = false;
    try {
      table.resolve("r_kernel", 0, TuneKeyView {fresh, 2, fp32, 1}, &calls);
    } catch (const triton_jit::FrozenMissError&) {
      refused = true;
    }
    CHECK(refused);
    CHECK(calls.load() == 5);
    const auto* still = table.resolve("r_kernel", 0, TuneKeyView {dims, 2, fp32, 1}, &calls);
    CHECK(still != nullptr && still->get_i64("BLOCK", -1) == 1000);
  }
  // an exception reaches the caller and is not cached
  explode.store(true);
  const int64_t boom[] = {129, 129};  // log -> {256, 256}
  bool threw = false;
  try {
    table.resolve("r_kernel", 0, TuneKeyView {boom, 2, fp32, 1}, &calls);
  } catch (const std::runtime_error& error) {
    threw = std::string(error.what()) == "benchmark failed";
  }
  CHECK(threw);
  explode.store(false);
  CHECK(table.resolve("r_kernel", 0, TuneKeyView {boom, 2, fp32, 1}, &calls) !=
        nullptr);  // retried after the failure
  // same key from 8 threads while the resolver is blocked: one call, one answer for everybody
  {
    std::lock_guard<std::mutex> lock(gate_mu);
    gate_open = false;
  }
  const int before = calls.load();
  std::vector<std::thread> workers;
  std::vector<const TunedConfig*> answers(8, nullptr);
  for (int t = 0; t < 8; ++t) {
    workers.emplace_back([&, t] {
      const int64_t thread_dims[] = {77 + t, 77 + t};  // different raw keys, same log bucket
      answers[t] = table.resolve("r_kernel", 0, TuneKeyView {thread_dims, 2, fp32, 1}, &calls);
    });
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  {
    std::lock_guard<std::mutex> lock(gate_mu);
    gate_open = true;
  }
  gate_cv.notify_all();
  for (auto& worker : workers) {
    worker.join();
  }
  CHECK(calls.load() == before + 1);
  for (const auto* answer : answers) {
    CHECK(answer != nullptr && answer == answers[0] && answer->get_i64("BLOCK", -1) >= 77 &&
          answer->get_i64("BLOCK", -1) <= 84);
  }
  // resolving into a loaded table keeps the table's own key layout, and a
  // key of the wrong width is an error rather than a silent bad row
  table.clear();
  table.load(kFixture, 0, h800());
  bool schema_refused = false;
  try {
    table.resolve("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp32, 1}, &calls);
  } catch (const std::runtime_error&) {
    schema_refused = true;
  }
  CHECK(schema_refused);  // Same width, different strategies must never be merged.
  CHECK(table.kernel("sgemv_n_kernel", 0)->info().entry_count == 3);
  const int64_t one_dim[] = {5};
  bool mismatch = false;
  try {
    table.resolve("sgemv_n_kernel", 0, TuneKeyView {one_dim, 1, fp32, 1}, &calls);
  } catch (const std::runtime_error& error) {
    mismatch = std::string(error.what()).find("key") != std::string::npos;
  }
  CHECK(mismatch);
  table.set_resolver(nullptr);
}

void test_concurrent_find_during_load() {
  auto& table = TunedTable::instance();
  table.clear();
  table.load(kFixture, 0, h800());
  std::atomic<bool> stop {false};
  std::atomic<long> hits {0};
  std::atomic<long> bad {0};
  std::vector<std::thread> readers;
  for (int t = 0; t < 8; ++t) {
    readers.emplace_back([&] {
      const int64_t dims[] = {4096, 4096};
      const char* fp32[] = {"torch.float32"};
      while (!stop.load(std::memory_order_relaxed)) {
        const auto* cfg = table.find("sgemv_n_kernel", 0, TuneKeyView {dims, 2, fp32, 1});
        if (cfg == nullptr) {
          continue;  // a reload is in flight
        }
        if (cfg->get_i64("BLOCK_M", -1) != 32 || cfg->num_warps != 4) {
          bad.fetch_add(1, std::memory_order_relaxed);
        }
        hits.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (int i = 0; i < 50; ++i) {
    table.load(kFixture, 0, h800());
  }
  stop.store(true);
  for (auto& thread : readers) {
    thread.join();
  }
  CHECK(bad.load() == 0);
  CHECK(hits.load() > 0);
}

}  // namespace

int main() {
  test_strategies();
  test_load_and_find();
  test_strategy_lookup_and_value_types();
  test_unsupported_kernel();
  test_fingerprint_rules();
  test_scoped_tables();
  test_malformed_tables();
  test_additive_load_and_device_isolation();
  test_exported_fixture_round_trip();
  test_resolver();
  test_concurrent_find_during_load();
  TunedTable::instance().clear();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed" << std::endl;
    return 1;
  }
  std::cout << "tuned_config: all checks passed" << std::endl;
  return 0;
}
