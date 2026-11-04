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

#pragma once

// Tuned configs from offline tables or an optional cold-miss resolver.
// find() is noexcept, allocates nothing and never calls Python.
// load() validates backend/device identity before publishing a table.
// See docs/autotune.md for source namespaces, table format and usage.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace triton_jit {

using TunedValue = std::variant<int64_t, bool, double, std::string>;

// One tuned configuration: the compile options plus the constexpr keyword
// arguments in the kernel's parameter order.
struct TunedConfig {
  int num_warps = 4;
  int num_stages = 3;
  // Mirrors CompileOptions::extra (backend-specific compiler switches).
  std::map<std::string, std::string> extra;
  // Constexpr kwargs in kernel parameter order, e.g. {"BLOCK_M", 64}.
  std::vector<std::pair<std::string, TunedValue>> kwargs;

  bool has(std::string_view name) const noexcept;
  int64_t get_i64(std::string_view name, int64_t fallback) const noexcept;
  bool get_bool(std::string_view name, bool fallback) const noexcept;
  double get_f64(std::string_view name, double fallback) const noexcept;
  const std::string* get_str(std::string_view name) const noexcept;

  // Build a CompileOptions (or any aggregate {num_warps, num_stages, extra})
  // without this header having to include jit_utils.h:
  //   auto copts = cfg.to_compile_options<triton_jit::CompileOptions>();
  template <typename CompileOptionsT>
  CompileOptionsT to_compile_options() const {
    return CompileOptionsT {num_warps, num_stages, extra};
  }
};

// Identity of the environment a table was tuned on. `backend` and
// `device_name` are enforced by TunedTable::load; the rest is recorded so a
// human can tell where a table came from.
struct BackendFingerprint {
  std::string backend;                // libtriton_jit backend name: CUDA, IX, NPU, ...
  std::string vendor;                 // FlagGems / FlagBLAS vendor name: nvidia, iluvatar, ...
  std::string device_name;            // e.g. "NVIDIA H800"
  std::string arch;                   // e.g. "sm_90"
  std::string triton_version;         // e.g. "3.6.0"
  std::string libtriton_jit_version;  // e.g. "0.1.0"
};

// Detect the fingerprint of the running process for `device_index`. `backend`
// comes from the build; `device_name` / `arch` from the device runtime when the
// backend exposes one (CUDA/IX/HCU/NPU today), otherwise they stay empty and
// load() only enforces the backend. TRITON_JIT_TUNED_DEVICE_NAME overrides the
// detected device name (tests, CI without the device).
BackendFingerprint detect_backend_fingerprint(int device_index);

// How the Python tuner normalised a key column before storing it. Must mirror
// the strategies in libentry.py so that a runtime shape maps to the same row.
// (libentry.py strategies: "default" -> key; "log" -> 2 ** ceil(log2(key));
// "align32" in FlagGems -> 0 if key == 0, 2 ** ceil(log2(key)) if key < 32,
// else ceil(key / 32) * 32; "align32" in FlagBLAS -> ceil(key / 32) * 32
// unconditionally. The exporter records which dialect it read.)
enum class KeyStrategy : uint8_t {
  kDefault,      // "default": exact value
  kLog,          // "log": 2 ** ceil(log2(key))
  kAlign32,      // "align32": FlagGems dialect (power-of-two below 32)
  kAlign32Ceil,  // "align32_ceil": FlagBLAS dialect (always ceil to 32)
};
std::optional<KeyStrategy> parse_key_strategy(std::string_view name) noexcept;
const char* key_strategy_name(KeyStrategy strategy) noexcept;
// Values <= 0 are returned unchanged for kLog / kAlign32 (Python would have
// raised, so no row can exist for them; the lookup simply misses).
int64_t normalize_dim(KeyStrategy strategy, int64_t value) noexcept;

// The tuner appends `str(torch_dtype)` to every key ("torch.float32"). A
// consumer that thinks in Triton dtype names can translate: "fp32" ->
// "torch.float32", "bf16" -> "torch.bfloat16", ... Returns nullptr for an
// unknown name.
const char* torch_dtype_name(std::string_view triton_dtype) noexcept;

// A lookup key as the caller has it at launch time: the raw integer key
// columns (in the tuner's `key` order) and the dtype strings the tuner appended
// (in argument order, spelled exactly as the exporter wrote them).
struct TuneKeyView {
  const int64_t* dims = nullptr;
  size_t ndims = 0;
  const char* const* dtypes = nullptr;
  size_t ndtypes = 0;
};

struct TunedKernelInfo {
  std::string kernel_id;          // Triton JIT function name, e.g. "sgemv_n_kernel"
  std::string cache_namespace;    // source identity; empty only for legacy unscoped tables
  std::string op_name;            // LibTuner name the table was tuned under
  std::string config_table_name;  // SQLite table the rows came from (provenance)
  std::vector<std::pair<std::string, KeyStrategy>> key_columns;
  size_t ndtype_keys = 0;
  std::string source_sha256;
  std::string candidate_set_hash;
  // Non-empty when the exporter refused the kernel (pre_hook / heuristics /
  // non-serialisable constexpr). The kernel is listed so introspection can say
  // why, but find() always misses.
  std::string unsupported;
  size_t entry_count = 0;
};

// Offline entries are immutable. Online entries are appended with stable addresses
// and become visible through existing handles to that online generation. Every
// table/config stays alive until clear(); info() describes its publication.
// A later load starts a new generation; old handles still refer to the old one.
class TunedKernelTable {
 public:
  const TunedConfig* find(TuneKeyView key) const noexcept;
  const TunedKernelInfo& info() const noexcept {
    return info_;
  }

 private:
  friend class TunedTable;
  struct Entry {
    std::vector<int64_t> dims;        // normalised key columns
    std::vector<std::string> dtypes;  // appended dtype keys
    TunedConfig config;
    uint32_t next = UINT32_MAX;  // hash chain
  };
  std::vector<Entry> entries_;
  std::vector<uint32_t> buckets_;  // hash -> first entry index
  TunedKernelInfo info_;
  std::string storage_id_;
  struct OnlineStorage;
  std::shared_ptr<OnlineStorage> online_;
  std::shared_ptr<const TunedKernelTable> offline_;
  void build_index();
  uint64_t hash_key(const int64_t* dims, const char* const* dtypes) const noexcept;
};

// Bind once and reuse this identifier with find()/kernel(). Namespaces must
// identify immutable source versions. Paths are a process-local default;
// consumers transporting tables may supply the same stable namespace at both ends.
std::string scoped_kernel_id(std::string_view kernel_id, std::string_view cache_namespace);

class TunedTable {
 public:
  static TunedTable& instance();

  struct LoadReport {
    std::string path;
    BackendFingerprint table_fingerprint;
    size_t kernels = 0;
    size_t entries = 0;
  };

  // Load a JSON table exported by scripts/export_tuned_table.py and bind it to
  // `device_index`. Validates the table fingerprint against `runtime`. Throws
  // std::runtime_error on fingerprint mismatch or malformed input; on success
  // the new kernels are published atomically and shadow same-named kernels
  // loaded earlier for that device. Not a hot path; takes a lock. Safe to call
  // while other threads are in find().
  LoadReport load(const std::filesystem::path& path, int device_index, const BackendFingerprint& runtime);
  // Same, with the runtime fingerprint detected for `device_index`.
  LoadReport load(const std::filesystem::path& path, int device_index);

  // Load every ':'-separated path in TRITON_JIT_TUNED_TABLE for `device_index`.
  // Never throws: a table that fails validation is reported on stderr and
  // skipped. Returns the number of tables loaded.
  size_t load_from_env(int device_index) noexcept;

  // Hot path. nullptr when no table is bound to the device, the kernel is
  // unknown, or the key has no row.
  const TunedConfig* find(std::string_view kernel_id, int device_index, TuneKeyView key) const noexcept;
  // Uses the installed resolver's source binding without entering Python.
  const TunedConfig* find_for_context(std::string_view kernel_id,
                                      int device_index,
                                      TuneKeyView key,
                                      const void* context) const;
  // Handle for callers that want to skip the kernel lookup on every launch.
  const TunedKernelTable* kernel(std::string_view kernel_id, int device_index) const noexcept;

  std::vector<TunedKernelInfo> kernels(int device_index) const;

  // ---- optional online layer -------------------------------------------
  // What a resolver hands back for a key it tuned: the configuration plus
  // the key columns (with their strategies) so the runtime can store the
  // entry under the normalised key and answer the next find() itself.
  struct ResolvedEntry {
    TunedConfig config;
    std::vector<std::pair<std::string, KeyStrategy>> key_columns;
    std::string cache_namespace;
  };
  // Called on a resolve() miss, outside every runtime lock and never while
  // frozen. `context` is whatever the caller of resolve() passed (typically
  // its argument pack, so a Python-backed resolver can benchmark with the
  // real tensors); the runtime never dereferences it. Returning nullopt
  // means "no configuration for this key"; the miss is not cached and the
  // next resolve() asks again. Exceptions propagate to every caller waiting
  // on that key.
  struct Resolver
      : std::function<std::optional<ResolvedEntry>(std::string_view, int, TuneKeyView, const void*)> {
    using Function =
        std::function<std::optional<ResolvedEntry>(std::string_view, int, TuneKeyView, const void*)>;
    using Function::Function;
    // Optional adapter hooks. identity must not enter Python. wait is where a
    // Python-facing adapter releases the GIL, including for non-owner callers.
    std::function<std::string(std::string_view, const void*)> identity;
    std::function<const TunedConfig*(std::shared_future<const TunedConfig*>&)> wait;
    std::function<std::optional<ResolvedEntry>(std::string_view, int, TuneKeyView, const void*, void*)>
        with_stream;
  };
  void set_resolver(Resolver resolver);
  bool has_resolver() const noexcept;

  // find(), then on a miss ask the resolver once per (kernel, device, key):
  // concurrent callers for the same key wait for the first one's answer.
  // Throws FrozenMissError before touching the resolver if the runtime is
  // frozen or `stream` is being captured. nullptr when there is no resolver
  // or the resolver declined.
  const TunedConfig* resolve(std::string_view kernel_id,
                             int device_index,
                             TuneKeyView key,
                             const void* context = nullptr,
                             void* stream = nullptr);
  // Drops every table. Unlike load(), this must not race with find()/resolve(): it is
  // meant for tests and process teardown.
  void clear();

 private:
  TunedTable() = default;
  struct Snapshot;
  // Published snapshot; readers take one acquire load and never touch a
  // refcount. Retired snapshots are kept in `history_` (they are a handful of
  // maps) so pointers handed out earlier stay valid. Entries are shared, not copied.
  std::atomic<const Snapshot*> snap_ {nullptr};
  std::vector<std::unique_ptr<const Snapshot>> history_;
  mutable std::mutex load_mu_;

  // Publishes `table` for (kernel_id, device_index), replacing an existing
  // one. Caller must not hold load_mu_.
  void publish(int device_index, std::shared_ptr<const TunedKernelTable> table);
  void publish_locked(int device_index, std::shared_ptr<const TunedKernelTable> table);
  const TunedConfig* insert_resolved(std::string_view kernel_id,
                                     std::string_view storage_id,
                                     int device_index,
                                     TuneKeyView raw_key,
                                     const ResolvedEntry& resolved);

  std::shared_ptr<const Resolver> resolver_;
  std::mutex resolve_mu_;
  std::unordered_map<std::string, std::shared_future<const TunedConfig*>> inflight_;
};

}  // namespace triton_jit
