// Source identity and concurrent configuration publication regressions.
#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "triton_jit/tuned_config.h"

using namespace triton_jit;

static TunedTable::ResolvedEntry entry(int64_t block) {
  TunedTable::ResolvedEntry result;
  result.key_columns = {
      {"M", KeyStrategy::kDefault}
  };
  result.config.kwargs.emplace_back("BLOCK_M", block);
  return result;
}

int main(int argc, char** argv) {
  auto& table = TunedTable::instance();
  const std::string mode = argc > 1 ? argv[1] : "identity";
  if (mode == "identity") {
    int calls = 0;
    TunedTable::Resolver resolver = [&](auto, auto, auto, const void* context) {
      ++calls;
      const auto& source_path = *static_cast<const std::string*>(context);
      return std::optional(entry(source_path == "/pkg/ops/mm.py" ? 64 : 128));
    };
    resolver.identity = [](auto name, const void* ctx) {
      return scoped_kernel_id(name, *static_cast<const std::string*>(ctx));
    };
    table.set_resolver(resolver);
    const int64_t dim = 1024;
    const std::string source_a = "/pkg/ops/mm.py", source_b = "/pkg/hopper/mm.py";
    const auto* a = table.resolve("mm_kernel", 0, {&dim, 1, nullptr, 0}, &source_a);
    const auto* b = table.resolve("mm_kernel", 0, {&dim, 1, nullptr, 0}, &source_b);
    std::cout << "{\"probe\":\"source_collision\",\"a_block\":" << a->get_i64("BLOCK_M", -1)
              << ",\"b_block\":" << b->get_i64("BLOCK_M", -1)
              << ",\"expected_b\":128,\"resolver_calls\":" << calls << "}\n";
    if (a->get_i64("BLOCK_M", -1) != 64 || b->get_i64("BLOCK_M", -1) != 128 || calls != 2) return 1;
  } else if (mode == "race") {
    constexpr int n = 16, seeds = 128;
    std::atomic<int> ready {0};
    std::atomic<bool> go {false};
    table.set_resolver([&](auto, auto, TuneKeyView key, const void*) {
      if (key.dims[0] >= seeds) {
        ready.fetch_add(1);
        while (!go.load()) std::this_thread::yield();
      }
      return std::optional(entry(key.dims[0]));
    });
    for (int64_t k = 0; k < seeds; ++k) table.resolve("race", 0, {&k, 1, nullptr, 0});
    std::vector<std::thread> workers;
    std::atomic<int> returned {0};
    for (int i = 0; i < n; ++i)
      workers.emplace_back([&, i] {
        const int64_t k = seeds + i;
        if (table.resolve("race", 0, {&k, 1, nullptr, 0})) returned.fetch_add(1);
      });
    while (ready.load() != n) std::this_thread::yield();
    go.store(true);
    for (auto& worker : workers) worker.join();
    int retained = 0;
    for (int64_t k = seeds; k < seeds + n; ++k)
      retained += table.find("race", 0, {&k, 1, nullptr, 0}) != nullptr;
    std::cout << "{\"probe\":\"different_key_publication\",\"returned\":" << returned
              << ",\"new_rows_retained\":" << retained << ",\"expected\":" << n << "}\n";
    if (retained != n || returned != n) return 1;
  } else {
    std::cerr << "Expected identity or race\n";
    return 1;
  }
  table.set_resolver(nullptr);
  table.clear();
}
