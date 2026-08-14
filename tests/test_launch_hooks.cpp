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

#include "triton_jit/triton_kernel.h"

#include <atomic>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    throw std::runtime_error("check failed at line " + std::to_string(line) + ": " + expression);
  }
}

#define REQUIRE(expression) check(static_cast<bool>(expression), #expression, __LINE__)

struct HooksGuard {
  HooksGuard() {
    triton_jit::clear_launch_hooks();
  }

  ~HooksGuard() {
    triton_jit::clear_launch_hooks();
  }
};

struct FakeBackend {
  using StreamType = void*;
  using ContextType = void*;
  using KernelHandle = int;
  using LaunchOptions = int;

  static constexpr unsigned int WARP_SIZE = 32;
  inline static std::atomic<int> launch_count {0};
  inline static std::atomic<bool> throw_on_launch {false};

  static void launch_kernel(StreamType,
                            KernelHandle,
                            unsigned,
                            unsigned,
                            unsigned,
                            unsigned,
                            unsigned,
                            unsigned,
                            void**,
                            const LaunchOptions&) {
    ++launch_count;
    if (throw_on_launch.load()) {
      throw std::runtime_error("backend launch failed");
    }
  }

  static void ensure_context() {
  }

  static int get_device_index() {
    return 0;
  }

  static KernelHandle load_kernel(const std::string&, const std::string&) {
    return 1;
  }

  static unsigned int get_shared_memory(const std::string&, const std::string&) {
    return 128;
  }

  static LaunchOptions prepare_launch(const std::string&,
                                      const std::string&,
                                      unsigned int,
                                      const std::string&,
                                      size_t) {
    return 0;
  }

  static void reset() {
    launch_count = 0;
    throw_on_launch = false;
  }
};

static_assert(triton_jit::BackendPolicy<FakeBackend>);

using FakeKernel = triton_jit::TritonKernelImpl<FakeBackend>;

void test_setters_preserve_both_hooks() {
  HooksGuard guard;
  triton_jit::set_launch_enter_hook([](const triton_jit::LaunchMetadata&) {});
  triton_jit::set_launch_exit_hook([](const triton_jit::LaunchMetadata&) {});

  auto snapshot = triton_jit::detail::get_launch_hooks_snapshot();
  REQUIRE(snapshot);
  REQUIRE(snapshot->enter);
  REQUIRE(snapshot->exit);
}

void test_clear_removes_snapshot() {
  HooksGuard guard;
  triton_jit::set_launch_enter_hook([](const triton_jit::LaunchMetadata&) {});
  triton_jit::clear_launch_hooks();

  REQUIRE(!triton_jit::detail::get_launch_hooks_snapshot());
}

void test_concurrent_setters_do_not_lose_updates() {
  HooksGuard guard;
  for (int iteration = 0; iteration < 1000; ++iteration) {
    triton_jit::clear_launch_hooks();

    std::thread enter([] {
      triton_jit::set_launch_enter_hook([](const triton_jit::LaunchMetadata&) {});
    });
    std::thread exit([] {
      triton_jit::set_launch_exit_hook([](const triton_jit::LaunchMetadata&) {});
    });
    enter.join();
    exit.join();

    auto snapshot = triton_jit::detail::get_launch_hooks_snapshot();
    REQUIRE(snapshot);
    REQUIRE(snapshot->enter);
    REQUIRE(snapshot->exit);
  }
}

void test_reentrant_clear_keeps_current_snapshot() {
  HooksGuard guard;
  FakeBackend::reset();
  int enter_count = 0;
  int exit_count = 0;
  int stream_token = 0;
  void* stream = &stream_token;
  triton_jit::LaunchMetadata captured;

  triton_jit::set_launch_enter_hook([&](const triton_jit::LaunchMetadata& metadata) {
    ++enter_count;
    captured = metadata;
    triton_jit::clear_launch_hooks();
  });
  triton_jit::set_launch_exit_hook(
      [&](const triton_jit::LaunchMetadata&) { ++exit_count; });

  FakeKernel kernel("unused", "fake_kernel");
  kernel.launch_with_signature(2, 3, 4, 5, stream, nullptr, "*fp32:16,i32", 2);

  REQUIRE(enter_count == 1);
  REQUIRE(exit_count == 1);
  REQUIRE(FakeBackend::launch_count == 1);
  REQUIRE(captured.kernel_name == "fake_kernel");
  REQUIRE(captured.grid_x == 2);
  REQUIRE(captured.grid_y == 3);
  REQUIRE(captured.grid_z == 4);
  REQUIRE(captured.num_warps == 5);
  REQUIRE(captured.shared_memory == 128);
  REQUIRE(captured.signature == "*fp32:16,i32");
  REQUIRE(captured.stream == stream);

  kernel.launch_with_signature(1, 1, 1, 1, stream, nullptr, "", 0);
  REQUIRE(enter_count == 1);
  REQUIRE(exit_count == 1);
  REQUIRE(FakeBackend::launch_count == 2);
}

void test_enter_exception_prevents_launch() {
  HooksGuard guard;
  FakeBackend::reset();
  int exit_count = 0;

  triton_jit::set_launch_enter_hook(
      [](const triton_jit::LaunchMetadata&) { throw std::runtime_error("enter failed"); });
  triton_jit::set_launch_exit_hook(
      [&](const triton_jit::LaunchMetadata&) { ++exit_count; });

  bool caught = false;
  try {
    FakeKernel("unused", "fake_kernel").launch(1, 1, 1, 1, nullptr, nullptr);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "enter failed";
  }

  REQUIRE(caught);
  REQUIRE(FakeBackend::launch_count == 0);
  REQUIRE(exit_count == 0);
}

void test_backend_exception_skips_exit() {
  HooksGuard guard;
  FakeBackend::reset();
  FakeBackend::throw_on_launch = true;
  int enter_count = 0;
  int exit_count = 0;

  triton_jit::set_launch_enter_hook(
      [&](const triton_jit::LaunchMetadata&) { ++enter_count; });
  triton_jit::set_launch_exit_hook(
      [&](const triton_jit::LaunchMetadata&) { ++exit_count; });

  bool caught = false;
  try {
    FakeKernel("unused", "fake_kernel").launch(1, 1, 1, 1, nullptr, nullptr);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "backend launch failed";
  }

  REQUIRE(caught);
  REQUIRE(FakeBackend::launch_count == 1);
  REQUIRE(enter_count == 1);
  REQUIRE(exit_count == 0);
}

void test_exit_exception_follows_launch() {
  HooksGuard guard;
  FakeBackend::reset();

  triton_jit::set_launch_exit_hook(
      [](const triton_jit::LaunchMetadata&) { throw std::runtime_error("exit failed"); });

  bool caught = false;
  try {
    FakeKernel("unused", "fake_kernel").launch(1, 1, 1, 1, nullptr, nullptr);
  } catch (const std::runtime_error& error) {
    caught = std::string(error.what()) == "exit failed";
  }

  REQUIRE(caught);
  REQUIRE(FakeBackend::launch_count == 1);
}

}  // namespace

int main() {
  try {
    test_setters_preserve_both_hooks();
    test_clear_removes_snapshot();
    test_concurrent_setters_do_not_lose_updates();
    test_reentrant_clear_keeps_current_snapshot();
    test_enter_exception_prevents_launch();
    test_backend_exception_skips_exit();
    test_exit_exception_follows_launch();
  } catch (const std::exception& error) {
    std::cerr << "launch hook test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "launch hook tests passed\n";
  return 0;
}
