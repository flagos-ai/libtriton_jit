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

#include <memory>
#include <utility>

namespace triton_jit {
namespace {

detail::LaunchHooksSnapshot launch_hooks;

template <typename Update>
void update_launch_hooks(Update&& update) {
  detail::LaunchHooksSnapshot current =
      std::atomic_load_explicit(&launch_hooks, std::memory_order_acquire);
  while (true) {
    auto next = std::make_shared<detail::LaunchHooksState>();
    if (current) {
      *next = *current;
    }
    update(*next);

    detail::LaunchHooksSnapshot desired;
    if (next->enter || next->exit) {
      desired = std::move(next);
    }

    if (std::atomic_compare_exchange_weak_explicit(&launch_hooks,
                                                   &current,
                                                   desired,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) {
      return;
    }
  }
}

}  // namespace

namespace detail {

LaunchHooksSnapshot get_launch_hooks_snapshot() {
  return std::atomic_load_explicit(&launch_hooks, std::memory_order_acquire);
}

}  // namespace detail

void set_launch_enter_hook(LaunchHook hook) {
  update_launch_hooks([&hook](detail::LaunchHooksState& hooks) { hooks.enter = hook; });
}

void set_launch_exit_hook(LaunchHook hook) {
  update_launch_hooks([&hook](detail::LaunchHooksState& hooks) { hooks.exit = hook; });
}

void clear_launch_hooks() {
  std::atomic_store_explicit(
      &launch_hooks, detail::LaunchHooksSnapshot {}, std::memory_order_release);
}

}  // namespace triton_jit
