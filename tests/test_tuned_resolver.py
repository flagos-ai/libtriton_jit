# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tests for scripts/tuned_resolver.py with stand-in tuner objects: the
resolver must reuse the tuner's own key, caches, pruning and policy, store the
best config under the normalised key, and describe it the way TunedTable
expects. For plain @triton.autotune tuners it must drive the tuner's own run()
(a stand-in with Triton 3.6's control flow, and a 3.1-style one) and intercept
only the final launch. No FlagGems, Triton or device needed.

Usage: python tests/test_tuned_resolver.py <path to scripts dir>
"""

from __future__ import annotations

import gc
import importlib.util
import math
import os
import sys
import tempfile
import types
from itertools import starmap
from typing import Any, Dict, List
from unittest.mock import patch

failures = 0


def check(condition: bool, message: str) -> None:
    global failures
    if not condition:
        failures += 1
        print(f"FAILED: {message}", file=sys.stderr)


def load(scripts_dir: str, name: str):
    path = os.path.join(scripts_dir, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---- stand-ins --------------------------------------------------------------


class Config:
    def __init__(
        self, kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None
    ):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages
        self.num_ctas, self.maxnreg, self.pre_hook = num_ctas, maxnreg, pre_hook

    def all_kwargs(self):
        return {
            **self.kwargs,
            "num_warps": self.num_warps,
            "num_ctas": self.num_ctas,
            "num_stages": self.num_stages,
            "maxnreg": self.maxnreg,
        }

    def __repr__(self):
        return f"Config({self.kwargs}, nw={self.num_warps})"


class JITFunction:
    def __init__(self, fn, arg_names):
        self.fn, self.arg_names, self.cache_key = fn, arg_names, "ck"
        self.launches: List[Any] = []

    def run(self, *args, **kwargs):
        self.launches.append((args, kwargs))
        return "kernel"


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


class KVCache:
    def __init__(self):
        self.rows: Dict[Any, Any] = {}

    def get(self, key):
        return self.rows.get(key)

    def __getitem__(self, key):
        return self.rows[key]

    def __setitem__(self, key, value):
        self.rows[key] = value

    def __contains__(self, key):
        return key in self.rows


class LibCache:
    def __init__(self):
        self.tables: Dict[Any, KVCache] = {}

    def __getitem__(self, key):
        return self.tables.setdefault(key, KVCache())


def default_strategy(key):
    return key


def log2_strategy(key):
    return 2 ** math.ceil(math.log2(key))


def align32_strategy(key):
    if key == 0:
        return 0
    if key < 32:
        return 2 ** math.ceil(math.log2(key))
    return math.ceil(key / 32) * 32


class LibTuner:
    _strategy_table = {
        None: default_strategy,
        "default": default_strategy,
        "log": log2_strategy,
        "align32": align32_strategy,
    }

    def __init__(self, name, fn, keys, strategy, configs, timings):
        self.__name__, self.fn, self.keys, self.strategy, self.configs = (
            name,
            fn,
            keys,
            strategy,
            configs,
        )
        self.arg_names = fn.arg_names
        self.configs_hash = "h" * 32
        self.config_table_name = f"{name}_hash"
        self.benchmark_table_name = f"{name}_bench"
        self.cache = KVCache()
        self.timings = timings  # config index -> p50
        self.trace = (
            {}
        )  # Shared observer; invocation state lives on a local tuner copy.
        self.bench_calls: List[Any] = []
        self.pre_hook_calls = 0
        self.policy_calls = 0
        self.nargs = None

    @property
    def policy_calls(self):
        return self.trace.get("policy", 0)

    @policy_calls.setter
    def policy_calls(self, value):
        self.trace["policy"] = value

    @property
    def pre_hook_calls(self):
        return self.trace.get("hook", 0)

    @pre_hook_calls.setter
    def pre_hook_calls(self, value):
        self.trace["hook"] = value

    @property
    def last_bench_args(self):
        return self.trace["args"]

    @last_bench_args.setter
    def last_bench_args(self, value):
        self.trace["args"] = value

    def get_key(self, args):
        if self.strategy is None:
            key = tuple(args[k] for k in self.keys if k in args)
        else:
            key = tuple(
                starmap(lambda i, k: self.strategy[i](args[k]), enumerate(self.keys))
            )
        key += tuple(str(v.dtype) for v in args.values() if isinstance(v, Tensor))
        return key

    def get_benchmark_key(self, args):
        return tuple(args[k] for k in self.keys if k in args) + ("proto",)

    def prune_configs(self, kwargs):
        return [
            c
            for c in self.configs
            if c.kwargs.get("BLOCK_M", 0) <= kwargs.get("max_block", 1 << 30)
        ]

    def _bench(self, *args, config, **kwargs):
        self.bench_calls.append(config)
        assert callable(kwargs.get("grid")) or isinstance(
            kwargs.get("grid"), tuple
        ), "grid must reach _bench"
        assert kwargs.get("warmup") is False, "warmup=False must reach _bench"
        self.last_bench_args = args
        p50 = self.timings[self.configs.index(config)]
        return (p50, p50 * 0.9, p50 * 1.1)

    def policy(self, bench_fn, configs, args, kwargs):
        self.policy_calls += 1
        timings = {c: bench_fn(c)[0] for c in configs}
        best = min(timings, key=timings.get)
        return best, timings

    def pre_hook(self, full_nargs, reset_only=False):
        self.pre_hook_calls += 1


class NativeTensor(Tensor):
    """A tensor stand-in with the in-place methods reset/restore hooks use."""

    def __init__(self, dtype, value=0):
        super().__init__(dtype)
        self.value = value

    def zero_(self):
        self.value = 0

    def clone(self):
        return NativeTensor(self.dtype, self.value)

    def copy_(self, other):
        self.value = other.value


class NativeAutotuner:
    """Stand-in for triton.runtime.autotuner.Autotuner: the control flow of
    Triton 3.6's run() / _bench() / check_disk_cache(), with launches recorded
    by the JITFunction stand-in instead of executed. style="3.1" mimics the
    older layout: key_idx instead of keys, hooks receive the positional tuple."""

    def __init__(
        self,
        fn,
        keys,
        configs,
        timings,
        reset_to_zero=(),
        restore_value=(),
        pre_hook=None,
        cache_results=False,
        disk=None,
        style="3.6",
    ):
        self.fn = fn
        self.arg_names = list(fn.arg_names)
        self.base_fn = fn.fn
        self.configs = list(configs)
        self.style = style
        if style == "3.1":
            self.key_idx = [self.arg_names.index(k) for k in keys]
        else:
            self.keys = list(keys)
        self.cache = {}
        self.timings = timings
        self.pre_hook = pre_hook or (lambda kwargs, reset_only=False: 0)
        self.post_hook = lambda kwargs, exception: 0
        if style == "3.1":
            self.reset_idx = [self.arg_names.index(name) for name in reset_to_zero]
            self.restore_idx = [self.arg_names.index(name) for name in restore_value]

            def builtin_pre(args, reset_only=False):
                for index in self.reset_idx:
                    args[index].zero_()
                if not reset_only:
                    self.restore_copies = [
                        args[index].clone() for index in self.restore_idx
                    ]

            def builtin_post(args, exception=None):
                for index, copy in zip(self.restore_idx, self.restore_copies):
                    args[index].copy_(copy)
                self.restore_copies = []

            if pre_hook is None and (self.reset_idx or self.restore_idx):
                self.pre_hook = builtin_pre
            if self.restore_idx:
                self.post_hook = builtin_post
        else:
            self.reset_to_zero = list(reset_to_zero)
            self.restore_value = list(restore_value)
            self.user_defined_pre_hook = pre_hook is not None
            self.user_defined_post_hook = False
        self.nargs = None
        self.best_config = None
        self.cache_results = cache_results
        self.disk = disk if disk is not None else {}
        self.trace = {"bench": [], "hook_reset": 0}

    def _key(self, args, kwargs):
        all_args = {**self.nargs, **kwargs}
        if self.style == "3.1":
            values = [all_args[name] for name in self.arg_names if name in all_args]
            key = [values[i] for i in self.key_idx]
        else:
            named = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key = [named[k] for k in self.keys if k in named]
            values = list(named.values())
        for arg in values:
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        return tuple(key)

    def prune_configs(self, kwargs):
        return [
            c
            for c in self.configs
            if c.kwargs.get("BLOCK", 0) <= kwargs.get("max_block", 1 << 30)
        ]

    def _bench(self, *args, config, **meta):
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}
        self.trace["bench"].append(config)
        hook_arg = args if self.style == "3.1" else full_nargs

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(hook_arg)
            self.fn.run(*args, **current)
            self.post_hook(hook_arg, exception=None)

        for _ in range(3):  # do_bench repeats the call
            kernel_call()
        p50 = self.timings[self.configs.index(config)]
        return [p50, p50 * 0.9, p50 * 1.1]

    def check_disk_cache(self, key, configs, bench_fn):
        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        hit = self.disk.get(key)
        if hit is not None:
            self.cache[key] = hit
            return True
        bench_fn()
        self.disk[key] = self.cache[key]
        return False

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        if len(self.configs) > 1:
            key = self._key(args, kwargs)
            if key not in self.cache:
                pruned = self.prune_configs(kwargs)

                def benchmark():
                    timings = {
                        c: self._bench(*args, config=c, **kwargs) for c in pruned
                    }
                    self.cache[key] = min(timings, key=timings.get)
                    full_nargs = {
                        **self.nargs,
                        **kwargs,
                        **self.cache[key].all_kwargs(),
                    }
                    self.pre_hook(
                        args if self.style == "3.1" else full_nargs, reset_only=True
                    )
                    self.trace["hook_reset"] += 1

                if self.cache_results:
                    self.check_disk_cache(key, pruned, benchmark)
                else:
                    benchmark()
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret


class DriftingAutotuner(NativeAutotuner):
    """A layout the adapter must refuse: run() times candidates with direct
    launches instead of _bench()."""

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        key = self._key(args, kwargs)
        if key not in self.cache:
            for config in self.prune_configs(kwargs):
                self.fn.run(*args, **kwargs, **config.all_kwargs())
            self.cache[key] = self.configs[0]
        self.best_config = self.cache[key]
        ret = self.fn.run(*args, **kwargs, **self.best_config.all_kwargs())
        self.nargs = None
        return ret


def fake_libentry(package):
    module = types.ModuleType(f"{package}.utils.libentry")
    module.LibTuner = LibTuner
    module.libcache = LibCache()
    return module


def mm_kernel(a, b, c, M, N, K, BLOCK_M, BLOCK_N, EVEN_K):  # noqa: N803
    pass


def vector_kernel(x, y, n, BLOCK):  # noqa: N803
    pass


def scale_kernel(x, y, n, SCALE, BLOCK):  # noqa: N803
    pass


def run(scripts_dir: str) -> int:
    load(scripts_dir, "export_tuned_table")
    resolver = load(scripts_dir, "tuned_resolver")
    libentry = fake_libentry("flag_gems")
    configs = [
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "EVEN_K": True}, num_warps=8, num_ctas=2
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "EVEN_K": False}, num_warps=4, maxnreg=128
        ),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "EVEN_K": True}, num_warps=8),
    ]
    tuner = LibTuner(
        "mm",
        JITFunction(
            mm_kernel, ["a", "b", "c", "M", "N", "K", "BLOCK_M", "BLOCK_N", "EVEN_K"]
        ),
        ["M", "N", "K"],
        [log2_strategy, log2_strategy, align32_strategy],
        configs,
        timings=[5.0, 3.0, 9.0],
    )
    a, b, c = Tensor("torch.float16"), Tensor("torch.float16"), Tensor("torch.float32")
    args = (a, b, c, 1000, 4000, 4090)
    kwargs = {"max_block": 200}

    # an untuned key without a grid is reported, not benchmarked
    nogrid = resolver.resolve_with_tuner(tuner, libentry, args, kwargs)
    check(
        "supplied no launch grid" in nogrid.get("unsupported", ""),
        f"no-grid answer: {nogrid}",
    )
    check(tuner.policy_calls == 0, "no benchmark without a grid")
    grid_expr = "(math.ceil(M / META['BLOCK_M']) * math.ceil(N / META['BLOCK_N']),)"
    out = resolver.resolve_with_tuner(tuner, libentry, args, kwargs, grid=grid_expr)
    check("unsupported" not in out, f"resolved: {out}")
    check(
        out.get("source_path", "").endswith("test_tuned_resolver.py"),
        "source_path reported",
    )
    check(out["kernel_id"] == "mm_kernel" and out["op_name"] == "mm", "identity")
    check(
        out["key"]
        == [1024, 4096, 4096, "torch.float16", "torch.float16", "torch.float32"],
        f"normalised key {out['key']}",
    )
    check(
        [c["strategy"] for c in out["key_columns"]] == ["log", "log", "align32"],
        "strategies",
    )
    check(out["dtype_keys"] == 3, "dtype keys")
    check(
        out["kwargs"] == [["BLOCK_M", 64], ["BLOCK_N", 64], ["EVEN_K", False]],
        f"kwargs {out['kwargs']}",
    )
    check(out["num_warps"] == 4 and out["num_stages"] == 3, "nw/ns")
    check(out["extra"] == {"maxnreg": "128"}, f"extra {out['extra']}")
    # pruning happened through the tuner (BLOCK_M 256 excluded), policy picked the fastest, key stored normalised
    check(
        tuner.policy_calls == 1 and len(tuner.bench_calls) == 2,
        f"policy/bench calls {tuner.policy_calls}/{len(tuner.bench_calls)}",
    )
    check(
        (1024, 4096, 4096, "torch.float16", "torch.float16", "torch.float32")
        in tuner.cache,
        "stored under get_key()",
    )
    check(
        tuner.pre_hook_calls == 1 and tuner.nargs is None,
        "pre_hook reset called, nargs cleared",
    )
    grid_fn = resolver.build_grid(grid_expr, tuner, args, {})
    check(
        grid_fn({"BLOCK_M": 64, "BLOCK_N": 64}) == (16 * 63,),
        f"grid expression evaluated: {grid_fn({'BLOCK_M': 64, 'BLOCK_N': 64})}",
    )
    check(
        resolver.build_grid((4, 1), tuner, args, {}) == (4, 1), "tuple grid passthrough"
    )
    # second call with a shape in the same buckets: cache hit, no benchmark
    out2 = resolver.resolve_with_tuner(
        tuner, libentry, (a, b, c, 1024, 4096, 4096), kwargs
    )
    check(
        out2["kwargs"] == out["kwargs"] and tuner.policy_calls == 1,
        "cache hit skips policy",
    )
    # benchmark cache is reused for the same raw key even when the config cache is bypassed
    bench_table = libentry.libcache[
        tuner.benchmark_table_name, (1000, 4000, 4090, "proto")
    ]
    check(len(bench_table.rows) == 2, f"benchmark cache rows {len(bench_table.rows)}")
    # different dtype -> different key -> benchmark again
    out3 = resolver.resolve_with_tuner(
        tuner,
        libentry,
        (Tensor("torch.bfloat16"), b, c, 1000, 4000, 4090),
        kwargs,
        grid=grid_expr,
    )
    check(
        out3["key"][3] == "torch.bfloat16" and tuner.policy_calls == 2,
        "dtype changes the key",
    )
    # a config with pre_hook is refused
    hooked = LibTuner(
        "mm_hooked",
        tuner.fn,
        ["M"],
        None,
        [
            Config(
                {"BLOCK_M": 8, "BLOCK_N": 8, "EVEN_K": True}, pre_hook=lambda n: None
            ),
            Config({"BLOCK_M": 16, "BLOCK_N": 8, "EVEN_K": True}),
        ],
        timings=[1.0, 2.0],
    )
    out4 = resolver.resolve_with_tuner(hooked, libentry, args, {}, grid=(1,))
    check("pre_hook" in out4.get("unsupported", ""), f"pre_hook refused: {out4}")
    # single-config tuner: no benchmarking, config[0]
    single = LibTuner(
        "mm_single",
        tuner.fn,
        ["M"],
        None,
        [Config({"BLOCK_M": 32, "BLOCK_N": 32, "EVEN_K": True})],
        timings=[1.0],
    )
    out5 = resolver.resolve_with_tuner(single, libentry, args, {})
    check(
        out5["kwargs"][0] == ["BLOCK_M", 32] and single.policy_calls == 0,
        "single config short-circuits",
    )
    # a kwarg that is not a kernel parameter is refused rather than silently dropped
    odd = LibTuner(
        "mm_odd",
        tuner.fn,
        ["M"],
        None,
        [
            Config({"BLOCK_M": 32, "BLOCK_N": 32, "EVEN_K": True, "GHOST": 1}),
            Config({"BLOCK_M": 64, "BLOCK_N": 32, "EVEN_K": True, "GHOST": 2}),
        ],
        timings=[2.0, 1.0],
    )
    out6 = resolver.resolve_with_tuner(odd, libentry, args, {}, grid=(1,))
    check("GHOST" in out6.get("unsupported", ""), f"ghost kwarg refused: {out6}")
    # ---- find_tuner: same kernel name in two files, and the same file loaded twice ----
    import gc

    export = sys.modules["export_tuned_table"]
    pkg = types.ModuleType("flag_fake")
    pkg.__path__ = []
    utils = types.ModuleType("flag_fake.utils")
    utils.__path__ = []
    sys.modules["flag_fake"] = pkg
    sys.modules["flag_fake.utils"] = utils
    sys.modules["flag_fake.utils.libentry"] = libentry
    this_file = os.path.abspath(__file__)

    def kernel_in(module_name):
        def dup_kernel(a, M, BLOCK):  # noqa: N803
            pass

        dup_kernel.__module__ = module_name
        return dup_kernel

    # a registered package module whose __file__ is this test file
    ops_mod = types.ModuleType("flag_fake.ops.dup")
    ops_mod.__file__ = this_file
    sys.modules["flag_fake.ops.dup"] = ops_mod
    registered_fn = kernel_in("flag_fake.ops.dup")
    stray_fn = kernel_in("dup")  # what a second spec_from_file_location load looks like
    registered = LibTuner(
        "dup",
        JITFunction(registered_fn, ["a", "M", "BLOCK"]),
        ["M"],
        None,
        [Config({"BLOCK": 1}), Config({"BLOCK": 2})],
        [1.0, 2.0],
    )
    registered.base_fn = registered_fn
    stray = LibTuner(
        "dup",
        JITFunction(stray_fn, ["a", "M", "BLOCK"]),
        ["M"],
        None,
        [Config({"BLOCK": 1}), Config({"BLOCK": 2})],
        [1.0, 2.0],
    )
    stray.base_fn = stray_fn
    gc.collect()
    found = resolver.find_tuner("flag_fake", "dup_kernel", source_path=this_file)
    check(found is registered, "find_tuner prefers the copy imported under the package")
    check(
        resolver.find_tuner("flag_fake", "dup_kernel") is registered,
        "without source_path the same single-file case still resolves",
    )
    check(
        export.dedupe_tuners([stray, registered], "flag_fake") == [registered],
        "exporter dedupes copies of one file",
    )
    try:
        resolver.find_tuner("flag_fake", "no_such_kernel")
        check(False, "unknown kernel must raise")
    except LookupError:
        pass
    run_native_checks(resolver, this_file)
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("tuned_resolver: all checks passed")
    return 0


def run_native_checks(resolver, this_file: str) -> None:
    """Plain @triton.autotune tuners: run() is driven, the final launch intercepted."""

    def native_grid(meta):
        return (math.ceil(257 / meta["BLOCK"]),)

    def make_native(**overrides):
        fn = JITFunction(vector_kernel, ["x", "y", "n", "BLOCK"])
        options = dict(
            keys=["n"],
            configs=[
                Config({"BLOCK": 128}),
                Config({"BLOCK": 256}, num_warps=8),
                Config({"BLOCK": 512}),
            ],
            timings=[3.0, 1.0, 2.0],
        )
        options.update(overrides)
        return NativeAutotuner(fn, **options)

    x, y = NativeTensor("torch.float32"), NativeTensor("torch.float32", 17)
    # what native run() does: choice, key, candidate order, launches
    native = make_native()
    native.run(x, y, 257, grid=native_grid, warmup=False)
    native_choice, native_key = native.best_config.all_kwargs(), list(native.cache)[0]
    native_order = [c.kwargs["BLOCK"] for c in native.trace["bench"]]
    native_launches = len(native.fn.launches)
    # the adapter reproduces it through run() itself, minus the final launch
    tuner = make_native()
    out = resolver.resolve_with_native_tuner(
        tuner, (x, y, 257), {}, grid="(math.ceil(n / META['BLOCK']),)"
    )
    check("unsupported" not in out, f"native resolved: {out}")
    check(
        out.get("kwargs") == [["BLOCK", native_choice["BLOCK"]]]
        and out.get("num_warps") == native_choice["num_warps"],
        f"native choice {out.get('kwargs')} nw={out.get('num_warps')}",
    )
    check(
        tuple(out.get("key", ())) == native_key
        and out.get("key") == [257, "torch.float32", "torch.float32"],
        f"native key {out.get('key')} vs {native_key}",
    )
    check(
        [c["strategy"] for c in out["key_columns"]] == ["default"]
        and [c["name"] for c in out["key_columns"]] == ["n"]
        and out["dtype_keys"] == 2,
        "native key schema",
    )
    check(
        [c.kwargs["BLOCK"] for c in tuner.trace["bench"]] == native_order,
        "native candidate order",
    )
    check(
        len(tuner.fn.launches) == native_launches - 1,
        f"benchmark launches forwarded, final one intercepted: "
        f"{len(tuner.fn.launches)} vs native {native_launches}",
    )
    check(
        native_key in tuner.cache and tuner.nargs is None,
        "stored in Triton's own cache; call state cleared",
    )
    check(tuner.trace["hook_reset"] == 1, "run() applied its reset hook once")
    # warm: Triton's cache answers, no grid needed, nothing benchmarked
    before = len(tuner.trace["bench"])
    with patch.object(
        resolver,
        "_argument_cloner",
        side_effect=AssertionError("warm hit cloned inputs"),
    ):
        warm = resolver.resolve_with_native_tuner(tuner, (x, y, 257), {})
    check(
        warm.get("kwargs") == out["kwargs"] and len(tuner.trace["bench"]) == before,
        f"native warm hit: {warm}",
    )
    # cold without a grid is reported, not benchmarked
    cold = resolver.resolve_with_native_tuner(make_native(), (x, y, 257), {})
    check(
        "supplied no launch grid" in cold.get("unsupported", ""),
        f"native no-grid: {cold}",
    )
    # single config short-circuits without run()
    single = make_native(configs=[Config({"BLOCK": 64})], timings=[1.0])
    out_single = resolver.resolve_with_native_tuner(single, (x, y, 257), {})
    check(
        out_single.get("kwargs") == [["BLOCK", 64]]
        and not single.fn.launches
        and out_single.get("key") == [257, "torch.float32", "torch.float32"],
        f"native single config: {out_single}",
    )
    # reset_to_zero: run()'s reset is applied to what _bench saw; no launch contract
    reset_tuner = make_native(reset_to_zero=["y"])
    out_reset = resolver.resolve_with_native_tuner(
        reset_tuner, (x, NativeTensor("torch.float32", 17), 257), {}, grid=native_grid
    )
    check(
        "unsupported" not in out_reset and reset_tuner.trace["hook_reset"] == 1,
        f"reset tuner resolved: {out_reset}",
    )
    # 3.1 layout: key_idx instead of keys, hooks receive the positional tuple
    with patch.object(
        resolver, "_native_autotuner_class", return_value=NativeAutotuner
    ):
        old = make_native(style="3.1", reset_to_zero=["y"])
        out_old = resolver.resolve_with_native_tuner(
            old, (x, NativeTensor("torch.float32", 17), 257), {}, grid=native_grid
        )
        check(
            "unsupported" not in out_old
            and out_old.get("key") == [257, "torch.float32", "torch.float32"]
            and [c["name"] for c in out_old.get("key_columns", [])] == ["n"],
            f"3.1-style tuner: {out_old}",
        )
        restored = make_native(style="3.1", restore_value=["y"])
        local = resolver._local_tuner(restored)
        scratch, live = NativeTensor("torch.float32", 3), NativeTensor(
            "torch.float32", 7
        )
        local.pre_hook((x, scratch, 257))
        restored.pre_hook((x, live, 257))
        scratch.value, live.value = 33, 77
        local.post_hook((x, scratch, 257), exception=None)
        restored.post_hook((x, live, 257), exception=None)
        check(
            scratch.value == 3 and live.value == 7, "3.1 restore closures are isolated"
        )
        custom = make_native(style="3.1", pre_hook=lambda args, reset_only=False: None)
        rejected = resolver.resolve_with_native_tuner(
            custom, (x, y, 257), {}, grid=native_grid
        )
        check(
            "custom tuner hooks" in rejected.get("unsupported", "")
            and not custom.trace["bench"],
            "3.1 custom hook refused before benchmarking",
        )
    # disk cache: a second tuner sharing the disk needs neither grid nor benchmark
    disk = {}
    first = make_native(cache_results=True, disk=disk)
    resolver.resolve_with_native_tuner(first, (x, y, 257), {}, grid=native_grid)
    second = make_native(cache_results=True, disk=disk)
    with patch.object(
        resolver,
        "_argument_cloner",
        side_effect=AssertionError("disk hit cloned inputs"),
    ):
        out_disk = resolver.resolve_with_native_tuner(second, (x, y, 257), {})
    check(
        out_disk.get("kwargs") == out["kwargs"] and not second.trace["bench"],
        f"native disk hit: {out_disk}",
    )
    # refusals: custom hooks, config pre_hook, float key column, unknown control flow
    hooked = make_native(pre_hook=lambda kwargs, reset_only=False: None)
    out_hooked = resolver.resolve_with_native_tuner(
        hooked, (x, y, 257), {}, grid=native_grid
    )
    check(
        "custom tuner hooks" in out_hooked.get("unsupported", ""),
        f"custom pre_hook refused: {out_hooked}",
    )
    hooked.cache[(257, "torch.float32", "torch.float32")] = hooked.configs[0]
    exported_hook = resolver.native_kernel_table(hooked)
    check(
        exported_hook.get("unsupported") == out_hooked["unsupported"]
        and not exported_hook["entries"],
        "custom hooks have the same online and offline refusal",
    )
    cfg_hooked = make_native(
        configs=[Config({"BLOCK": 8}, pre_hook=lambda n: None), Config({"BLOCK": 16})],
        timings=[1.0, 2.0],
    )
    out_cfg = resolver.resolve_with_native_tuner(
        cfg_hooked, (x, y, 257), {}, grid=native_grid
    )
    check(
        "pre_hook" in out_cfg.get("unsupported", ""),
        f"config pre_hook refused: {out_cfg}",
    )
    scaled = NativeAutotuner(
        JITFunction(scale_kernel, ["x", "y", "n", "SCALE", "BLOCK"]),
        ["SCALE"],
        [Config({"BLOCK": 8}), Config({"BLOCK": 16})],
        [1.0, 2.0],
    )
    out_scaled = resolver.resolve_with_native_tuner(
        scaled, (x, y, 257, 0.5), {}, grid=native_grid
    )
    check(
        "SCALE" in out_scaled.get("unsupported", "")
        and "TuneKeyView" in out_scaled.get("unsupported", "")
        and not scaled.trace["bench"],
        f"float key refused before benchmarking: {out_scaled}",
    )
    drifting = DriftingAutotuner(
        JITFunction(vector_kernel, ["x", "y", "n", "BLOCK"]),
        ["n"],
        [Config({"BLOCK": 8}), Config({"BLOCK": 16})],
        [1.0, 2.0],
    )
    out_drift = resolver.resolve_with_native_tuner(
        drifting, (x, y, 257), {}, grid=native_grid
    )
    check(
        "outside _bench" in out_drift.get("unsupported", "") and not drifting.cache,
        f"unknown control flow refused, cache left clean: {out_drift} {drifting.cache}",
    )
    # ---- with a stand-in triton: dtype markers, resolve() dispatch, export ----
    fake_tl = types.ModuleType("triton.language")

    class FakeDtype:
        pass

    fake_tl.dtype = FakeDtype
    fake_tl.float32 = FakeDtype()
    fake_triton = types.ModuleType("triton")
    fake_triton.__version__ = "3.6.0"
    fake_triton.__path__ = []
    fake_runtime = types.ModuleType("triton.runtime")
    fake_runtime.__path__ = []
    fake_autotuner = types.ModuleType("triton.runtime.autotuner")
    fake_autotuner.Autotuner = NativeAutotuner
    names = ("triton", "triton.language", "triton.runtime", "triton.runtime.autotuner")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(
        {
            "triton": fake_triton,
            "triton.language": fake_tl,
            "triton.runtime": fake_runtime,
            "triton.runtime.autotuner": fake_autotuner,
        }
    )
    try:
        check(
            resolver.TritonDtype("float32").materialize() is fake_tl.float32
            and resolver.TritonDtype("tl.float32").materialize() is fake_tl.float32,
            "TritonDtype materialises with or without the tl. prefix",
        )
        try:
            resolver.TritonDtype("float99").materialize()
            check(False, "unknown dtype must raise")
        except ValueError:
            pass
        # resolve(): a package without libentry, tuner found by kernel and file,
        # dtype marker applied before the tuner sees the arguments
        pkg = types.ModuleType("native_fake")
        pkg.__path__ = []
        sys.modules["native_fake"] = pkg
        ops = types.ModuleType("native_fake.ops.vec")
        ops.__file__ = this_file
        sys.modules["native_fake.ops.vec"] = ops

        def cast_kernel(x, y, n, DTYPE, BLOCK):  # noqa: N803
            pass

        cast_kernel.__module__ = "native_fake.ops.vec"
        casting = NativeAutotuner(
            JITFunction(cast_kernel, ["x", "y", "n", "DTYPE", "BLOCK"]),
            ["n"],
            [Config({"BLOCK": 8}), Config({"BLOCK": 16})],
            [2.0, 1.0],
        )
        gc.collect()
        out_pkg = resolver.resolve(
            "native_fake",
            "cast_kernel",
            (x, y, 257),
            {"DTYPE": resolver.TritonDtype("float32")},
            source_path=this_file,
            grid=native_grid,
            device_index=0,
            stream=0,
        )
        check(
            "unsupported" not in out_pkg
            and out_pkg.get("kwargs") == [["BLOCK", 16]]
            and out_pkg.get("key") == [257, "torch.float32", "torch.float32"],
            f"resolve() native dispatch: {out_pkg}",
        )
        seen = [kw.get("DTYPE") for _, kw in casting.fn.launches]
        check(
            bool(seen) and all(v is fake_tl.float32 for v in seen),
            f"DTYPE materialised before the launches: {seen[:1]}",
        )
        check(
            resolver.find_tuner("native_fake", "cast_kernel", source_path=this_file)
            is casting,
            "find_tuner returns the native tuner",
        )

        # inside a libentry package LibTuner kernels keep their path, native ones fall through
        def native_in_pkg(x, n, BLOCK):  # noqa: N803
            pass

        native_in_pkg.__module__ = "flag_fake.ops.dup"
        in_pkg = NativeAutotuner(
            JITFunction(native_in_pkg, ["x", "n", "BLOCK"]),
            ["n"],
            [Config({"BLOCK": 8}), Config({"BLOCK": 16})],
            [1.0, 2.0],
        )
        gc.collect()
        out_mixed = resolver.resolve(
            "flag_fake",
            "native_in_pkg",
            (x, 257),
            {},
            source_path=this_file,
            grid=native_grid,
        )
        check(
            out_mixed.get("kwargs") == [["BLOCK", 8]]
            and [c["strategy"] for c in out_mixed.get("key_columns", [])]
            == ["default"],
            f"native tuner inside a libentry package: {out_mixed}",
        )
        out_lib = resolver.resolve(
            "flag_fake",
            "dup_kernel",
            (Tensor("torch.float32"), 4),
            {},
            source_path=this_file,
            grid=(1,),
        )
        check(
            out_lib.get("op_name") == "dup" and "unsupported" not in out_lib,
            f"LibTuner dispatch through resolve(): {out_lib}",
        )
        check(
            in_pkg.trace["bench"] and not in_pkg.fn.launches[len(in_pkg.fn.launches) :],
            "native tuner in package benchmarked",
        )
        # export: canonical entries resolved here, plus unambiguous Python-tuned ones
        casting.cache[(999, "torch.float16", "torch.float16")] = casting.configs[0]
        casting.cache[(998, "torch.float16", "torch.float32")] = casting.configs[0]
        logged = []
        kernel = resolver.native_kernel_table(casting, log=logged.append)
        exported_keys = sorted(tuple(e["key"]) for e in kernel["entries"])
        check(
            exported_keys
            == [
                (257, "torch.float32", "torch.float32"),
                (999, "torch.float16", "torch.float16"),
            ]
            and kernel["dtype_keys"] == 2
            and kernel["cache_namespace"] == this_file
            and kernel["key_columns"] == [{"name": "n", "strategy": "default"}],
            f"native export entries {exported_keys} {kernel['dtype_keys']}",
        )
        check(
            len(logged) == 1 and "998" in logged[0],
            f"ambiguous entry reported: {logged}",
        )
        table = resolver.export_native_table(
            os.path.join(tempfile.mkdtemp(), "native.json"),
            [casting],
            backend="CUDA",
            device_index=0,
        )
        check(
            table["format_version"] == 2
            and table["fingerprint"]["backend"] == "CUDA"
            and table["kernels"][0]["kernel_id"] == "cast_kernel"
            and len(table["kernels"][0]["entries"]) == 2,
            "export_native_table writes a v2 table",
        )
        duplicate = NativeAutotuner(casting.fn, ["n"], casting.configs, [2.0, 1.0])
        duplicate.cache[(1000, "torch.float32", "torch.float32")] = duplicate.configs[0]
        transient = resolver._local_tuner(casting)

        def foreign_kernel(x, y, n, BLOCK):  # noqa: N803
            pass

        foreign = NativeAutotuner(
            JITFunction(foreign_kernel, ["x", "y", "n", "BLOCK"]),
            ["n"],
            casting.configs,
            [2.0, 1.0],
        )
        package_table = resolver.export_native_table(
            os.path.join(tempfile.mkdtemp(), "package.json"),
            "native_fake",
            backend="CUDA",
        )
        check(
            len(package_table["kernels"]) == 1
            and len(package_table["kernels"][0]["entries"]) == 3,
            "package export filters foreign/transient tuners and merges duplicate rows",
        )
        check(
            transient not in resolver.native_tuners()
            and foreign not in resolver.native_tuners(package_name="native_fake"),
            "discovery excludes local copies and other packages",
        )
        duplicate.cache[(257, "torch.float32", "torch.float32")] = duplicate.configs[0]
        try:
            resolver.export_native_table(
                os.path.join(tempfile.mkdtemp(), "conflict.json"),
                [casting, duplicate],
                backend="CUDA",
            )
            check(False, "conflicting duplicate configs must be rejected")
        except ValueError as error:
            check("conflicting tuned configs" in str(error), str(error))
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
