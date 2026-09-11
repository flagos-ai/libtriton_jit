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
"""Select and cache a launch config with the tuner that owns the kernel.

Two kinds of tuner are served:

* ``LibTuner`` (FlagGems / FlagBLAS ``@libtuner``): the resolver mirrors the
  resolve half of ``LibTuner.run()`` and reuses its key strategies, SQLite
  config cache, benchmark cache, pruning and policy.
* plain ``triton.runtime.Autotuner`` (``@triton.autotune``): the resolver calls
  the tuner's own ``run()``, isolates its trial launches and intercepts the
  final business launch, so key derivation, pruning, disk cache and vendor
  specialisations stay whatever that Triton version does. Nothing of that
  control flow is re-implemented here.

Cold selection benchmarks isolated tensor views; it never performs the final
business launch. Output initialisation is the caller's contract: a kernel with
``reset_to_zero`` is launched on the caller's buffer as-is, exactly like a warm
native call. Unsupported cases return a reason; errors propagate.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
import threading
import types
import warnings
import weakref
from collections.abc import MutableMapping
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import export_tuned_table as _export  # noqa: E402  (same directory)
from export_tuned_table import kernel_identity, unwrap_to_jit_function  # noqa: E402

_tuner_index: Dict[Tuple[str, str, Optional[str]], Any] = {}
# Serialises benchmarks of @triton.autotune tuners that share no libentry lock.
_native_benchmark_lock = threading.RLock()
# tuner -> {canonical key: config}: what this process resolved for the C++ side,
# keyed the way TuneKeyView is built (see _canonical_native_key).
_native_records: "weakref.WeakKeyDictionary[Any, Dict[Tuple[Any, ...], Any]]" = (
    weakref.WeakKeyDictionary()
)
_warned: set = set()


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        print(f"[tuned_resolver] {message}", file=sys.stderr)


class TritonDtype(str):
    """A ``triton.language`` dtype handed over by the C++ bridge as a constexpr
    argument (``DTYPE=tl.float32``). ``ArgValue`` cannot carry the Python
    object, so the bridge sends the name ("float32", "tl.float32") and the
    resolver materialises it right before the tuner sees the arguments."""

    def materialize(self) -> Any:
        import triton.language as tl

        name = self[3:] if self.startswith("tl.") else str(self)
        value = getattr(tl, name, None)
        if value is None or not isinstance(value, tl.dtype):
            raise ValueError(f"'{name}' is not a triton.language dtype")
        return value


def _materialize(value: Any) -> Any:
    return value.materialize() if isinstance(value, TritonDtype) else value


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return a == b


def _module_is_registered(tuner: Any, package_name: str) -> bool:
    """True when the tuner's kernel comes from a module imported as part of the
    package (``flag_gems.ops.mm``), as opposed to a copy of the same file that
    libtriton_jit loaded on its own for launching (module ``mm``, absent from
    ``sys.modules``). Both wrap the same kernel; the package's copy is the one
    the Python side tunes with."""
    identity = kernel_identity(tuner)
    jit_fn, _ = unwrap_to_jit_function(tuner.fn)
    py_fn = getattr(jit_fn, "fn", jit_fn)
    module_name = getattr(getattr(tuner, "base_fn", py_fn), "__module__", "") or ""
    if module_name != package_name and not module_name.startswith(package_name + "."):
        return False
    module = sys.modules.get(module_name)
    if module is None:
        return False
    module_file = getattr(module, "__file__", None)
    return bool(module_file) and _same_file(module_file, identity["source_path"])


# --------------------------------------------------------------------------
# Discovery


def _libentry_module(package_name: str):
    """The package's libentry module, or None when the package has none (its
    kernels carry plain @triton.autotune). Other import failures propagate."""
    try:
        return _export.libentry_module_for(package_name)
    except ModuleNotFoundError as error:
        missing = getattr(error, "name", None) or ""
        if missing in (
            package_name,
            f"{package_name}.utils",
            f"{package_name}.utils.libentry",
        ):
            return None
        raise


def _native_autotuner_class():
    try:
        from triton.runtime.autotuner import Autotuner
    except Exception:  # noqa: BLE001 - without Triton there is nothing native to find
        return None
    return Autotuner


def native_tuners(
    libtuner_cls: Any = None, package_name: Optional[str] = None
) -> List[Any]:
    """Every live plain ``triton.runtime.Autotuner`` (LibTuner instances are
    excluded when `libtuner_cls` is given), in a stable order."""
    autotuner = _native_autotuner_class()
    if autotuner is None:
        return []
    with warnings.catch_warnings():
        # isinstance() on every live object trips deprecation proxies
        # (torch.distributed.reduce_op); they are not tuners.
        warnings.simplefilter("ignore")
        tuners = [
            obj
            for obj in gc.get_objects()
            if isinstance(obj, autotuner)
            and not (libtuner_cls is not None and isinstance(obj, libtuner_cls))
            and not getattr(obj, "_tuned_resolver_local", False)
        ]
    if package_name is not None:
        # Keep aliases of package kernels loaded by the C++ compiler, but not
        # unrelated tuners merely alive in the same process.
        owned = {
            _native_identity(t)
            for t in tuners
            if _module_is_registered(t, package_name)
        }
        tuners = [t for t in tuners if _native_identity(t) in owned]
    tuners.sort(
        key=lambda t: (getattr(getattr(t, "base_fn", None), "__name__", ""), id(t))
    )
    return tuners


def _native_identity(tuner: Any) -> Tuple[str, str]:
    identity = _export.kernel_identity(tuner)
    return os.path.realpath(identity["source_path"]), identity["kernel_id"]


def _matching(tuners: Iterable[Any], kernel_id: str, source_path: Optional[str]):
    matches = [
        t for t in tuners if _export.kernel_identity(t)["kernel_id"] == kernel_id
    ]
    if source_path:
        matches = [
            t
            for t in matches
            if _same_file(_export.kernel_identity(t)["source_path"], source_path)
        ]
    return matches


def find_tuner(
    package_name: str,
    kernel_id: str,
    source_path: Optional[str] = None,
    walk_ops: bool = True,
) -> Any:
    """The tuner whose kernel is the Triton function `kernel_id`: the package's
    LibTuner if one wraps it, otherwise the plain @triton.autotune tuner.

    `source_path` (the .py file the C++ side launches from) disambiguates
    kernels that share a name across files (FlagGems' per-arch overrides);
    among copies of the same file the one imported under the package wins.
    """
    key = (
        package_name,
        kernel_id,
        os.path.realpath(source_path) if source_path else None,
    )
    if key in _tuner_index:
        return _tuner_index[key]
    _export.import_package_tree(package_name, walk_ops=walk_ops)
    libentry = _libentry_module(package_name)
    libtuner_cls = getattr(libentry, "LibTuner", None) if libentry else None
    matches = []
    if libentry is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # same gc scan as native_tuners()
            matches = _matching(_export.find_tuners(libentry), kernel_id, source_path)
    if not matches:
        matches = _matching(
            native_tuners(libtuner_cls, package_name), kernel_id, source_path
        )
    if not matches:
        where = f" in {source_path}" if source_path else ""
        raise LookupError(
            f"no LibTuner or @triton.autotune tuner in {package_name} wraps a kernel "
            f"named '{kernel_id}'{where}"
        )
    if len(matches) > 1:
        preferred = [t for t in matches if _module_is_registered(t, package_name)]
        if preferred:
            matches = preferred
    if len(matches) > 1:
        files = {
            os.path.realpath(_export.kernel_identity(t)["source_path"]) for t in matches
        }
        if len(files) > 1:
            raise LookupError(
                f"kernel '{kernel_id}' is defined in several files of {package_name}: "
                + ", ".join(sorted(files))
                + "; pass source_path (ResolveArgs.source_path on the C++ side)"
            )
        # identical copies of one file (loaded more than once): any of them will do
    _tuner_index[key] = matches[0]
    return matches[0]


def build_grid(
    grid: Any, tuner: Any, args: Sequence[Any], kwargs: Dict[str, Any]
) -> Any:
    """Turn the caller's grid description into what the Triton autotuner needs.

    Accepts a callable (used as is), a tuple/list (constant grid), or a Python
    expression string evaluated with the kernel's argument names, `META`
    (the candidate config's kwargs), `triton` and `math` in scope, e.g.
    ``"(triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)"``.
    """
    if grid is None or callable(grid):
        return grid
    if isinstance(grid, (tuple, list)):
        return tuple(grid)
    if isinstance(grid, str):
        scope: Dict[str, Any] = {"math": math}
        try:
            import triton

            scope["triton"] = triton
        except ImportError:  # the expression may not need it
            pass
        scope.update(dict(zip(getattr(tuner, "arg_names", []), args)))
        scope.update({k: v for k, v in kwargs.items() if k not in ("grid", "warmup")})
        code = compile(grid, "<tuned_resolver grid>", "eval")

        def grid_fn(META):
            result = eval(
                code, {"__builtins__": {}}, {**scope, "META": META}
            )  # noqa: S307 - caller-supplied expression
            return (
                tuple(result) if isinstance(result, (tuple, list)) else (int(result),)
            )

        return grid_fn
    raise TypeError(
        f"grid must be a callable, a tuple, or an expression string, not {type(grid).__name__}"
    )


class _NeedsGrid(Exception):
    """Raised when a benchmark is required but no grid was given."""


class _UnsupportedBenchmark(Exception):
    pass


def _tuner_hook_reason(tuner: Any) -> Optional[str]:
    custom = "custom tuner hooks require a consumer launch hook and are unsupported"
    if getattr(tuner, "user_defined_pre_hook", False) or getattr(
        tuner, "user_defined_post_hook", False
    ):
        return custom
    if getattr(tuner, "shared_config_pre_hook", None) is not None:
        return "shared_config_pre_hook requires a consumer launch hook"
    if hasattr(tuner, "reset_idx") or hasattr(tuner, "restore_idx"):
        # Triton 3.1 has no user_defined_* flags. Accept only closures defined
        # by the installed Autotuner constructor, not arbitrary Python hooks.
        constructor = getattr(
            getattr(_native_autotuner_class(), "__init__", None), "__code__", None
        )
        builtin_codes = {
            item
            for item in getattr(constructor, "co_consts", ())
            if isinstance(item, types.CodeType)
        }
        for name in ("pre_hook", "post_hook"):
            if (
                getattr(getattr(tuner, name, None), "__code__", None)
                not in builtin_codes
            ):
                return custom
    return None


def _hook_names(tuner: Any, name: str, legacy_name: str) -> Tuple[str, ...]:
    values = getattr(tuner, name, None)
    if values is not None:
        return tuple(values)
    names = tuner.arg_names
    return tuple(names[index] for index in getattr(tuner, legacy_name, ()) or ())


def _local_tuner(tuner):
    """Own transient call state; share only the tuner's persistent caches/JIT.

    In particular, ordinary Python run() can continue using the original nargs.
    Triton's built-in reset/restore closures capture the original self, so they
    are rebuilt here against this invocation's state.
    """
    reason = _tuner_hook_reason(tuner)
    if reason is not None:
        raise _UnsupportedBenchmark(reason)
    local = copy.copy(tuner)
    local._tuned_resolver_local = True
    for name, value in vars(tuner).items():
        if isinstance(value, types.MethodType) and value.__self__ is tuner:
            setattr(local, name, types.MethodType(value.__func__, local))
    local.configs = list(tuner.configs)
    local.nargs = None
    reset = _hook_names(tuner, "reset_to_zero", "reset_idx")
    restore = _hook_names(tuner, "restore_value", "restore_idx")
    if reset or restore:
        copies = {}
        names = list(getattr(tuner, "arg_names", []) or [])

        def lookup(named, name):
            # Triton >= 3.2 hands the hooks the named arguments; 3.1 the
            # positional tuple _bench received.
            if isinstance(named, dict):
                return named[name]
            return named[names.index(name)]

        def pre_hook(named, reset_only=False):
            for name in reset:
                lookup(named, name).zero_()
            if not reset_only:
                copies.clear()
                copies.update((name, lookup(named, name).clone()) for name in restore)

        def post_hook(named, exception=None):
            for name in restore:
                lookup(named, name).copy_(copies[name])
            copies.clear()

        local.pre_hook, local.post_hook = pre_hook, post_hook
    return local


def _argument_cloner():
    """Copy backing storage once, retaining strides, offsets and aliases.

    A byte view avoids interpreting uninitialised output values and permits
    differently typed views of the same storage. All view construction is
    detached from autograd. A bounded budget prevents tiny slices of huge
    allocations from unexpectedly exhausting device memory.
    """
    try:
        import torch
    except ImportError:
        return lambda value: value
    budget = int(os.environ.get("TRITON_JIT_BENCH_MAX_BYTES", str(1024**3)))
    storages = {}
    used = 0

    def clone(value):
        nonlocal used
        if not torch.is_tensor(value):
            return value
        if (
            value.layout != torch.strided
            or value.is_conj()
            or value.is_neg()
            or value.is_quantized
        ):
            raise _UnsupportedBenchmark(
                "benchmark requires ordinary strided tensor views"
            )
        storage = value.untyped_storage()
        key = (value.device, storage._cdata)
        if key not in storages:
            size = storage.nbytes()
            used += size
            if used > budget:
                raise _UnsupportedBenchmark(
                    "benchmark storage copies exceed TRITON_JIT_BENCH_MAX_BYTES"
                )
            with torch.no_grad():
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                    storage, 0, (size,), (1,)
                )
                storages[key] = raw.clone().untyped_storage()
        with torch.no_grad():
            return torch.empty(0, dtype=value.dtype, device=value.device).set_(
                storages[key],
                value.storage_offset(),
                tuple(value.shape),
                tuple(value.stride()),
            )

    return clone


def _clone_arguments(args, kwargs):
    clone = _argument_cloner()
    return tuple(clone(value) for value in args), {
        name: clone(value) for name, value in kwargs.items()
    }


# --------------------------------------------------------------------------
# LibTuner selection


def _select_config(
    tuner: Any,
    libcache: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    launch_kwargs: Optional[Dict[str, Any]] = None,
    clone_tensors: bool = True,
) -> Tuple[Any, Tuple[Any, ...]]:
    """The resolve half of LibTuner.run(): returns (config, normalised key).

    `launch_kwargs` (grid / warmup) are only needed when a benchmark has to
    run; without them a cache miss raises _NeedsGrid. When benchmarking,
    tensor arguments are cloned first so the tuner's trial launches never
    touch the caller's live buffers.
    """
    tuner = _local_tuner(tuner)
    tuner.nargs = dict(zip(tuner.arg_names, args))
    try:
        all_args = {**tuner.nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in tuner.arg_names}
        config_key = tuner.get_key(_args)
        configs = list(getattr(tuner, "configs", []) or [])
        if len(configs) <= 1:
            return (configs[0] if configs else None), config_key
        if config_key in tuner.cache:
            config = tuner.cache[config_key]
        else:
            if not launch_kwargs:
                raise _NeedsGrid()
            bench_args, isolated_kwargs = (
                _clone_arguments(args, kwargs)
                if clone_tensors
                else (tuple(args), dict(kwargs))
            )
            bench_kwargs = {**isolated_kwargs, **launch_kwargs}
            tuner.nargs = dict(zip(tuner.arg_names, bench_args))
            if hasattr(tuner, "get_benchmark_key"):
                benchmark_key = tuner.get_benchmark_key(_args)
            else:
                benchmark_key = config_key
            bench_cache = libcache[tuner.benchmark_table_name, benchmark_key]
            pruned = tuner.prune_configs(bench_kwargs)

            def bench(config: Any) -> List[float]:
                ret = bench_cache.get(config)
                if ret is None:
                    ret = tuner._bench(*bench_args, config=config, **bench_kwargs)
                    if isinstance(ret, (int, float)):
                        ret = (ret, ret, ret)
                    if ret and all(math.isfinite(float(value)) for value in ret):
                        bench_cache[config] = tuple(ret)
                return list(ret)

            best_config, _timings = tuner.policy(
                bench, pruned, tuple(bench_args), bench_kwargs
            )
            tuner.cache[config_key] = best_config
            config = tuner.cache[config_key]
            full_nargs = {**tuner.nargs, **isolated_kwargs, **config.all_kwargs()}
            tuner.pre_hook(full_nargs, reset_only=True)
        # a config read back from the database has no pre_hook; recover the original
        if getattr(config, "pre_hook", None) is None:
            cached_kwargs = config.all_kwargs()
            for original in configs:
                if original.all_kwargs() == cached_kwargs:
                    config = original
                    break
        tuner.best_config = config
        return config, config_key
    finally:
        tuner.nargs = None


# --------------------------------------------------------------------------
# @triton.autotune selection


class _KeyRecorder(MutableMapping):
    """Stand-in for ``Autotuner.cache`` that shares the tuner's dict and
    remembers the key ``run()`` looked up or stored."""

    def __init__(self, backing: Dict[Any, Any]):
        self.backing = backing
        self.last_key: Any = None
        self.inserted = False

    def __getitem__(self, key):
        self.last_key = key
        return self.backing[key]

    def __setitem__(self, key, value):
        self.last_key = key
        self.inserted = key not in self.backing
        self.backing[key] = value

    def __delitem__(self, key):
        del self.backing[key]

    def __contains__(self, key):
        self.last_key = key
        return key in self.backing

    def __iter__(self):
        return iter(self.backing)

    def __len__(self):
        return len(self.backing)


class _LaunchInterceptor:
    """Stands in for ``tuner.fn`` while ``run()`` selects: launches made by
    ``_bench`` are forwarded, the final business launch is counted and
    skipped. Everything else is the wrapped function's."""

    def __init__(self, fn: Any, state: Dict[str, Any]):
        self.fn = fn
        self._state = state

    def __getattr__(self, name):
        return getattr(self.fn, name)

    def run(self, *args, **kwargs):
        if self._state["benchmarking"]:
            return self.fn.run(*args, **kwargs)
        self._state["final"] += 1
        return None


def _native_keys(tuner: Any) -> List[str]:
    keys = getattr(tuner, "keys", None)
    if keys is None:  # Triton <= 3.1 keeps the indices only
        names = list(getattr(tuner, "arg_names", []) or [])
        keys = [names[index] for index in getattr(tuner, "key_idx", []) or []]
    return list(keys)


def _native_support_reason(tuner: Any) -> Optional[str]:
    """None when the tuner has the shape of Autotuner.run() this adapter drives."""
    for attr in ("run", "_bench", "fn", "configs", "arg_names"):
        if not hasattr(tuner, attr):
            return (
                f"triton.runtime.Autotuner without '{attr}': this Triton version "
                "is not supported by the native adapter"
            )
    if not isinstance(getattr(tuner, "cache", None), dict):
        return "Autotuner.cache is not a dict: this Triton version is not supported by the native adapter"
    if getattr(tuner, "keys", None) is None and getattr(tuner, "key_idx", None) is None:
        return "Autotuner exposes neither keys nor key_idx"
    return _tuner_hook_reason(tuner)


def _canonical_native_key(
    tuner: Any, keys: Sequence[str], args: Sequence[Any], kwargs: Dict[str, Any]
) -> Tuple[Any, ...]:
    """The key Autotuner.run() derives from a positional call: the key columns
    in `key` order, then str(dtype) of every tensor argument in kernel parameter
    order. The C++ caller builds TuneKeyView in this order too; a Python caller
    that passes tensors as keywords in another order gets a different tail."""
    names = list(getattr(tuner, "arg_names", []) or [])
    named = {**dict(zip(names, args)), **kwargs}
    named = {name: named[name] for name in names if name in named}
    key: List[Any] = [named[name] for name in keys if name in named]
    key += [str(value.dtype) for value in named.values() if hasattr(value, "dtype")]
    return tuple(key)


def _non_integer_column(keys: Sequence[str], key: Sequence[Any]) -> Optional[str]:
    """Reason a key cannot be a TuneKeyView (int64 columns), else None."""
    for name, value in zip(keys, key):
        if isinstance(value, bool) or (isinstance(value, float) and value.is_integer()):
            continue
        if not isinstance(value, int):
            return (
                f"key column '{name}' holds {value!r}; TuneKeyView carries integer "
                "key columns only"
            )
    return None


def _select_native_config(
    tuner: Any,
    keys: Sequence[str],
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    launch_kwargs: Optional[Dict[str, Any]] = None,
    clone_tensors: bool = True,
) -> Tuple[Any, Tuple[Any, ...]]:
    """The selection half of Autotuner.run() for a plain @triton.autotune
    tuner: returns (config, canonical key).

    run() itself derives the key, prunes, benchmarks through _bench, consults
    its disk cache and fills the (shared) in-memory cache; only the final
    launch is intercepted. A benchmark without a grid raises _NeedsGrid.
    """
    local = _local_tuner(tuner)
    configs = list(local.configs)
    canonical = _canonical_native_key(tuner, keys, args, kwargs)
    if len(configs) <= 1:
        return (configs[0] if configs else None), canonical
    grid = (launch_kwargs or {}).get("grid")
    run_kwargs = dict(kwargs)
    run_kwargs["warmup"] = False
    if grid is not None:
        run_kwargs["grid"] = grid
    recorder = _KeyRecorder(tuner.cache)
    local.cache = recorder
    state = {"benchmarking": False, "final": 0, "benchmarks": 0}
    native_bench = local._bench
    native_pre_hook = local.pre_hook
    cloner = None

    def isolate(value):
        nonlocal cloner
        if not clone_tensors:
            return value
        if cloner is None:
            cloner = _argument_cloner()
        return cloner(value)

    def pre_hook(named, reset_only=False):
        # run() resets once after tuning, outside _bench. Redirect that reset
        # to the same storage copies used by the trials.
        if not state["benchmarking"]:
            named = (
                {name: isolate(value) for name, value in named.items()}
                if isinstance(named, dict)
                else tuple(isolate(value) for value in named)
            )
        return native_pre_hook(named, reset_only=reset_only)

    def bench(*bench_positional, config, **bench_kwargs):
        if grid is None:
            raise _NeedsGrid()
        bench_positional = tuple(isolate(value) for value in bench_positional)
        bench_kwargs = {name: isolate(value) for name, value in bench_kwargs.items()}
        nargs = local.nargs
        local.nargs = dict(zip(local.arg_names, bench_positional))
        state["benchmarking"] = True
        state["benchmarks"] += 1
        try:
            return native_bench(*bench_positional, config=config, **bench_kwargs)
        finally:
            state["benchmarking"] = False
            local.nargs = nargs

    local._bench = bench
    local.pre_hook = pre_hook
    local.fn = _LaunchInterceptor(tuner.fn, state)
    try:
        local.run(*args, **run_kwargs)
    finally:
        local.nargs = None
        cloner = None
    if state["final"] != 1:
        # run() launched outside _bench: a control flow this adapter does not
        # know, whose timings (if any) went nowhere. Leave no trace of it.
        if recorder.inserted and recorder.last_key is not None:
            tuner.cache.pop(recorder.last_key, None)
        raise _UnsupportedBenchmark(
            f"Autotuner.run() made {state['final']} launches outside _bench; "
            "this Triton version is not supported by the native adapter"
        )
    config = getattr(local, "best_config", None)
    if config is None:
        raise _UnsupportedBenchmark("Autotuner.run() left no best_config behind")
    stored = recorder.last_key
    if stored is not None and tuple(stored) != canonical:
        _warn_once(
            f"{getattr(getattr(tuner, 'base_fn', None), '__name__', '?')}: Triton "
            f"cached the configuration under {tuple(stored)!r}; the C++ table uses "
            f"the canonical key {canonical!r}"
        )
    return config, canonical


# --------------------------------------------------------------------------
# Answers


def _describe_config(
    tuner: Any,
    identity: Dict[str, Any],
    keys: Sequence[str],
    strategies: Sequence[str],
    config: Any,
    config_key: Sequence[Any],
    integer_key: bool = False,
) -> Dict[str, Any]:
    """The answer dict TunedTable expects for one selected config, or
    {"kernel_id", "unsupported"}. With `integer_key` the key columns must be
    integers (TuneKeyView) and the dtype tail is spelled as strings."""
    kernel_id = identity["kernel_id"]
    if getattr(config, "pre_hook", None) is not None:
        return {
            "kernel_id": kernel_id,
            "unsupported": "the selected config carries a pre_hook",
        }
    arg_order = {name: index for index, name in enumerate(identity["arg_names"])}
    types_ = _export.kwarg_types(tuner)
    kw_pairs: List[Tuple[str, Any]] = []
    for name, value in config.kwargs.items():
        if name not in arg_order:
            return {
                "kernel_id": kernel_id,
                "unsupported": f"constexpr '{name}' is not a kernel parameter",
            }
        if not isinstance(value, (bool, int, float, str)):
            return {
                "kernel_id": kernel_id,
                "unsupported": f"constexpr '{name}' has a non-serialisable value {value!r}",
            }
        kw_pairs.append(
            (name, _export.coerce_value(value, types_.get(name, type(value))))
        )
    kw_pairs.sort(key=lambda item: arg_order[item[0]])
    extra: Dict[str, str] = {}
    for field in sorted(
        _export.triton_config_fields() - {"num_warps", "num_stages", "pre_hook"}
    ):
        value = getattr(config, field, None)
        if value is None or (field == "num_ctas" and value == 1):
            continue
        extra[field] = str(value)
    normalised: List[Any] = []
    for index, element in enumerate(config_key):
        if isinstance(element, float) and element.is_integer():
            element = int(element)
        if integer_key:
            if index < len(keys):
                if isinstance(element, bool):
                    element = int(element)
                if not isinstance(element, int):
                    return {
                        "kernel_id": kernel_id,
                        "unsupported": f"key column '{keys[index]}' holds {element!r}; "
                        "TuneKeyView carries integer key columns only",
                    }
            else:
                element = str(element)
        normalised.append(element)
    return {
        "kernel_id": kernel_id,
        "op_name": getattr(tuner, "__name__", kernel_id),
        "source_path": identity["source_path"],
        "key_columns": [
            {"name": key, "strategy": strategy}
            for key, strategy in zip(keys, strategies)
        ],
        "dtype_keys": max(0, len(config_key) - len(keys)),
        "key": normalised,
        "num_warps": int(getattr(config, "num_warps", 4)),
        "num_stages": int(getattr(config, "num_stages", 3)),
        "extra": extra,
        "kwargs": [[name, value] for name, value in kw_pairs],
    }


def resolve_with_tuner(
    tuner: Any,
    libentry_module: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    grid: Any = None,
    clone_tensors: bool = True,
) -> Dict[str, Any]:
    """Resolve and describe the configuration for one launch of a LibTuner.

    Returns a dict with kernel_id, key_columns ([{name, strategy}]), dtype_keys,
    key (normalised, as the tuner stores it), num_warps, num_stages, extra
    ({name: str}) and kwargs ([[name, value], ...] in kernel parameter order),
    or {"kernel_id": ..., "unsupported": reason}. `grid` is needed only when
    the key has not been tuned yet (see build_grid); without it such a key is
    reported as unsupported rather than benchmarked.
    """
    identity = _export.kernel_identity(tuner)
    kernel_id = identity["kernel_id"]
    reason = _export.refusal_reason(tuner, identity["wrappers"])
    if reason is not None:
        return {"kernel_id": kernel_id, "unsupported": reason}
    libcache = getattr(libentry_module, "libcache")
    blas_dialect = _export.is_flagblas(libentry_module)
    names = _export.strategy_names(tuner, libentry_module, blas_dialect)
    keys = list(getattr(tuner, "keys", []) or [])
    launch_kwargs = None
    grid_fn = build_grid(grid, tuner, args, kwargs)
    if grid_fn is not None:
        launch_kwargs = {"grid": grid_fn, "warmup": False}
    try:
        # Participating libraries share this lock with native _bench. It also
        # protects scratch allocation from another tuner's graph capture.
        with getattr(libentry_module, "benchmark_lock", nullcontext()):
            config, config_key = _select_config(
                tuner, libcache, args, kwargs, launch_kwargs, clone_tensors
            )
    except _UnsupportedBenchmark as error:
        return {"kernel_id": kernel_id, "unsupported": str(error)}
    except _NeedsGrid:
        return {
            "kernel_id": kernel_id,
            "unsupported": "this key has not been tuned yet and the caller supplied no launch grid; "
            "tune it from Python first or pass ResolveArgs.grid",
        }
    if config is None:
        return {
            "kernel_id": kernel_id,
            "unsupported": "the tuner has no candidate configs",
        }
    return _describe_config(tuner, identity, keys, names, config, config_key)


def resolve_with_native_tuner(
    tuner: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    grid: Any = None,
    clone_tensors: bool = True,
    benchmark_lock: Any = None,
) -> Dict[str, Any]:
    """Resolve and describe the configuration for one launch of a plain
    @triton.autotune tuner. Same answer shape as resolve_with_tuner; every key
    column is reported with the "default" strategy (exact value) because the
    native tuner keys on exact values.

    Refused, with a reason: custom tuner hooks, config pre_hooks, heuristics,
    non-serialisable constexprs, non-integer key columns (TuneKeyView holds
    int64), and a Triton whose Autotuner.run() no longer benchmarks through
    _bench. `benchmark_lock` serialises benchmarks with a library's own lock;
    without one a module-level lock is used.
    """
    identity = _export.kernel_identity(tuner)
    kernel_id = identity["kernel_id"]
    reason = _native_support_reason(tuner) or _export.refusal_reason(
        tuner, identity["wrappers"]
    )
    if reason is not None:
        return {"kernel_id": kernel_id, "unsupported": reason}
    keys = _native_keys(tuner)
    reason = _non_integer_column(keys, _canonical_native_key(tuner, keys, args, kwargs))
    if reason is not None:  # known before any benchmark: do not pay for one
        return {"kernel_id": kernel_id, "unsupported": reason}
    launch_kwargs = None
    grid_fn = build_grid(grid, tuner, args, kwargs)
    if grid_fn is not None:
        launch_kwargs = {"grid": grid_fn, "warmup": False}
    try:
        with benchmark_lock if benchmark_lock is not None else _native_benchmark_lock:
            config, config_key = _select_native_config(
                tuner, keys, args, kwargs, launch_kwargs, clone_tensors
            )
    except _UnsupportedBenchmark as error:
        return {"kernel_id": kernel_id, "unsupported": str(error)}
    except _NeedsGrid:
        return {
            "kernel_id": kernel_id,
            "unsupported": "this key has not been tuned yet and the caller supplied no launch grid; "
            "tune it from Python first or pass ResolveArgs.grid",
        }
    if config is None:
        return {
            "kernel_id": kernel_id,
            "unsupported": "the tuner has no candidate configs",
        }
    answer = _describe_config(
        tuner,
        identity,
        keys,
        ["default"] * len(keys),
        config,
        config_key,
        integer_key=True,
    )
    if "unsupported" not in answer:
        answer["op_name"] = kernel_id
        try:
            _native_records.setdefault(tuner, {})[tuple(answer["key"])] = config
        except TypeError:  # not weak-referenceable: nothing to export later
            pass
    return answer


def _device_guard(args: Sequence[Any], kwargs: Dict[str, Any], device_index: Any):
    """Make `device_index` current for the backend the tensor arguments live
    on, so trial launches and scratch copies land where the caller launches."""
    if device_index is None:
        return nullcontext()
    try:
        import torch
    except ImportError:
        return nullcontext()
    for value in [*args, *kwargs.values()]:
        if torch.is_tensor(value) and value.device.type != "cpu":
            module = getattr(torch, value.device.type, None)
            guard = getattr(module, "device", None)
            return guard(int(device_index)) if guard is not None else nullcontext()
    return nullcontext()


def resolve(
    package_name: str,
    kernel_id: str,
    args: Sequence[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    source_path: Optional[str] = None,
    grid: Any = None,
    clone_tensors: bool = True,
    device_index: Optional[int] = None,
    stream: Optional[int] = None,
) -> Dict[str, Any]:
    """Entry point for the C++ bridge: find the tuner and resolve.

    `source_path` is the kernel file the caller launches from; `grid` is a
    Python expression, tuple or callable used only if the key must be tuned
    now (see build_grid). `device_index` is the device the C++ side resolves
    for and is made current around the selection; `stream` (a raw handle) is
    accepted for bridges that cannot guard the stream themselves and is
    currently informational. TritonDtype markers among the arguments are
    materialised into triton.language dtypes.
    """
    tuner = find_tuner(package_name, kernel_id, source_path=source_path)
    positional = [_materialize(value) for value in args]
    named = {name: _materialize(value) for name, value in (kwargs or {}).items()}
    libentry = _libentry_module(package_name)
    libtuner_cls = getattr(libentry, "LibTuner", None) if libentry else None
    with _device_guard(positional, named, device_index):
        if libtuner_cls is not None and isinstance(tuner, libtuner_cls):
            return resolve_with_tuner(
                tuner,
                libentry,
                positional,
                named,
                grid=grid,
                clone_tensors=clone_tensors,
            )
        return resolve_with_native_tuner(
            tuner,
            positional,
            named,
            grid=grid,
            clone_tensors=clone_tensors,
            benchmark_lock=getattr(libentry, "benchmark_lock", None),
        )


# --------------------------------------------------------------------------
# Export of @triton.autotune results (in-process; the native cache has no
# database to read back from another process)


def _native_entry(
    tuner: Any,
    identity: Dict[str, Any],
    keys: Sequence[str],
    key: Sequence[Any],
    config: Any,
) -> Optional[Dict[str, Any]]:
    answer = _describe_config(
        tuner, identity, keys, ["default"] * len(keys), config, key, integer_key=True
    )
    if "unsupported" in answer:
        return None
    entry = {"key": answer["key"]}
    if answer["extra"]:
        entry["extra"] = answer["extra"]
    entry["kwargs"] = answer["kwargs"]
    entry["num_warps"] = answer["num_warps"]
    entry["num_stages"] = answer["num_stages"]
    return entry


def native_kernel_table(
    tuner: Any,
    include_python_tuned: bool = True,
    log=lambda message: None,
) -> Dict[str, Any]:
    """One ``kernels[]`` element of the JSON table for a @triton.autotune tuner.

    Entries come from what this process resolved for the C++ side (canonical
    keys). With `include_python_tuned`, entries the Python side tuned through
    ordinary calls are added when their dtype tail is unambiguous: Triton
    orders that tail by call form, so an entry whose tensors have different
    dtypes and that was not resolved here is skipped and reported.
    """
    identity = _export.kernel_identity(tuner)
    keys = _native_keys(tuner)
    kernel: Dict[str, Any] = {
        "kernel_id": identity["kernel_id"],
        "op_name": identity["kernel_id"],
        "config_table_name": "",
        "source_path": identity["source_path"],
        "cache_namespace": identity["source_path"],
        "source_sha256": identity["source_sha256"],
        "candidate_set_hash": "",
        "key_columns": [{"name": key, "strategy": "default"} for key in keys],
        "dtype_keys": 0,
        "entries": [],
    }
    reason = _native_support_reason(tuner) or _export.refusal_reason(
        tuner, identity["wrappers"]
    )
    if reason is not None:
        kernel["unsupported"] = reason
        log(f"  {kernel['kernel_id']}: unsupported ({reason})")
        return kernel
    rows: Dict[Tuple[Any, ...], Any] = dict(_native_records.get(tuner, {}))
    if include_python_tuned:
        for key, config in list(tuner.cache.items()):
            key = tuple(key)
            if key in rows:
                continue
            if len(set(key[len(keys) :])) <= 1:
                rows[key] = config
            else:
                log(
                    f"  {kernel['kernel_id']}: skipped Python-tuned key {key!r} "
                    "(dtype order depends on the call form; resolve it through the bridge to export it)"
                )
    for key, config in rows.items():
        entry = _native_entry(tuner, identity, keys, key, config)
        if entry is None:
            log(f"  {kernel['kernel_id']}: skipped key {key!r} (not expressible)")
            continue
        width = len(entry["key"]) - len(keys)
        if kernel["entries"] and width != kernel["dtype_keys"]:
            log(f"  {kernel['kernel_id']}: skipped key {key!r} (different dtype width)")
            continue
        kernel["dtype_keys"] = width
        kernel["entries"].append(entry)
    return kernel


def export_native_table(
    path: Union[str, "os.PathLike[str]"],
    tuners: Union[str, Iterable[Any]],
    backend: str,
    device_index: int = 0,
    vendor: str = "",
    device_name: Optional[str] = None,
    include_python_tuned: bool = True,
    log=lambda message: print(message, file=sys.stderr),
) -> Dict[str, Any]:
    """Write the JSON table (format v2, readable by TunedTable::load) for
    @triton.autotune tuners resolved or tuned in this process.

    `tuners` is an iterable of Autotuner objects or a package name whose live
    plain Autotuners are exported. `backend` is the libtriton_jit backend the
    table is for ("CUDA", "IX", ...); the device name and Triton version are
    detected unless given.

    Copies of the same source/kernel are merged. Conflicting schemas or
    configurations raise ValueError before the output file is written.
    """
    if isinstance(tuners, str):
        package_name = tuners
        _export.import_package_tree(package_name)
        libentry = _libentry_module(package_name)
        tuners = native_tuners(
            getattr(libentry, "LibTuner", None) if libentry else None, package_name
        )
        tuners.sort(key=lambda tuner: not _module_is_registered(tuner, package_name))
    kernels = {}
    entries = {}
    for tuner in tuners:
        if getattr(tuner, "_tuned_resolver_local", False):
            continue
        identity = _native_identity(tuner)
        kernel = native_kernel_table(tuner, include_python_tuned, log)
        if identity not in kernels:
            kernels[identity] = kernel
            entries[identity] = {tuple(row["key"]): row for row in kernel["entries"]}
            continue
        merged = kernels[identity]
        if (
            merged["key_columns"] != kernel["key_columns"]
            or merged.get("unsupported") != kernel.get("unsupported")
            or (
                merged["entries"]
                and kernel["entries"]
                and merged["dtype_keys"] != kernel["dtype_keys"]
            )
        ):
            raise ValueError(
                f"conflicting tuner schemas for {identity}; export one tuner explicitly"
            )
        for row in kernel["entries"]:
            key = tuple(row["key"])
            previous = entries[identity].get(key)
            if previous is not None and previous != row:
                raise ValueError(
                    f"conflicting tuned configs for {identity}, key {key!r}; export one tuner explicitly"
                )
            if previous is None:
                entries[identity][key] = row
                merged["entries"].append(row)
                merged["dtype_keys"] = kernel["dtype_keys"]
    fingerprint = _export.detect_fingerprint(backend, vendor, device_index, device_name)
    table = {
        "format_version": _export.FORMAT_VERSION,
        "generator": "scripts/tuned_resolver.py export_native_table",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_db": "",
        "fingerprint": fingerprint,
        "kernels": list(kernels.values()),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=2)
        handle.write("\n")
    return table


__all__ = [
    "TritonDtype",
    "build_grid",
    "export_native_table",
    "find_tuner",
    "native_kernel_table",
    "native_tuners",
    "resolve",
    "resolve_with_native_tuner",
    "resolve_with_tuner",
]
