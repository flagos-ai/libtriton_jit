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
"""Export LibTuner configurations as source-scoped JSON tables for C++ lookup.

Use --list to inspect discovered tuners, --source to disambiguate kernel files,
and --cache-namespace to bind a portable source revision. See docs/autotune.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import gc
import hashlib
import importlib
import inspect
import json
import os
import pkgutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

FORMAT_VERSION = 2

# FlagGems / FlagBLAS vendor name -> libtriton_jit BACKEND name.
VENDOR_TO_BACKEND = {
    "nvidia": "CUDA",
    "iluvatar": "IX",
    "cambricon": "MLU",
    "ascend": "NPU",
    "mthreads": "MUSA",
    "metax": "MACA",
    "hygon": "HCU",
    "enflame": "GCU",
}

# Names that triton.Config.all_kwargs() adds next to the user kwargs. Anything
# in this set that is not num_warps / num_stages goes to CompileOptions.extra.
_CONFIG_FIELD_CACHE: Optional[set] = None


def triton_config_fields() -> set:
    """Field names of triton.Config other than the kwargs dict."""
    global _CONFIG_FIELD_CACHE
    if _CONFIG_FIELD_CACHE is None:
        fields = {"num_warps", "num_stages", "num_ctas", "maxnreg", "pre_hook"}
        try:
            import triton

            params = inspect.signature(triton.Config.__init__).parameters
            fields |= {name for name in params if name not in ("self", "kwargs")}
        except Exception:  # noqa: BLE001 - triton is optional for --help / tests
            pass
        _CONFIG_FIELD_CACHE = fields
    return _CONFIG_FIELD_CACHE


# --------------------------------------------------------------------------
# Discovery


def import_package_tree(package_name: str, walk_ops: bool = True) -> None:
    """Import a package and, by default, every module under ``<package>.ops`` so
    that decorators run and LibTuner instances exist."""
    package = importlib.import_module(package_name)
    if not walk_ops:
        return
    for sub in ("ops",):
        try:
            ops = importlib.import_module(f"{package_name}.{sub}")
        except ImportError:
            continue
        path = getattr(ops, "__path__", None)
        if not path:
            continue
        for module_info in pkgutil.walk_packages(path, prefix=f"{package_name}.{sub}."):
            try:
                importlib.import_module(module_info.name)
            except (
                Exception
            ) as error:  # noqa: BLE001 - vendor-specific modules may not import
                print(
                    f"[export_tuned_table] skipping {module_info.name}: {error}",
                    file=sys.stderr,
                )
    del package


def tuner_module_is_registered(tuner: Any, package_name: str) -> bool:
    """True when the tuner's kernel module was imported as part of the package
    (``flag_gems.ops.mm`` in sys.modules with that file), not a stray copy of
    the same file loaded again by the C++ runtime for launching."""
    identity = kernel_identity(tuner)
    jit_fn, _ = unwrap_to_jit_function(tuner.fn)
    py_fn = getattr(jit_fn, "fn", jit_fn)
    module_name = getattr(getattr(tuner, "base_fn", py_fn), "__module__", "") or ""
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None) if module is not None else None
    if not module_file:
        return False
    try:
        return os.path.realpath(module_file) == os.path.realpath(
            identity["source_path"]
        )
    except OSError:
        return False


def dedupe_tuners(tuners: List[Any], package_name: str) -> List[Any]:
    """Collapse tuners that wrap the same (source file, kernel) - the same file
    imported more than once - keeping the copy imported under the package when
    there is one. Kernels of the same name in different files are kept apart
    (use --source to pick one)."""
    groups: Dict[Tuple[str, str], List[Any]] = {}
    order: List[Tuple[str, str]] = []
    for tuner in tuners:
        identity = kernel_identity(tuner)
        key = (
            os.path.realpath(identity["source_path"])
            if identity["source_path"]
            else "",
            identity["kernel_id"],
        )
        if key not in groups:
            order.append(key)
        groups.setdefault(key, []).append(tuner)
    result = []
    for key in order:
        candidates = groups[key]
        preferred = [
            t for t in candidates if tuner_module_is_registered(t, package_name)
        ]
        result.append((preferred or candidates)[0])
    return result


def find_tuners(libentry_module) -> List[Any]:
    """Every live LibTuner instance, in a stable order."""
    tuner_cls = getattr(libentry_module, "LibTuner")
    tuners = [obj for obj in gc.get_objects() if isinstance(obj, tuner_cls)]
    tuners.sort(key=lambda t: (getattr(t, "__name__", ""), id(t)))
    return tuners


def libentry_module_for(package_name: str):
    return importlib.import_module(f"{package_name}.utils.libentry")


def is_flagblas(libentry_module) -> bool:
    return libentry_module.__name__.startswith("flag_blas")


# --------------------------------------------------------------------------
# Per-tuner metadata


def unwrap_to_jit_function(fn: Any) -> Tuple[Any, List[Any]]:
    """Follow the ``.fn`` chain down to the JITFunction; return it and every
    wrapper passed on the way (Heuristics, nested Autotuner, ...)."""
    wrappers: List[Any] = []
    current = fn
    for _ in range(32):
        if (
            type(current).__name__ == "JITFunction"
            or hasattr(current, "cache_key")
            and hasattr(current, "arg_names")
            and not hasattr(current, "fn")
        ):
            return current, wrappers
        if not hasattr(current, "fn"):
            break
        wrappers.append(current)
        current = current.fn
    return current, wrappers


def kernel_identity(tuner: Any) -> Dict[str, Any]:
    jit_fn, wrappers = unwrap_to_jit_function(tuner.fn)
    py_fn = getattr(jit_fn, "fn", jit_fn)
    kernel_id = getattr(py_fn, "__name__", getattr(tuner, "__name__", "?"))
    source_path = getattr(getattr(py_fn, "__code__", None), "co_filename", "")
    source_sha256 = ""
    if source_path and os.path.isfile(source_path):
        with open(source_path, "rb") as handle:
            source_sha256 = hashlib.sha256(handle.read()).hexdigest()
    arg_names = list(getattr(jit_fn, "arg_names", []))
    return {
        "kernel_id": kernel_id,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "arg_names": arg_names,
        "wrappers": wrappers,
    }


def refusal_reason(tuner: Any, wrappers: Sequence[Any]) -> Optional[str]:
    for wrapper in wrappers:
        name = type(wrapper).__name__
        if name == "Heuristics":
            return "kernel is wrapped in @triton.heuristics; heuristic constexprs are recomputed per call and never reach the database"
        if name == "Autotuner" and wrapper is not tuner:
            return "a nested triton.runtime.Autotuner sits between the LibTuner and the kernel"
    configs = list(getattr(tuner, "configs", []) or [])
    if any(getattr(config, "pre_hook", None) is not None for config in configs):
        return "a candidate config carries a pre_hook, which runs Python before the launch and is not stored"
    if getattr(tuner, "_flagtune_pre_hook", None) is not None:
        return "the tuner carries a FlagTune pre_hook"
    for config in configs:
        for name, value in getattr(config, "kwargs", {}).items():
            if not isinstance(value, (bool, int, float, str)):
                return f"constexpr '{name}' has a candidate value of type {type(value).__name__}, which cannot be serialised"
    return None


def strategy_names(tuner: Any, libentry_module, blas_dialect: bool) -> List[str]:
    """Map the tuner's strategy callables back to their registered names."""
    keys = list(getattr(tuner, "keys", []) or [])
    strategies = getattr(tuner, "strategy", None)
    if strategies is None:
        return ["default"] * len(keys)
    table = None
    tuner_cls = getattr(libentry_module, "LibTuner")
    for attr in ("_strategy_table", "_strategies", "strategy_table"):
        table = getattr(tuner_cls, attr, None)
        if isinstance(table, dict):
            break
    reverse: Dict[int, str] = {}
    if isinstance(table, dict):
        for name, fn in table.items():
            if name is None:
                continue
            reverse.setdefault(id(fn), str(name))
            if str(name) == "default":
                reverse[id(fn)] = "default"
    names: List[str] = []
    for strategy in strategies:
        name = reverse.get(id(strategy))
        if name is None:
            fn_name = getattr(strategy, "__name__", "")
            name = {
                "default_strategy": "default",
                "log2_strategy": "log",
                "align32_strategy": "align32",
            }.get(fn_name)
        if name is None:
            raise ValueError(
                f"tuner {tuner.__name__}: strategy {strategy!r} is not a registered LibTuner strategy"
            )
        if name == "align32" and blas_dialect:
            name = "align32_ceil"
        names.append(name)
    if len(names) != len(keys):
        raise ValueError(
            f"tuner {tuner.__name__}: {len(names)} strategies for {len(keys)} keys"
        )
    return names


def kwarg_types(tuner: Any) -> Dict[str, type]:
    """Python type of every constexpr kwarg across the candidate configs. Used
    to recover bools, which the database stores as integers."""
    types: Dict[str, type] = {}
    for config in getattr(tuner, "configs", []) or []:
        for name, value in getattr(config, "kwargs", {}).items():
            if isinstance(value, bool):
                types[name] = bool
            elif name not in types:
                types[name] = type(value)
    return types


# --------------------------------------------------------------------------
# Database access


def physical_table_names(engine, config_table_name: str) -> List[str]:
    import sqlalchemy

    prefix = config_table_name + "-"
    inspector = sqlalchemy.inspect(engine)
    return sorted(
        name for name in inspector.get_table_names() if name.startswith(prefix)
    )


def read_rows(engine, table_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    import sqlalchemy

    inspector = sqlalchemy.inspect(engine)
    columns = [column["name"] for column in inspector.get_columns(table_name)]
    quoted = table_name.replace('"', '""')
    with engine.connect() as connection:
        result = connection.execute(sqlalchemy.text(f'SELECT * FROM "{quoted}"'))
        rows = [dict(zip(result.keys(), row)) for row in result]
    return columns, rows


def coerce_value(value: Any, py_type: Optional[type]) -> Any:
    if py_type is bool:
        return bool(value)
    if py_type is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if py_type is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if py_type is str:
        return str(value)
    return value


def row_to_entry(
    row: Dict[str, Any],
    n_key_columns: int,
    n_keys: int,
    arg_order: Dict[str, int],
    types: Dict[str, type],
    tuner_name: str,
) -> Dict[str, Any]:
    key: List[Any] = []
    for index in range(n_key_columns):
        value = row[f"key_{index}"]
        if index < n_keys:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"tuner {tuner_name}: key column {index} holds {value!r}, expected an integer"
                )
        else:
            value = str(value)
        key.append(value)
    entry: Dict[str, Any] = {"key": key}
    extra: Dict[str, str] = {}
    kwargs: List[Tuple[str, Any]] = []
    config_fields = triton_config_fields()
    for name, value in row.items():
        if name.startswith("key_") or value is None:
            continue
        if name == "num_warps":
            entry["num_warps"] = int(value)
        elif name == "num_stages":
            entry["num_stages"] = int(value)
        elif name in config_fields:
            if name == "num_ctas" and int(value) == 1:
                continue
            extra[name] = str(value)
        else:
            if name not in arg_order:
                raise ValueError(
                    f"tuner {tuner_name}: column '{name}' is not a parameter of the kernel"
                )
            kwargs.append((name, coerce_value(value, types.get(name))))
    kwargs.sort(key=lambda item: arg_order[item[0]])
    if extra:
        entry["extra"] = extra
    entry["kwargs"] = [[name, value] for name, value in kwargs]
    return entry


# --------------------------------------------------------------------------
# Export


def export_tuner(
    tuner: Any, engine, libentry_module, blas_dialect: bool, log
) -> Dict[str, Any]:
    identity = kernel_identity(tuner)
    keys = list(getattr(tuner, "keys", []) or [])
    kernel: Dict[str, Any] = {
        "kernel_id": identity["kernel_id"],
        "op_name": getattr(tuner, "__name__", identity["kernel_id"]),
        "config_table_name": getattr(tuner, "config_table_name", ""),
        "source_path": identity["source_path"],
        "cache_namespace": identity["source_path"],
        "source_sha256": identity["source_sha256"],
        "candidate_set_hash": getattr(tuner, "configs_hash", ""),
        "key_columns": [],
        "dtype_keys": 0,
        "entries": [],
    }
    reason = refusal_reason(tuner, identity["wrappers"])
    if reason is not None:
        kernel["unsupported"] = reason
        log(f"  {kernel['kernel_id']}: unsupported ({reason})")
        return kernel

    names = strategy_names(tuner, libentry_module, blas_dialect)
    kernel["key_columns"] = [
        {"name": key, "strategy": strategy} for key, strategy in zip(keys, names)
    ]
    tables = physical_table_names(engine, kernel["config_table_name"])
    if not tables:
        log(
            f"  {kernel['kernel_id']}: no rows in the database (table {kernel['config_table_name']}-*)"
        )
        return kernel

    arg_order = {name: index for index, name in enumerate(identity["arg_names"])}
    types = kwarg_types(tuner)
    widths: Dict[int, List[Dict[str, Any]]] = {}
    for table in tables:
        columns, rows = read_rows(engine, table)
        n_key_columns = sum(1 for column in columns if column.startswith("key_"))
        if n_key_columns < len(keys):
            raise ValueError(
                f"tuner {tuner.__name__}: table {table} has {n_key_columns} key columns but the tuner has {len(keys)} keys"
            )
        for row in rows:
            entry = row_to_entry(
                row, n_key_columns, len(keys), arg_order, types, tuner.__name__
            )
            widths.setdefault(n_key_columns, []).append(entry)
    if not widths:
        log(f"  {kernel['kernel_id']}: tables exist but hold no rows")
        return kernel
    chosen = max(widths, key=lambda width: (len(widths[width]), width))
    if len(widths) > 1:
        others = ", ".join(
            f"{width - len(keys)} dtype keys ({len(widths[width])} rows)"
            for width in widths
            if width != chosen
        )
        log(
            f"  {kernel['kernel_id']}: rows exist with different dtype-key counts; exporting {chosen - len(keys)} dtype keys, dropping {others}"
        )
    kernel["dtype_keys"] = chosen - len(keys)
    kernel["entries"] = widths[chosen]
    log(
        f"  {kernel['kernel_id']}: {len(kernel['entries'])} entries, keys {keys} strategies {names}, {kernel['dtype_keys']} dtype keys"
    )
    return kernel


def detect_fingerprint(
    backend: str, vendor: str, device_index: int, device_name: Optional[str]
) -> Dict[str, str]:
    fp = {
        "backend": backend,
        "vendor": vendor,
        "device_name": device_name or "",
        "arch": "",
        "triton_version": "",
        "libtriton_jit_version": "",
    }
    try:
        import triton

        fp["triton_version"] = getattr(triton, "__version__", "")
    except Exception:  # noqa: BLE001
        pass
    if device_name:
        return fp
    try:
        import torch

        if (
            backend in ("CUDA", "IX", "MUSA", "MACA", "HCU")
            and torch.cuda.is_available()
        ):
            fp["device_name"] = torch.cuda.get_device_name(device_index)
            if backend in ("CUDA", "IX"):
                major, minor = torch.cuda.get_device_capability(device_index)
                fp["arch"] = f"sm_{major}{minor}"
        elif backend == "NPU" and hasattr(torch, "npu"):
            fp["device_name"] = torch.npu.get_device_name(device_index)
        elif backend == "MLU" and hasattr(torch, "mlu"):
            fp["device_name"] = torch.mlu.get_device_name(device_index)
    except Exception as error:  # noqa: BLE001
        print(
            f"[export_tuned_table] could not detect the device name: {error}",
            file=sys.stderr,
        )
    return fp


def build_table(
    tuners: Iterable[Any],
    engine,
    libentry_module,
    fingerprint: Dict[str, str],
    source_db: str,
    log=lambda message: None,
    include_unsupported: bool = True,
) -> Dict[str, Any]:
    blas_dialect = is_flagblas(libentry_module)
    kernels: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for tuner in tuners:
        kernel = export_tuner(tuner, engine, libentry_module, blas_dialect, log)
        if "unsupported" in kernel and not include_unsupported:
            continue
        kernel_id = kernel["kernel_id"]
        if kernel_id in seen:
            raise ValueError(
                f"kernel id '{kernel_id}' is produced by two tuners from different files "
                f"({seen[kernel_id]} and {kernel.get('source_path', '?')}); use --source to select one"
            )
        seen[kernel_id] = kernel.get("source_path", "?")
        kernels.append(kernel)
    return {
        "format_version": FORMAT_VERSION,
        "generator": "scripts/export_tuned_table.py",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_db": source_db,
        "fingerprint": fingerprint,
        "kernels": kernels,
    }


def resolve_engine(libentry_module, db_url: Optional[str]):
    import sqlalchemy

    if db_url:
        return sqlalchemy.create_engine(db_url), db_url
    libcache = getattr(libentry_module, "libcache")
    model = getattr(libcache, "model")
    return model.engine, str(model.engine.url)


def vendor_name(package_name: str) -> str:
    """The vendor the package resolved at import time ("nvidia", "iluvatar", ...).

    FlagBLAS's libentry imports ``vendor_module`` at module level; FlagGems keeps
    it in ``libentry._state``; older layouts expose it from ``<package>.runtime``.
    """
    libentry = libentry_module_for(package_name)
    candidates = [
        lambda: libentry.vendor_module.vendor_info.vendor_name,
        lambda: libentry._state.vendor_module.vendor_info.vendor_name,
        lambda: importlib.import_module(
            f"{package_name}.runtime.backend"
        ).vendor_module.vendor_info.vendor_name,
        lambda: importlib.import_module(
            f"{package_name}.runtime"
        ).vendor_module.vendor_info.vendor_name,
    ]
    for candidate in candidates:
        try:
            value = candidate()
        except Exception:  # noqa: BLE001 - try the next layout
            continue
        if value:
            return str(value)
    return ""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--package",
        default="flag_gems",
        help="package whose LibTuner instances to export (flag_gems, flag_blas)",
    )
    parser.add_argument(
        "--no-walk-ops",
        action="store_true",
        help="do not import every module under <package>.ops",
    )
    parser.add_argument(
        "--import",
        dest="extra_imports",
        action="append",
        default=[],
        help="additional module to import before discovery",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLAlchemy URL of the tuned-config database (default: the package's own)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="libtriton_jit backend name (default: derived from the vendor)",
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--device-name", default=None, help="override the detected device name"
    )
    parser.add_argument(
        "--kernel",
        action="append",
        default=[],
        help="glob on the Triton kernel name; repeatable",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="glob on the kernel's source file path (disambiguates same-named kernels); repeatable",
    )
    parser.add_argument(
        "--op",
        action="append",
        default=[],
        help="glob on the LibTuner (op) name; repeatable",
    )
    parser.add_argument(
        "--exclude-unsupported",
        action="store_true",
        help="drop refused kernels instead of listing them",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="only print the tuners that would be exported",
    )
    parser.add_argument(
        "--output", "-o", default=None, help="output JSON path (default: stdout)"
    )
    parser.add_argument(
        "--cache-namespace",
        help="portable versioned source identity; requires exactly one exported kernel",
    )
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message, file=sys.stderr)

    import_package_tree(args.package, walk_ops=not args.no_walk_ops)
    for module in args.extra_imports:
        importlib.import_module(module)
    libentry = libentry_module_for(args.package)
    tuners = find_tuners(libentry)

    def selected(tuner: Any) -> bool:
        identity = kernel_identity(tuner)
        kernel_id = identity["kernel_id"]
        if args.kernel and not any(
            fnmatch.fnmatch(kernel_id, pattern) for pattern in args.kernel
        ):
            return False
        if args.source and not any(
            fnmatch.fnmatch(identity["source_path"], pattern) for pattern in args.source
        ):
            return False
        if args.op and not any(
            fnmatch.fnmatch(getattr(tuner, "__name__", ""), pattern)
            for pattern in args.op
        ):
            return False
        return True

    tuners = dedupe_tuners([tuner for tuner in tuners if selected(tuner)], args.package)
    log(f"[export_tuned_table] {len(tuners)} tuner(s) selected from {args.package}")
    if args.list:
        for tuner in tuners:
            identity = kernel_identity(tuner)
            reason = refusal_reason(tuner, identity["wrappers"])
            status = f"unsupported: {reason}" if reason else "ok"
            log(
                f"  {identity['kernel_id']:40s} op={getattr(tuner, '__name__', '?')} keys={list(getattr(tuner, 'keys', []))} {status}"
            )
        return 0

    vendor = vendor_name(args.package)
    backend = args.backend or VENDOR_TO_BACKEND.get(vendor, "")
    if not backend:
        parser.error(
            f"cannot derive the backend from vendor '{vendor}'; pass --backend"
        )
    engine, source_db = resolve_engine(libentry, args.db)
    fingerprint = detect_fingerprint(
        backend, vendor, args.device_index, args.device_name
    )
    if not fingerprint["device_name"]:
        log(
            "[export_tuned_table] warning: no device name detected; the C++ loader will accept this table on any device of the backend"
        )
    table = build_table(
        tuners,
        engine,
        libentry,
        fingerprint,
        source_db,
        log=log,
        include_unsupported=not args.exclude_unsupported,
    )
    if args.cache_namespace:
        if len(table["kernels"]) != 1:
            parser.error("--cache-namespace requires exactly one exported kernel")
        table["kernels"][0]["cache_namespace"] = args.cache_namespace
    text = json.dumps(table, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        total = sum(len(kernel["entries"]) for kernel in table["kernels"])
        log(
            f"[export_tuned_table] wrote {args.output}: {len(table['kernels'])} kernels, {total} entries, device '{fingerprint['device_name']}'"
        )
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
