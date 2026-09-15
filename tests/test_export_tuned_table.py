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
"""Tests for scripts/export_tuned_table.py that need neither FlagGems nor a
device: a synthetic SQLite database laid out exactly like LibTuner's
(physical table ``<config_table_name>-<md5(key_0key_1...)>``, ``key_i``
primary-key columns, one value column per Config field) plus stand-in tuner
objects.

Usage:
    python tests/test_export_tuned_table.py <path to export_tuned_table.py>
    python tests/test_export_tuned_table.py <script> --write-fixture <json>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import types
from typing import Any, Dict, List, Optional

failures = 0


def check(condition: bool, message: str) -> None:
    global failures
    if not condition:
        failures += 1
        print(f"FAILED: {message}", file=sys.stderr)


def load_script(path: str):
    spec = importlib.util.spec_from_file_location("export_tuned_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Stand-ins for triton / libentry objects


class FakeConfig:
    def __init__(
        self,
        kwargs: Dict[str, Any],
        num_warps=4,
        num_stages=3,
        num_ctas=1,
        maxnreg=None,
        pre_hook=None,
    ):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def all_kwargs(self) -> Dict[str, Any]:
        return {
            **self.kwargs,
            "num_warps": self.num_warps,
            "num_ctas": self.num_ctas,
            "num_stages": self.num_stages,
            "maxnreg": self.maxnreg,
        }


class JITFunction:  # the class name is what the exporter looks for
    def __init__(self, py_fn, arg_names: List[str]):
        self.fn = py_fn
        self.arg_names = arg_names
        self.cache_key = "cachekey-" + py_fn.__name__


class Heuristics:
    def __init__(self, fn):
        self.fn = fn


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

    def __init__(
        self,
        name: str,
        fn,
        keys: List[str],
        strategy: Optional[List[Any]],
        configs: List[FakeConfig],
    ):
        self.__name__ = name
        self.fn = fn
        self.keys = keys
        self.strategy = strategy
        self.configs = configs
        self.configs_hash = hashlib.md5(
            ",".join(map(str, range(len(configs)))).encode()
        ).hexdigest()[:32]
        self.config_table_name = f"{name}_{hashlib.md5(name.encode()).hexdigest()[:32]}"


def fake_libentry(package: str) -> types.ModuleType:
    module = types.ModuleType(f"{package}.utils.libentry")
    module.LibTuner = LibTuner
    return module


def sgemv_n_kernel(
    A, x, y, m, n, BLOCK_M, BLOCK_K
):  # noqa: N803 - mirrors a Triton kernel signature
    pass


def mm_kernel(
    a, b, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_K, SPLIT_K, SCALE
):  # noqa: N803
    pass


def softmax_kernel(x, y, n, BLOCK):  # noqa: N803
    pass


# --------------------------------------------------------------------------
# Synthetic database


def physical_name(config_table_name: str, n_key_columns: int) -> str:
    key_names = "".join(f"key_{i}" for i in range(n_key_columns))
    return f"{config_table_name}-{hashlib.md5(key_names.encode()).hexdigest()}"


def create_db(path: str, tuners: Dict[str, LibTuner]) -> None:
    import sqlalchemy

    engine = sqlalchemy.create_engine(f"sqlite:///{path}")
    metadata = sqlalchemy.MetaData()

    def table(
        config_table_name: str, key_types: List[Any], value_columns: Dict[str, Any]
    ):
        columns = [
            sqlalchemy.Column(f"key_{i}", key_type, primary_key=True)
            for i, key_type in enumerate(key_types)
        ]
        columns += [
            sqlalchemy.Column(name, col_type)
            for name, col_type in value_columns.items()
        ]
        return sqlalchemy.Table(
            physical_name(config_table_name, len(key_types)), metadata, *columns
        )

    big, flt, txt = sqlalchemy.BigInteger, sqlalchemy.Float, sqlalchemy.String
    # sgemv: keys (m, n) + 3 tensor dtypes; bool-free
    sgemv = table(
        tuners["sgemv_n"].config_table_name,
        [big, big, txt, txt, txt],
        {
            "BLOCK_M": big,
            "BLOCK_K": big,
            "num_warps": big,
            "num_ctas": big,
            "num_stages": big,
        },
    )
    # mm: keys (M, N, K) normalised + 2 dtypes; EVEN_K stored as int, maxnreg set, a float constexpr
    mm = table(
        tuners["mm"].config_table_name,
        [big, big, big, txt, txt],
        {
            "SPLIT_K": big,
            "EVEN_K": big,
            "BLOCK_K": big,
            "BLOCK_N": big,
            "BLOCK_M": big,
            "SCALE": flt,
            "num_warps": big,
            "num_ctas": big,
            "num_stages": big,
            "maxnreg": big,
        },
    )
    # a second physical table for mm with only one dtype key (fewer rows) -> must be dropped
    mm_narrow = table(
        tuners["mm"].config_table_name,
        [big, big, big, txt],
        {"BLOCK_M": big, "num_warps": big, "num_ctas": big, "num_stages": big},
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            sgemv.insert(),
            [
                {
                    "key_0": 4096,
                    "key_1": 4096,
                    "key_2": "torch.float32",
                    "key_3": "torch.float32",
                    "key_4": "torch.float32",
                    "BLOCK_M": 32,
                    "BLOCK_K": 128,
                    "num_warps": 4,
                    "num_ctas": 1,
                    "num_stages": 4,
                },
                {
                    "key_0": 1,
                    "key_1": 8192,
                    "key_2": "torch.float32",
                    "key_3": "torch.float32",
                    "key_4": "torch.float32",
                    "BLOCK_M": 8,
                    "BLOCK_K": 256,
                    "num_warps": 8,
                    "num_ctas": 1,
                    "num_stages": 2,
                },
            ],
        )
        connection.execute(
            mm.insert(),
            [
                {
                    "key_0": 1024,
                    "key_1": 4096,
                    "key_2": 4096,
                    "key_3": "torch.float16",
                    "key_4": "torch.float16",
                    "SPLIT_K": 1,
                    "EVEN_K": 1,
                    "BLOCK_K": 64,
                    "BLOCK_N": 128,
                    "BLOCK_M": 128,
                    "SCALE": 0.5,
                    "num_warps": 8,
                    "num_ctas": 2,
                    "num_stages": 3,
                    "maxnreg": 255,
                },
                {
                    "key_0": 64,
                    "key_1": 64,
                    "key_2": 64,
                    "key_3": "torch.bfloat16",
                    "key_4": "torch.bfloat16",
                    "SPLIT_K": 4,
                    "EVEN_K": 0,
                    "BLOCK_K": 32,
                    "BLOCK_N": 64,
                    "BLOCK_M": 64,
                    "SCALE": 1.0,
                    "num_warps": 4,
                    "num_ctas": 1,
                    "num_stages": 2,
                    "maxnreg": 255,
                },
            ],
        )
        connection.execute(
            mm_narrow.insert(),
            [
                {
                    "key_0": 64,
                    "key_1": 64,
                    "key_2": 64,
                    "key_3": "torch.float16",
                    "BLOCK_M": 16,
                    "num_warps": 4,
                    "num_ctas": 1,
                    "num_stages": 2,
                },
            ],
        )
    engine.dispose()


def make_tuners(blas: bool = False) -> Dict[str, LibTuner]:
    sgemv_configs = [
        FakeConfig({"BLOCK_M": bm, "BLOCK_K": bk})
        for bm in (8, 32)
        for bk in (128, 256)
    ]
    mm_configs = [
        FakeConfig(
            {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": 64,
                "EVEN_K": True,
                "SPLIT_K": 1,
                "SCALE": 0.5,
            }
        ),
        FakeConfig(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "EVEN_K": False,
                "SPLIT_K": 4,
                "SCALE": 1.0,
            }
        ),
    ]
    softmax_configs = [FakeConfig({"BLOCK": 1024})]
    return {
        "sgemv_n": LibTuner(
            "sgemv_n",
            JITFunction(
                sgemv_n_kernel, ["A", "x", "y", "m", "n", "BLOCK_M", "BLOCK_K"]
            ),
            ["m", "n"],
            None,
            sgemv_configs,
        ),
        "mm": LibTuner(
            "mm",
            JITFunction(
                mm_kernel,
                [
                    "a",
                    "b",
                    "c",
                    "M",
                    "N",
                    "K",
                    "BLOCK_M",
                    "BLOCK_N",
                    "BLOCK_K",
                    "EVEN_K",
                    "SPLIT_K",
                    "SCALE",
                ],
            ),
            ["M", "N", "K"],
            [log2_strategy, log2_strategy, align32_strategy],
            mm_configs,
        ),
        "softmax": LibTuner(
            "softmax",
            Heuristics(JITFunction(softmax_kernel, ["x", "y", "n", "BLOCK"])),
            ["n"],
            None,
            softmax_configs,
        ),
    }


# --------------------------------------------------------------------------
# Tests


def run(script_path: str, fixture_out: Optional[str]) -> int:
    import sqlalchemy

    export = load_script(script_path)
    workdir = tempfile.mkdtemp(prefix="export_tuned_table_test_")
    db_path = os.path.join(workdir, "TunedConfig_nvidia_triton_3_6.db")
    tuners = make_tuners()
    create_db(db_path, tuners)
    engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    fingerprint = {
        "backend": "CUDA",
        "vendor": "nvidia",
        "device_name": "NVIDIA H800",
        "arch": "sm_90",
        "triton_version": "3.6.0",
        "libtriton_jit_version": "0.1.0",
    }
    messages: List[str] = []
    table = export.build_table(
        list(tuners.values()),
        engine,
        fake_libentry("flag_gems"),
        fingerprint,
        f"sqlite:///{db_path}",
        log=messages.append,
    )

    check(table["format_version"] == 2, "format_version")
    check(table["fingerprint"] == fingerprint, "fingerprint passthrough")
    kernels = {kernel["kernel_id"]: kernel for kernel in table["kernels"]}
    check(
        set(kernels) == {"sgemv_n_kernel", "mm_kernel", "softmax_kernel"},
        f"kernel ids: {sorted(kernels)}",
    )
    check(
        all(k["cache_namespace"] == k["source_path"] != "" for k in kernels.values()),
        "v2 namespaces default to the source path",
    )

    sgemv = kernels["sgemv_n_kernel"]
    check(sgemv["op_name"] == "sgemv_n", "op_name")
    check(
        sgemv["key_columns"]
        == [{"name": "m", "strategy": "default"}, {"name": "n", "strategy": "default"}],
        f"sgemv key_columns {sgemv['key_columns']}",
    )
    check(sgemv["dtype_keys"] == 3, f"sgemv dtype_keys {sgemv['dtype_keys']}")
    check(len(sgemv["entries"]) == 2, "sgemv entry count")
    first = next(e for e in sgemv["entries"] if e["key"][0] == 4096)
    check(
        first["key"] == [4096, 4096, "torch.float32", "torch.float32", "torch.float32"],
        f"sgemv key {first['key']}",
    )
    check(
        first["kwargs"] == [["BLOCK_M", 32], ["BLOCK_K", 128]],
        f"sgemv kwargs order {first['kwargs']}",
    )
    check(first["num_warps"] == 4 and first["num_stages"] == 4, "sgemv nw/ns")
    check("extra" not in first, "num_ctas == 1 must not appear in extra")
    check(
        sgemv["source_sha256"] != ""
        and sgemv["source_path"].endswith("test_export_tuned_table.py"),
        "source hash",
    )
    check(
        sgemv["candidate_set_hash"] == tuners["sgemv_n"].configs_hash, "candidate hash"
    )

    mm = kernels["mm_kernel"]
    check(
        [c["strategy"] for c in mm["key_columns"]] == ["log", "log", "align32"],
        f"mm strategies {mm['key_columns']}",
    )
    check(
        mm["dtype_keys"] == 2,
        f"mm dtype_keys {mm['dtype_keys']} (narrow table must be dropped)",
    )
    check(len(mm["entries"]) == 2, "mm entry count")
    check(
        any("different dtype-key counts" in m for m in messages), "width warning logged"
    )
    big = next(e for e in mm["entries"] if e["key"][0] == 1024)
    check(
        [name for name, _ in big["kwargs"]]
        == ["BLOCK_M", "BLOCK_N", "BLOCK_K", "EVEN_K", "SPLIT_K", "SCALE"],
        f"mm kwargs follow kernel parameter order (database column order was scrambled): {big['kwargs']}",
    )
    values = dict(big["kwargs"])
    check(
        values["EVEN_K"] is True, f"EVEN_K recovered as bool, got {values['EVEN_K']!r}"
    )
    check(
        values["BLOCK_M"] == 128 and isinstance(values["BLOCK_M"], int), "BLOCK_M int"
    )
    check(values.get("SCALE") == 0.5, "float constexpr kept")
    check(
        big["extra"] == {"num_ctas": "2", "maxnreg": "255"}, f"extra {big.get('extra')}"
    )
    small = next(e for e in mm["entries"] if e["key"][0] == 64)
    check(dict(small["kwargs"])["EVEN_K"] is False, "EVEN_K false recovered")
    check(small["extra"] == {"maxnreg": "255"}, f"small extra {small.get('extra')}")

    softmax = kernels["softmax_kernel"]
    check(
        "heuristics" in softmax.get("unsupported", ""),
        f"softmax refused: {softmax.get('unsupported')}",
    )
    check(
        softmax["entries"] == [] and softmax["key_columns"] == [],
        "unsupported kernel carries no rows",
    )

    # pre_hook refusal
    hooked = LibTuner(
        "hooked",
        JITFunction(sgemv_n_kernel, ["A", "x", "y", "m", "n", "BLOCK_M", "BLOCK_K"]),
        ["m"],
        None,
        [FakeConfig({"BLOCK_M": 8, "BLOCK_K": 8}, pre_hook=lambda nargs: None)],
    )
    reason = export.refusal_reason(hooked, [])
    check(reason is not None and "pre_hook" in reason, f"pre_hook refusal: {reason}")

    # FlagBLAS dialect renames align32
    names = export.strategy_names(
        tuners["mm"], fake_libentry("flag_blas"), blas_dialect=True
    )
    check(names == ["log", "log", "align32_ceil"], f"blas dialect strategies {names}")
    names = export.strategy_names(
        tuners["mm"], fake_libentry("flag_gems"), blas_dialect=False
    )
    check(names == ["log", "log", "align32"], f"gems dialect strategies {names}")

    # duplicate kernel ids are an error
    twin = LibTuner(
        "sgemv_n_twin",
        JITFunction(sgemv_n_kernel, ["A", "x", "y", "m", "n", "BLOCK_M", "BLOCK_K"]),
        ["m", "n"],
        None,
        [],
    )
    try:
        export.build_table(
            [tuners["sgemv_n"], twin],
            engine,
            fake_libentry("flag_gems"),
            fingerprint,
            "x",
        )
        check(False, "duplicate kernel id must raise")
    except ValueError as error:
        check("sgemv_n_kernel" in str(error), f"duplicate error text: {error}")

    # a value column that is not a kernel parameter is an error
    bad = LibTuner(
        "mm",
        JITFunction(mm_kernel, ["a", "b", "c", "M", "N", "K"]),
        ["M", "N", "K"],
        [log2_strategy, log2_strategy, align32_strategy],
        [],
    )
    bad.config_table_name = tuners["mm"].config_table_name
    try:
        export.export_tuner(
            bad, engine, fake_libentry("flag_gems"), False, lambda m: None
        )
        check(False, "unknown column must raise")
    except ValueError as error:
        check(
            "is not a parameter of the kernel" in str(error),
            f"unknown column error: {error}",
        )

    # vendor discovery must work for the FlagBLAS layout (module-level vendor_module)
    # and the FlagGems layout (libentry._state.vendor_module)
    import types as _types

    def install_fake_package(name: str, layout: str) -> None:
        pkg = _types.ModuleType(name)
        pkg.__path__ = []  # mark as package
        utils = _types.ModuleType(f"{name}.utils")
        utils.__path__ = []
        lib = _types.ModuleType(f"{name}.utils.libentry")
        vendor = _types.SimpleNamespace(
            vendor_info=_types.SimpleNamespace(vendor_name="iluvatar")
        )
        if layout == "blas":
            lib.vendor_module = vendor
        else:
            lib._state = _types.SimpleNamespace(vendor_module=vendor)
        sys.modules[name] = pkg
        sys.modules[f"{name}.utils"] = utils
        sys.modules[f"{name}.utils.libentry"] = lib

    install_fake_package("flag_fake_blas", "blas")
    install_fake_package("flag_fake_gems", "gems")
    check(
        export.vendor_name("flag_fake_blas") == "iluvatar",
        "vendor from module-level vendor_module",
    )
    check(
        export.vendor_name("flag_fake_gems") == "iluvatar",
        "vendor from libentry._state",
    )
    check(
        export.VENDOR_TO_BACKEND[export.vendor_name("flag_fake_blas")] == "IX",
        "vendor -> backend",
    )

    # the JSON must be a valid document and, optionally, become a C++ fixture
    text = json.dumps(table, indent=2)
    json.loads(text)
    if fixture_out:
        fixed = dict(table)
        fixed["generated_at"] = "2026-09-08T00:00:00+00:00"
        fixed[
            "source_db"
        ] = "sqlite:///TunedConfig_nvidia_triton_3_6.db (synthetic, tests/test_export_tuned_table.py)"
        for kernel in fixed["kernels"]:
            kernel["source_path"] = "tests/test_export_tuned_table.py"
            kernel["cache_namespace"] = kernel["source_path"]
        with open(fixture_out, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(fixed, indent=2) + "\n")
        print(f"wrote {fixture_out}")

    engine.dispose()
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("export_tuned_table: all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    fixture = None
    if "--write-fixture" in sys.argv:
        fixture = sys.argv[sys.argv.index("--write-fixture") + 1]
    sys.exit(run(sys.argv[1], fixture))
