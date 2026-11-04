"""Plain @triton.autotune kernels through the resolver ("python") and through
the C++ bridge ("bridge"). Real Triton, one GPU, no FlagGems.

Usage: native_tuner.py <scripts dir> python
       native_tuner.py <scripts dir> bridge <dir of tuned_runtime_probe>
"""

import importlib
import inspect
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.language as tl

scripts, mode = sys.argv[1], sys.argv[2]
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
sys.path.insert(0, scripts)
resolver = importlib.import_module("tuned_resolver")
k = importlib.import_module("native_kernels")
failures = []
path = k.__file__
grid_expr = "(triton.cdiv(n, META['BLOCK']),)"


def check(name, condition, **details):
    print(json.dumps({"case": name, "pass": bool(condition), **details}), flush=True)
    if not condition:
        failures.append(name)


def tensors(n, dtype=torch.float32):
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    return x, torch.full((n,), -17.0, device="cuda", dtype=dtype)


def fresh(kernel, **options):
    """A new Autotuner over the same JIT function: same candidates, own cache."""
    return triton.autotune(
        configs=k.configs(), key=["n"], **k.bench_options(), **options
    )(kernel.fn)


def launch(tuner, cfg, *args, **kwargs):
    n = args[2]
    tuner.fn.run(
        *args,
        grid=(triton.cdiv(n, cfg.kwargs["BLOCK"]),),
        warmup=False,
        **kwargs,
        **cfg.all_kwargs(),
    )
    torch.cuda.synchronize()


def counting_bench(tuner):
    calls = []
    original = type(tuner)._bench

    def bench(self, *args, config, **kwargs):
        calls.append(config.kwargs["BLOCK"])
        return original(self, *args, config=config, **kwargs)

    tuner._bench = types.MethodType(bench, tuner)
    return calls


if mode == "python":
    # parity with native run(): fixed fake timings -> same choice, key and order
    x, y = tensors(257)
    observed = []
    fixed = {128: [3.0, 3.0, 3.0], 256: [1.0, 1.0, 1.0], 512: [2.0, 2.0, 2.0]}

    def fake_bench(self, *args, config, **kwargs):
        observed.append(config.kwargs["BLOCK"])
        return fixed[config.kwargs["BLOCK"]]

    native = fresh(k.native_vector_kernel)
    native._bench = types.MethodType(fake_bench, native)
    native.run(
        x, y, 257, grid=lambda meta: (triton.cdiv(257, meta["BLOCK"]),), warmup=False
    )
    torch.cuda.synchronize()
    native_choice, native_key = native.best_config.all_kwargs(), list(native.cache)[0]
    native_order = list(observed)
    observed.clear()
    y.fill_(-17)
    t = fresh(k.native_vector_kernel)
    t._bench = types.MethodType(fake_bench, t)
    out = resolver.resolve_with_native_tuner(t, (x, y, 257), {}, grid=grid_expr)
    check(
        "parity",
        "unsupported" not in out
        and out["kwargs"] == [["BLOCK", native_choice["BLOCK"]]]
        and out["num_warps"] == native_choice["num_warps"]
        and tuple(out["key"]) == native_key
        and observed == native_order
        and torch.all(y == -17).item(),
        answer=out,
        native_key=list(native_key),
    )
    # a real cold selection on isolated clones, then the answer launched by the caller
    n = 65536
    x, y = tensors(n)
    t = fresh(k.native_vector_kernel)
    calls = counting_bench(t)
    out = resolver.resolve_with_native_tuner(t, (x, y, n), {}, grid=grid_expr)
    torch.cuda.synchronize()
    check(
        "isolated_cold",
        "unsupported" not in out and torch.all(y == -17).item() and len(calls) == 3,
        answer=out,
        benchmarks=len(calls),
    )
    cfg = t.cache[tuple(out["key"])]
    launch(t, cfg, x, y, n)
    check("launch_correct", torch.equal(y, x * 2))
    with patch.dict(os.environ, {"TRITON_JIT_BENCH_MAX_BYTES": "1"}), patch.object(
        resolver,
        "_argument_cloner",
        side_effect=AssertionError("cache hit copied inputs"),
    ):
        warm = resolver.resolve_with_native_tuner(t, (x, y, n), {})
    check("warm_no_grid", warm.get("kwargs") == out["kwargs"] and len(calls) == 3)
    # an ordinary Python call after the resolve hits the same cache: no re-benchmark
    y.fill_(-17)
    t.run(x, y, n, grid=lambda meta: (triton.cdiv(n, meta["BLOCK"]),), warmup=False)
    torch.cuda.synchronize()
    check("python_call_shares_cache", len(calls) == 3 and torch.equal(y, x * 2))
    if "cache_results" in inspect.signature(triton.autotune).parameters:
        with tempfile.TemporaryDirectory() as cache_dir, patch.dict(
            os.environ, {"TRITON_CACHE_DIR": cache_dir}
        ):
            disk_writer = fresh(k.native_vector_kernel, cache_results=True)
            writes = counting_bench(disk_writer)
            first = resolver.resolve_with_native_tuner(
                disk_writer, (x, y, n), {}, grid=grid_expr
            )
            check("disk_cache_write", "unsupported" not in first and len(writes) == 3)
            disk_reader = fresh(k.native_vector_kernel, cache_results=True)
            reads = counting_bench(disk_reader)
            with patch.object(
                resolver,
                "_argument_cloner",
                side_effect=AssertionError("disk hit copied inputs"),
            ):
                second = resolver.resolve_with_native_tuner(disk_reader, (x, y, n), {})
            check("disk_cache_read_no_grid", second == first and not reads)
    else:
        print(
            json.dumps(
                {"case": "disk_cache", "skip": "Triton has no cache_results option"}
            )
        )
    # reset_to_zero: selection leaves the caller's output alone; the launch adds to it
    acc = fresh(k.native_accumulate_kernel, reset_to_zero=["y"])
    x = torch.ones(257, device="cuda")
    y = torch.full_like(x, 17)
    out = resolver.resolve_with_native_tuner(acc, (x, y, 257), {}, grid=grid_expr)
    torch.cuda.synchronize()
    check(
        "reset_isolated",
        "unsupported" not in out and torch.all(y == 17).item(),
        answer=out,
    )
    cfg = acc.cache[tuple(out["key"])]
    launch(acc, cfg, x, y, 257)
    check("reset_caller_owns_init", torch.all(y == 18).item())
    y.zero_()
    launch(acc, cfg, x, y, 257)
    check("reset_after_caller_zero", torch.all(y == 1).item())
    restored = fresh(k.native_accumulate_kernel, restore_value=["y"])
    local = resolver._local_tuner(restored)
    scratch, live = torch.full_like(x, 3), torch.full_like(x, 7)

    def hook_args(output):
        if hasattr(restored, "restore_idx"):
            return (x, output, 257)
        return {"x": x, "y": output, "n": 257}

    local.pre_hook(hook_args(scratch))
    restored.pre_hook(hook_args(live))
    scratch.fill_(33)
    live.fill_(77)
    local.post_hook(hook_args(scratch), exception=None)
    restored.post_hook(hook_args(live), exception=None)
    check(
        "restore_interleaving",
        torch.all(scratch == 3).item() and torch.all(live == 7).item(),
    )
    hook_calls = []
    hooked = fresh(
        k.native_vector_kernel, pre_hook=lambda *args, **kwargs: hook_calls.append(1)
    )
    hooked.run(
        x, y, 257, grid=lambda meta: (triton.cdiv(257, meta["BLOCK"]),), warmup=False
    )
    before = len(hook_calls)
    online = resolver.resolve_with_native_tuner(hooked, (x, y, 257), {}, grid=grid_expr)
    offline = resolver.native_kernel_table(hooked)
    check(
        "custom_hook_refused_online_and_offline",
        "custom tuner hooks" in online.get("unsupported", "")
        and online["unsupported"] == offline.get("unsupported")
        and not offline["entries"]
        and len(hook_calls) == before,
    )
    # DTYPE constexpr through the marker, found by kernel name and file
    x, y16 = tensors(513, torch.float16)
    out = resolver.resolve(
        "native_kernels",
        "native_cast_kernel",
        (x, y16, 513),
        {"DTYPE": resolver.TritonDtype("float16")},
        source_path=path,
        grid=grid_expr,
        device_index=0,
    )
    torch.cuda.synchronize()
    check(
        "dtype_marker",
        "unsupported" not in out
        and out["key"] == [513, "torch.float32", "torch.float16"],
        answer=out,
    )
    cfg = k.native_cast_kernel.cache[tuple(out["key"])]
    launch(k.native_cast_kernel, cfg, x, y16, 513, DTYPE=tl.float16)
    check("dtype_launch_correct", torch.equal(y16, (x * 2).half()))
    # a float key column is refused before anything is benchmarked
    fk = k.native_float_key_kernel
    calls = counting_bench(fk)
    out = resolver.resolve(
        "native_kernels",
        "native_float_key_kernel",
        (x, y16, 513, 0.5),
        {},
        source_path=path,
        grid=grid_expr,
    )
    check(
        "float_key_refused",
        "SCALE" in out.get("unsupported", "") and not calls,
        answer=out,
    )
    # export what this process resolved
    table = resolver.export_native_table(
        os.path.join(tempfile.mkdtemp(), "native.json"),
        [t, acc, k.native_cast_kernel],
        backend="CUDA",
        device_index=0,
    )
    rows = {kernel["kernel_id"]: kernel for kernel in table["kernels"]}
    check(
        "export",
        rows["native_vector_kernel"]["entries"][0]["key"]
        == [n, "torch.float32", "torch.float32"]
        and rows["native_cast_kernel"]["entries"][0]["key"]
        == [513, "torch.float32", "torch.float16"]
        and rows["native_accumulate_kernel"]["dtype_keys"] == 2
        and table["fingerprint"]["backend"] == "CUDA",
        kernels=sorted(rows),
    )
elif mode == "bridge":
    sys.path.insert(0, sys.argv[3])
    probe = importlib.import_module("tuned_runtime_probe")
    probe.install_native("native_kernels", str(here))
    python_calls = []
    original_resolve = resolver.resolve

    def counted_resolve(*args, **kwargs):
        python_calls.append(kwargs.get("device_index"))
        return original_resolve(*args, **kwargs)

    resolver.resolve = counted_resolve
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    n = 65536
    x, y = tensors(n)
    f32 = ["torch.float32", "torch.float32"]
    calls = counting_bench(k.native_vector_kernel)
    cfg = probe.resolve_native(
        "native_vector_kernel",
        path,
        x,
        y,
        n,
        f32,
        "",
        grid_expr,
        False,
        stream.cuda_stream,
    )
    torch.cuda.synchronize()
    check(
        "bridge_cold",
        bool(cfg)
        and len(calls) == 3
        and python_calls == [0]
        and torch.all(y == -17).item(),
        config=cfg,
        benchmarks=len(calls),
    )
    again = probe.resolve_native(
        "native_vector_kernel",
        path,
        x,
        y,
        n,
        f32,
        "",
        grid_expr,
        False,
        stream.cuda_stream,
    )
    check(
        "bridge_warm_stays_in_cpp",
        again == cfg and python_calls == [0] and len(calls) == 3,
    )
    # the Python side shares the answer: an ordinary call neither re-benchmarks nor disagrees
    k.native_vector_kernel.run(
        x, y, n, grid=lambda meta: (triton.cdiv(n, meta["BLOCK"]),), warmup=False
    )
    torch.cuda.synchronize()
    check(
        "bridge_python_shares_cache",
        len(calls) == 3
        and k.native_vector_kernel.best_config.kwargs["BLOCK"] == cfg["BLOCK"]
        and torch.equal(y, x * 2),
    )
    # DTYPE=tl.float16 travels as a TritonDtype through ArgValue
    xc, yc = tensors(513, torch.float16)
    cfg16 = probe.resolve_native(
        "native_cast_kernel",
        path,
        xc,
        yc,
        513,
        ["torch.float32", "torch.float16"],
        "float16",
        grid_expr,
        False,
        stream.cuda_stream,
    )
    torch.cuda.synchronize()
    check(
        "bridge_dtype",
        bool(cfg16) and python_calls == [0, 0] and torch.all(yc == -17).item(),
        config=cfg16,
    )
    chosen = k.native_cast_kernel.cache[(513, "torch.float32", "torch.float16")]
    launch(k.native_cast_kernel, chosen, xc, yc, 513, DTYPE=tl.float16)
    check(
        "bridge_dtype_launch",
        chosen.kwargs["BLOCK"] == cfg16["BLOCK"] and torch.equal(yc, (xc * 2).half()),
    )
    # frozen or captured: a cold key is refused before Python is entered
    x2, y2 = tensors(4097)
    try:
        probe.resolve_native(
            "native_vector_kernel",
            path,
            x2,
            y2,
            4097,
            f32,
            "",
            grid_expr,
            True,
            stream.cuda_stream,
        )
        frozen_refused = False
    except RuntimeError:
        frozen_refused = True
    check("bridge_frozen_cold_refused", frozen_refused and python_calls == [0, 0])
    denied = False
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        try:
            probe.resolve_native(
                "native_vector_kernel",
                path,
                x2,
                y2,
                4097,
                f32,
                "",
                grid_expr,
                False,
                stream.cuda_stream,
            )
        except RuntimeError as error:
            denied = "captur" in str(error)
    check("bridge_capture_cold_refused", denied and python_calls == [0, 0])
    # export from this process, read back by the C++ table reader
    table_path = os.path.join(tempfile.mkdtemp(), "native.json")
    resolver.export_native_table(
        table_path, "native_kernels", backend="CUDA", device_index=0
    )
    probe.load_table(table_path)
    found = probe.find_native("native_vector_kernel", path, n, f32)
    missing = probe.find_native("native_vector_kernel", path, n + 1, f32)
    check("bridge_offline_roundtrip", found == cfg and not missing, found=found)
    spec = importlib.util.spec_from_file_location("native_kernels_duplicate", path)
    duplicate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = duplicate
    spec.loader.exec_module(duplicate)
    duplicate.native_vector_kernel.cache[
        (n + 2, *f32)
    ] = k.native_vector_kernel.best_config
    transient = resolver._local_tuner(k.native_vector_kernel)
    table = resolver.export_native_table(table_path, "native_kernels", backend="CUDA")
    probe.load_table(table_path)
    check(
        "bridge_duplicate_module_export",
        len(table["kernels"]) == 4
        and transient not in resolver.native_tuners()
        and probe.find_native("native_vector_kernel", path, n, f32) == cfg
        and probe.find_native("native_vector_kernel", path, n + 2, f32) == cfg,
    )
else:
    raise ValueError(mode)
raise SystemExit(bool(failures))
