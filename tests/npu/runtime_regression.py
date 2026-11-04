"""Exercise actual Ascend kernels, PR67 preparation and online/offline tuning."""

import argparse
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TRITON_JIT_BACKEND"] = "NPU"
# Initialize Torch and the Ascend extension before Triton discovers plugins.
import torch  # noqa: E402
import torch_npu  # noqa: F401, E402
import triton  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("build_dir")
parser.add_argument("scripts_dir")
parser.add_argument(
    "mode", choices=["grid", "prepare", "autotune", "counts", "options", "all"]
)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, str(Path(args.build_dir).resolve()))
sys.path.insert(0, str(Path(args.scripts_dir).resolve()))
import npu_kernels  # noqa: E402
import npu_runtime_probe as probe  # noqa: E402

torch.npu.set_device(0)
stream = torch.npu.current_stream(0).npu_stream
kernel_path = str(Path(npu_kernels.__file__).resolve())
results = []
device_info = probe.info()
assert device_info["ai_core_status"] == 0 and device_info["ai_core_count"] > 0
vector_cores = device_info["ai_core_count"] * 2
print(
    json.dumps(
        {
            "device": device_info,
            "auto_blockify_env": os.getenv("TRITON_ALL_BLOCKS_PARALLEL"),
        }
    ),
    flush=True,
)


@contextmanager
def auto_blockify_environment(value):
    previous = os.environ.get("TRITON_ALL_BLOCKS_PARALLEL")
    os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TRITON_ALL_BLOCKS_PARALLEL", None)
        else:
            os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = previous


def opposite_environment():
    return (
        "0"
        if os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "false").lower() in ("true", "1")
        else "1"
    )


def tensors(rows, dtype):
    # Small integer inputs give exact FP16/BF16 reference outputs, including
    # non-multiple-of-tile boundary shapes. No external data is used.
    cols = 64
    a_cpu = (torch.arange(rows * cols).reshape(rows, cols) % 7 - 3).to(dtype)
    x_cpu = (torch.arange(cols) % 5 - 2).to(dtype)
    reference = torch.mv(a_cpu.float(), x_cpu.float()).to(dtype)
    a = a_cpu.to("npu:0")
    x = x_cpu.to("npu:0")
    y = torch.full((rows,), float("nan"), dtype=dtype, device="npu:0")
    return a, x, y, reference


def record(name, function):
    try:
        detail = function()
        result = {"test": name, "passed": True, "detail": detail}
    except Exception as error:
        result = {
            "test": name,
            "passed": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    results.append(result)
    print(json.dumps(result), flush=True)


def grid_case(rows, dtype):
    a, x, y, reference = tensors(rows, dtype)
    native = torch.empty_like(y)
    npu_kernels.gemv_kernel[(triton.cdiv(rows, 4),)](
        a, x, native, rows, 64, 4, 64, num_warps=4, num_stages=1
    )
    torch.npu.synchronize()
    torch.testing.assert_close(native.cpu(), reference, rtol=0, atol=0)
    probe.launch(kernel_path, "gemv_kernel", a, x, y, rows, 64, 4, 64, stream, False)
    torch.npu.synchronize()
    actual = y.cpu()
    invalid = torch.isnan(actual).nonzero().flatten().tolist()
    assert (
        not invalid
    ), f"Unwritten rows={len(invalid)}, first={invalid[:8]}, grid={triton.cdiv(rows, 4)}"
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    return {
        "rows": rows,
        "dtype": str(dtype),
        "blocks": triton.cdiv(rows, 4),
        "native_and_cpp_match_reference": True,
    }


def prepare_case(dtype):
    a, x, y, reference = tensors(300, dtype)
    assert probe.prepare(kernel_path, "gemv_kernel", a, x, y, 300, 64, 4, 64, stream)
    torch.npu.synchronize()
    assert torch.isnan(y.cpu()).all(), "prepare() changed output"
    with auto_blockify_environment(opposite_environment()):
        probe.launch(kernel_path, "gemv_kernel", a, x, y, 300, 64, 4, 64, stream, True)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), reference, rtol=0, atol=0)
    cold_a, cold_x, cold_y, _ = tensors(301, dtype)
    try:
        probe.launch(
            kernel_path,
            "gemv_kernel",
            cold_a,
            cold_x,
            cold_y,
            301,
            64,
            4,
            64,
            stream,
            True,
        )
    except RuntimeError as error:
        assert "frozen" in str(error).lower() or "freeze" in str(error).lower(), str(
            error
        )
    else:
        raise AssertionError("cold launch succeeded under ScopedFreeze")
    return {
        "prepare_did_not_launch": True,
        "warm_frozen_launch": True,
        "cold_frozen_refused": True,
    }


def count_case(blocks, auto_blockify=-1):
    counts = torch.zeros(blocks, dtype=torch.int32, device="npu:0")
    native = torch.zeros_like(counts)
    options = {} if auto_blockify < 0 else {"enable_auto_blockify": bool(auto_blockify)}
    # FlagTree's Linalg compiler reads the environment even with an explicit
    # option. Use a consistent native reference; the C++ call below deliberately
    # keeps the conflicting environment to exercise our isolated compilation.
    native_mode = (
        os.getenv("TRITON_ALL_BLOCKS_PARALLEL", "0")
        if auto_blockify < 0
        else str(auto_blockify)
    )
    with auto_blockify_environment(native_mode):
        npu_kernels.counter_kernel[(blocks,)](
            native, blocks, num_warps=4, num_stages=1, **options
        )
    before = os.getenv("TRITON_ALL_BLOCKS_PARALLEL")
    probe.count_programs(kernel_path, counts, blocks, stream, auto_blockify)
    assert (
        os.getenv("TRITON_ALL_BLOCKS_PARALLEL") == before
    ), "compile changed the caller environment"
    torch.npu.synchronize()
    torch.testing.assert_close(
        native.cpu(), torch.ones(blocks, dtype=torch.int32), rtol=0, atol=0
    )
    torch.testing.assert_close(
        counts.cpu(), torch.ones(blocks, dtype=torch.int32), rtol=0, atol=0
    )
    counts.zero_()
    with auto_blockify_environment(opposite_environment()):
        probe.count_programs(kernel_path, counts, blocks, stream, auto_blockify, True)
        torch.npu.synchronize()
    torch.testing.assert_close(
        counts.cpu(), torch.ones(blocks, dtype=torch.int32), rtol=0, atol=0
    )
    return {
        "blocks": blocks,
        "every_program_executed_exactly_once": True,
        "frozen_launch_after_environment_change": True,
        "explicit_option": auto_blockify,
        "native_reference_environment": native_mode,
        "cpp_environment": before,
        "caller_environment_unchanged": True,
    }


def explicit_option_case(enabled):
    # Exercise option precedence against the opposite environment setting.
    with auto_blockify_environment("0" if enabled else "1"):
        return count_case(vector_cores * 5, int(enabled))


def atomic_tail_workaround_case(blocks):
    # The native auto-blockify tail defect is external to this runtime. Verify
    # the supported per-kernel opt-out while the process default stays enabled.
    with auto_blockify_environment("1"):
        return count_case(blocks, 0)


def disk_cache_case():
    from standalone_compile import compile_a_kernel

    directories = []
    for enabled in (False, True, False, True):
        with auto_blockify_environment("1" if enabled else "0"):
            directory = compile_a_kernel(
                kernel_path, "counter_kernel", "*i32:16,113", 4, 1, 0
            )
        metadata = json.loads((Path(directory) / "counter_kernel.json").read_text())
        assert metadata["enable_auto_blockify"] is enabled, metadata
        assert metadata["TRITON_ALL_BLOCKS_PARALLEL"] == ("1" if enabled else "0")
        directories.append(directory)
    assert directories[0] != directories[1], "on/off reused the same disk artifact"
    assert directories[:2] == directories[2:], "identical options missed the disk cache"
    return {"distinct_mode_artifacts": True, "mode_metadata": True, "cache_hits": True}


def autotune_case(dtype):
    import tuned_resolver

    a, x, y, reference = tensors(128, dtype)
    dtype_names = [str(dtype)] * 3
    probe.install_resolver(str(Path(kernel_path).parent))
    config = probe.resolve(
        kernel_path, a, x, y, 128, 64, dtype_names, stream, False, False
    )
    assert config, "online resolver returned a miss"
    torch.npu.synchronize()
    assert torch.isnan(y.cpu()).all(), "resolver modified caller output"
    warm = probe.resolve(
        kernel_path, a, x, y, 128, 64, dtype_names, stream, True, False
    )
    assert warm == config, "warm frozen config lookup changed config"
    bm, bn = config["BLOCK_M"], config["BLOCK_N"]
    probe.launch(kernel_path, "native_gemv", a, x, y, 128, 64, bm, bn, stream, False)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu(), reference, rtol=0, atol=0)
    output = Path(args.output).with_name("table_" + str(dtype).split(".")[-1] + ".json")
    tuned_resolver.export_native_table(
        str(output), "npu_kernels", backend="NPU", vendor="ascend", device_index=0
    )
    probe.load_table(str(output))
    offline = probe.resolve(
        kernel_path, a, x, y, 128, 64, dtype_names, stream, True, True
    )
    assert offline == config, f"offline round trip mismatch: {offline} != {config}"
    return {
        "selected": config,
        "isolated_output": True,
        "warm_frozen_hit": True,
        "cpp_result_matches": True,
        "offline_round_trip": True,
        "table": str(output),
    }


if args.mode in ["grid", "all"]:
    for dtype in [torch.float16, torch.bfloat16]:
        for rows in [128, 160, 161, 164, 256, 512, 1024]:
            record(
                f"grid/{dtype}/{rows}",
                lambda rows=rows, dtype=dtype: grid_case(rows, dtype),
            )
if args.mode in ["prepare", "all"]:
    for dtype in [torch.float16, torch.bfloat16]:
        record(f"prepare/{dtype}", lambda dtype=dtype: prepare_case(dtype))
if args.mode in ["counts", "options", "all"]:
    # Non-integral waves have a separate upstream atomic-loop bug; see the
    # standalone reproducer. Full waves still expose both runtime regressions.
    for blocks in [
        max(1, vector_cores - 8),
        vector_cores,
        vector_cores * 2,
        vector_cores * 3,
        vector_cores * 4,
    ]:
        record(f"count_programs/{blocks}", lambda blocks=blocks: count_case(blocks))
if args.mode in ["options", "all"]:
    for enabled in [False, True]:
        record(
            f"explicit_option/{enabled}",
            lambda enabled=enabled: explicit_option_case(enabled),
        )
    record("disk_cache/mode_identity", disk_cache_case)
    for blocks in [41, 64, 128]:
        record(
            f"atomic_tail/explicit_disable/{blocks}",
            lambda blocks=blocks: atomic_tail_workaround_case(blocks),
        )
if args.mode in ["autotune", "all"]:
    for dtype in [torch.float16, torch.bfloat16]:
        record(f"autotune/{dtype}", lambda dtype=dtype: autotune_case(dtype))
summary = {
    "auto_blockify_env": os.getenv("TRITON_ALL_BLOCKS_PARALLEL"),
    "device": device_info,
    "passed": sum(r["passed"] for r in results),
    "total": len(results),
    "results": results,
}
Path(args.output).write_text(json.dumps(summary, indent=2))
print(json.dumps({"passed": summary["passed"], "total": summary["total"]}), flush=True)
raise SystemExit(0 if all(r["passed"] for r in results) else 1)
