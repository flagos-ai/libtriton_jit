"""Offline table helpers and --help must work without accelerator packages."""

import os
import subprocess
import sys
from pathlib import Path

scripts = str(Path(sys.argv[1]).resolve())
# Run without site packages, and explicitly reject accelerator imports even on
# machines which have them installed. Check import/help before optional probes.
code = r"""
import importlib.abc
import os
import sys
import types

attempts = []
class MissingAccelerators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "torch_npu", "triton"}:
            attempts.append(fullname)
            raise ModuleNotFoundError("blocked optional dependency: " + fullname, name=fullname)

sys.meta_path.insert(0, MissingAccelerators())
sys.path.insert(0, sys.argv[1])
import export_tuned_table as exporter
import tuned_resolver as resolver
assert not attempts, attempts
try:
    exporter.main(["--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("--help did not exit")
assert not attempts, attempts
assert exporter.coerce_value(42.0, int) == 42
assert resolver.build_grid([2, 3], None, (), {}) == (2, 3)
grid = resolver.build_grid("(math.ceil(M / META['B']),)",
                           types.SimpleNamespace(arg_names=["M"]), [17], {})
assert grid({"B": 4}) == (5,)
assert "num_warps" in exporter.triton_config_fields()
assert resolver._native_autotuner_class() is None
fp = exporter.detect_fingerprint("NPU", "ascend", 0, "offline-device")
assert fp["device_name"] == "offline-device"
# Actual kernel discovery must still report a missing required backend.
try:
    exporter.import_package_tree("not_a_real_kernel_package")
except ModuleNotFoundError as error:
    assert error.name == "torch", error
else:
    raise AssertionError("kernel discovery swallowed a required backend failure")
print("offline imports, --help, helpers, and required dependency errors passed")
"""
subprocess.run(
    [sys.executable, "-S", "-c", code, scripts],
    env={**os.environ, "TRITON_JIT_BACKEND": "NPU"},
    check=True,
)
