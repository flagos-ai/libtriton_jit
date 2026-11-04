"""Compile an NPU override without mutating the caller's process environment."""

import json
import sys
from pathlib import Path


def main():
    request = json.load(sys.stdin)
    sys.path[:] = request.pop("python_path")
    result_path = Path(request.pop("result_path"))
    from standalone_compile import compile_a_kernel, torch

    # This child has a fresh device context; select the caller's visible device.
    torch.npu.set_device(request["device_id"])
    result_path.write_text(compile_a_kernel(**request))


if __name__ == "__main__":
    main()
