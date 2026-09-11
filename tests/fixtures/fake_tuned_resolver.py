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
"""Stand-in for scripts/tuned_resolver.py used by tests/test_torch_tuned_resolver.cpp.

Echoes what the C++ bridge handed over so the test can check the conversion,
and exercises the three answer shapes: a configuration, "unsupported", and an
exception.
"""

import threading
import time

import torch

entered = threading.Event()

_calls = {}


class TritonDtype(str):
    """The marker the bridge instantiates for a TritonDtype argument."""


def _describe(value):
    if isinstance(value, TritonDtype):
        return f"dtype:{value}"
    if torch.is_tensor(value):
        return f"tensor:{value.dtype}:{list(value.shape)}"
    return f"{type(value).__name__}:{value}"


def resolve(
    package,
    kernel_id,
    args,
    kwargs,
    source_path=None,
    grid=None,
    device_index=None,
    stream=None,
):
    _calls[kernel_id] = _calls.get(kernel_id, 0) + 1
    if kernel_id == "slow_kernel":
        entered.set()
        time.sleep(0.5)
    if kernel_id == "heuristic_kernel":
        return {
            "kernel_id": kernel_id,
            "unsupported": "kernel is wrapped in @triton.heuristics",
        }
    if kernel_id == "boom_kernel":
        raise RuntimeError("deliberate failure")
    kw = [
        ["BLOCK_M", 64],
        ["EVEN_K", True],
        ["SCALE", 0.25],
        ["MODE", "fast"],
        ["CALLS", _calls[kernel_id]],
        ["PACKAGE", package],
        ["SRC", str(source_path)],
        ["GRID", str(grid)],
        ["DEVICE", str(device_index)],
        ["STREAM", str(stream)],
        [
            "KW",
            ",".join(
                f"{k}={_describe(v) if isinstance(v, TritonDtype) else v}"
                for k, v in kwargs.items()
            ),
        ],
    ]
    for index, value in enumerate(args):
        kw.append([f"ARG{index}", _describe(value)])
    return {
        "kernel_id": kernel_id,
        "key_columns": [
            {"name": "M", "strategy": "log"},
            {"name": "N", "strategy": "log"},
        ],
        "dtype_keys": 2,
        "key": [1024, 4096, "torch.float16", "torch.bfloat16"],
        "num_warps": 8,
        "num_stages": 2,
        "extra": {"maxnreg": "255"},
        "kwargs": kw,
    }
