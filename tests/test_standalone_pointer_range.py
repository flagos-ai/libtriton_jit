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

"""The ":range32" pointer hint must become tt.pointer_range = 32, exactly where
the native Python runtime would add it.

On HCU the native runtime tags every tensor whose storage fits a signed 32-bit
offset with the backend's "S" specialization, which its parse_attr turns into
tt.pointer_range = 32 -- but only while buffer ops are enabled
(AMDGCN_USE_BUFFER_OPS). The C++ side emits ":range32" for such pointers;
standalone_compile must translate it into the same attribute, keep it next to
the alignment hint, keep the pointer a plain runtime argument, follow the same
switch, and reject the hint where it makes no sense.

The test intercepts ASTSource instead of compiling, so it needs no device.
"""

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _backend_init import initialize_backend  # noqa: E402

initialize_backend()
import triton  # noqa: E402

RANGE = ["tt.pointer_range", 32]
DIV16 = ["tt.divisibility", 16]


class _Captured(Exception):
    """Raised by the ASTSource stub to stop before the real compilation."""


def load_module(path):
    spec = importlib.util.spec_from_file_location("standalone_compile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(module, kernel_path, signature):
    seen = {}
    original = triton.compiler.ASTSource

    def stub(*args, **kwargs):
        seen.update(kwargs)
        raise _Captured

    triton.compiler.ASTSource = stub
    try:
        module.compile_a_kernel(kernel_path, "strided_copy_kernel", signature)
    except _Captured:
        pass
    finally:
        triton.compiler.ASTSource = original
    assert seen, "ASTSource was never constructed"
    return seen


def lookup(mapping, index, name):
    """Read an entry regardless of the key convention of the Triton version."""
    for key in ((index,), index, name):
        if key in mapping:
            return mapping[key]
    return None


def pointer_attrs(seen):
    attrs = seen["attrs"]
    return attrs.get((0,), []) if isinstance(attrs, dict) else None


def expect_rejected(module, kernel_path, signature, message, what):
    try:
        capture(module, kernel_path, signature)
    except ValueError as e:
        assert message in str(e), f"{what}: unexpected error {e!r}"
        return
    raise AssertionError(f"{what} was accepted: {signature}")


def main():
    script_path, fixture_path = sys.argv[1], sys.argv[2]
    module = load_module(script_path)
    # strided_copy_kernel(out_ptr, stride, size, BLOCK)
    rest = ",i32,i32:16,64"
    on_hcu = module.get_backend() == "HCU"
    previous = os.environ.get("AMDGCN_USE_BUFFER_OPS")
    if on_hcu:
        # Exercise the main path even if the caller's environment disabled it.
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "1"
    supported = module._pointer_range_supported()
    if on_hcu and module.triton_version.release[:2] >= (3, 3):
        assert supported, "HCU with buffer ops on must support range32"

    for token, with_range, without_range in (
        ("*fp32:16:range32", [DIV16, RANGE], [DIV16]),
        ("*fp32:range32", [RANGE], []),
        ("*fp32:16", [DIV16], [DIV16]),
    ):
        seen = capture(module, fixture_path, token + rest)
        got = pointer_attrs(seen)
        if got is not None:  # dict-style attrs (Triton >= 3.3)
            expected = with_range if supported else without_range
            assert got == expected, f"{token}: attrs {got}, expected {expected}"
            assert seen["attrs"].get((2,)) == [
                DIV16
            ], "neighbouring hints must be untouched"
        # The hint is compile-time metadata: the pointer stays a plain runtime argument.
        assert (
            lookup(seen["signature"], 0, "out_ptr") == "*fp32"
        ), "the pointer type must stay '*fp32'"

    if supported:
        # ASTSource.hash() serializes attrs with str(), so their order is part
        # of the cache key. The native runtime lists arguments in order; a
        # range-only pointer must not be moved behind the integer hints.
        attrs = capture(module, fixture_path, "*fp32:range32,i32:16,i32:16,64")["attrs"]
        native = {(0,): [RANGE], (1,): [DIV16], (2,): [DIV16]}
        assert str(attrs) == str(native), f"attrs {attrs} != native order {native}"

    expect_rejected(
        module,
        fixture_path,
        "*fp32,i32:range32,i32:16,64",
        "only valid on a pointer",
        "a range hint on an integer",
    )
    expect_rejected(
        module,
        fixture_path,
        "*fp32:range32:range32" + rest,
        "duplicate",
        "a duplicate range hint",
    )

    if on_hcu and supported:
        # Disabling buffer ops must drop the attribute, as it does natively.
        os.environ["AMDGCN_USE_BUFFER_OPS"] = "0"
        assert (
            not module._pointer_range_supported()
        ), "AMDGCN_USE_BUFFER_OPS=0 must disable range32"
        got = pointer_attrs(capture(module, fixture_path, "*fp32:16:range32" + rest))
        assert got == [
            DIV16
        ], f"buffer ops off: attrs {got}, expected only divisibility"

    if previous is None:
        os.environ.pop("AMDGCN_USE_BUFFER_OPS", None)
    else:
        os.environ["AMDGCN_USE_BUFFER_OPS"] = previous

    print(f"PASS (backend={module.get_backend()}, pointer range supported={supported})")


if __name__ == "__main__":
    main()
