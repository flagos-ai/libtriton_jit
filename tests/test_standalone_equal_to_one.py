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

"""An equal-to-one argument must be handed to Triton as a constant only.

Declaring it in the runtime signature as well leaves the compiler with two
descriptions of the same parameter. Triton folds it either way, so the emitted
code is identical, but the ASTSource -- and therefore the compilation cache key
-- stops matching the one the native Python runtime builds for the same call,
and the C++ and Python paths silently stop sharing compiled kernels.

The test intercepts ASTSource instead of compiling, so it needs no device.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _backend_init import initialize_backend  # noqa: E402

initialize_backend()
import triton  # noqa: E402


class _Captured(Exception):
    """Raised by the ASTSource stub to stop before the real compilation."""


def load_module(path):
    spec = importlib.util.spec_from_file_location("standalone_compile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_ast_source(module, kernel_path, kernel_name, signature):
    """Return the signature/constant mappings handed to triton.compiler.ASTSource."""
    seen = {}
    original = triton.compiler.ASTSource

    def stub(*args, **kwargs):
        seen.update(kwargs)
        raise _Captured

    triton.compiler.ASTSource = stub
    try:
        module.compile_a_kernel(kernel_path, kernel_name, signature)
    except _Captured:
        pass
    finally:
        triton.compiler.ASTSource = original

    assert seen, "ASTSource was never constructed"
    constants = seen.get("constexprs", seen.get("constants"))
    assert constants is not None, "ASTSource received neither constexprs nor constants"
    return seen["signature"], constants


def lookup(mapping, index, name):
    """Read an entry regardless of the key convention of the Triton version."""
    for key in ((index,), index, name):
        if key in mapping:
            return mapping[key]
    return None


def main():
    script_path, fixture_path = sys.argv[1], sys.argv[2]
    module = load_module(script_path)

    # strided_copy_kernel(out_ptr, stride, size, BLOCK) with stride == 1.
    signature, constants = capture_ast_source(
        module, fixture_path, "strided_copy_kernel", "*fp32:16,i32:1,i32:16,64"
    )

    stride_type = lookup(signature, 1, "stride")
    assert stride_type in (None, "constexpr"), (
        "an equal-to-one argument must not also be declared as a runtime type "
        f"(got {stride_type!r}); this splits the compilation cache key away from "
        "the one the native Python runtime builds"
    )
    assert (
        lookup(constants, 1, "stride") == 1
    ), "an equal-to-one argument must be a constant of 1"

    # Neighbouring arguments must be untouched: a divisibility hint stays a
    # runtime argument, and a real constexpr keeps its value.
    assert (
        lookup(signature, 2, "size") == "i32"
    ), "a divisible-by-16 argument stays a runtime i32"
    assert (
        lookup(constants, 2, "size") is None
    ), "a divisible-by-16 argument is not a constant"
    assert lookup(constants, 3, "BLOCK") == 64, "a constexpr keeps its value"

    # Without an equal-to-one argument the same parameter stays a runtime i32.
    signature, constants = capture_ast_source(
        module, fixture_path, "strided_copy_kernel", "*fp32:16,i32,i32:16,64"
    )
    assert (
        lookup(signature, 1, "stride") == "i32"
    ), "a plain integer stays a runtime argument"
    assert (
        lookup(constants, 1, "stride") is None
    ), "a plain integer must not become a constant"


if __name__ == "__main__":
    main()
