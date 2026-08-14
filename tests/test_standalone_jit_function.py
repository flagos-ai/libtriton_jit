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

import importlib.util
import linecache
import os
import sys
import tempfile
from pathlib import Path

import triton


def load_module(path):
    spec = importlib.util.spec_from_file_location("standalone_compile_jit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_exception(exception_type, fn):
    try:
        fn()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def make_token(module, source_path, function_name):
    source = source_path.read_bytes()
    return (
        f"@jit:{str(source_path.resolve()).encode().hex()}:"
        f"{function_name.encode().hex()}:{module._fnv1a64(source)}"
    )


def module_name_for(module, source_path, fingerprint):
    return (
        f"_triton_jit_callee_{fingerprint}_"
        f"{module._fnv1a64(str(source_path).encode('utf-8'))}"
    )


def expect_failed_load_restores_registries(
    module, source_path, function_name, fingerprint, exception_type
):
    module_name = module_name_for(module, source_path, fingerprint)
    linecache_key = str(source_path)
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    previous_linecache = linecache.cache.get(linecache_key, missing)

    expect_exception(
        exception_type,
        lambda: module._load_jitfunction(
            str(source_path), function_name, fingerprint
        ),
    )

    restored_module = sys.modules.get(module_name, missing)
    restored_linecache = linecache.cache.get(linecache_key, missing)
    assert restored_module is previous_module
    assert restored_linecache is previous_linecache


def main():
    module = load_module(sys.argv[1])
    fixture = Path(sys.argv[2]).resolve()
    token = make_token(module, fixture, "mul_func")
    path, name, fingerprint = module._parse_jitfunction_token(token)

    assert path == str(fixture)
    assert name == "mul_func"
    assert fingerprint == module._fnv1a64(fixture.read_bytes())
    assert isinstance(
        module._load_jitfunction(path, name, fingerprint),
        triton.runtime.JITFunction,
    )
    resolved = module._resolve_jitfunction_constants([token, "*fp32"])
    assert list(resolved) == [0]
    assert isinstance(resolved[0], triton.runtime.JITFunction)
    assert module.generate_arg_layout([token, "*fp32", "i32"], []) == [
        {"type": "ptr", "dtype": "fp32"},
        {"type": "i32"},
    ]

    invalid_tokens = [
        "jit:00:00:0000000000000000",
        "@jit:00:00",
        "@jit:0:61:0000000000000000",
        "@jit:zz:61:0000000000000000",
        "@jit::61:0000000000000000",
        "@jit:2f746d70::0000000000000000",
        "@jit:2f746d70:61:1234",
        "@jit:2f746d70:61:gggggggggggggggg",
    ]
    for invalid in invalid_tokens:
        expect_exception(ValueError, lambda value=invalid: module._parse_jitfunction_token(value))

    expect_exception(
        ValueError,
        lambda: module._load_jitfunction(path, name, "0000000000000000"),
    )
    expect_failed_load_restores_registries(
        module,
        fixture,
        "missing_function",
        fingerprint,
        AttributeError,
    )
    expect_failed_load_restores_registries(
        module,
        fixture,
        "not_a_jit_function",
        fingerprint,
        TypeError,
    )

    with tempfile.TemporaryDirectory() as directory:
        mutable_path = Path(directory) / "mutable.py"
        first_source = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def operation(x, y):\n"
            "    return x * y\n"
        )
        second_source = first_source.replace("x * y", "x + y")
        assert len(first_source) == len(second_source)
        mutable_path.write_text(first_source, encoding="utf-8")
        first_fingerprint = module._fnv1a64(mutable_path.read_bytes())
        first_function = module._load_jitfunction(
            str(mutable_path), "operation", first_fingerprint
        )
        mutable_path.write_text(second_source, encoding="utf-8")
        second_fingerprint = module._fnv1a64(mutable_path.read_bytes())
        second_function = module._load_jitfunction(
            str(mutable_path), "operation", second_fingerprint
        )
        assert first_fingerprint != second_fingerprint
        assert first_function.src != second_function.src

        cycle_path = Path(directory) / "cycle.py"
        cycle_path.write_text(
            "class Wrapper:\n"
            "    pass\n"
            "cycle = Wrapper()\n"
            "cycle.fn = cycle\n",
            encoding="utf-8",
        )
        cycle_fingerprint = module._fnv1a64(cycle_path.read_bytes())
        expect_failed_load_restores_registries(
            module,
            cycle_path,
            "cycle",
            cycle_fingerprint,
            TypeError,
        )

        broken_path = Path(directory) / "broken.py"
        broken_path.write_text(
            "raise RuntimeError('module execution failed')\n",
            encoding="utf-8",
        )
        broken_fingerprint = module._fnv1a64(broken_path.read_bytes())
        expect_failed_load_restores_registries(
            module,
            broken_path,
            "operation",
            broken_fingerprint,
            RuntimeError,
        )

    if os.environ.get("TRITON_JIT_TEST_COMPILE") == "1":
        compile_fixture = Path(sys.argv[3]).resolve()
        compile_module = load_module(compile_fixture)
        compile_token = make_token(module, compile_fixture, "mul_func")
        cache_dir = Path(
            module._compile_a_kernel(
                compile_module.apply_jitfunction_kernel,
                f"*fp32,{compile_token},32,32",
                num_warps=1,
                num_stages=1,
            )
        )
        assert cache_dir.is_dir()
        assert any(cache_dir.iterdir())


if __name__ == "__main__":
    main()
