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
import sys


def load_module(path):
    spec = importlib.util.spec_from_file_location("standalone_compile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_value_error(fn, value):
    try:
        fn(value)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {value!r}")


def main():
    module = load_module(sys.argv[1])

    assert module._bracket_aware_split("*fp32:16,(fp32,i32),64") == [
        "*fp32:16",
        "(fp32,i32)",
        "64",
    ]
    assert module._parse_type_token("(fp32,i32)") == ("fp32", "i32")

    expect_value_error(module._bracket_aware_split, "*fp32,(fp32,i32")
    expect_value_error(module._bracket_aware_split, "*fp32,fp32)")
    expect_value_error(module._parse_type_token, "()")
    expect_value_error(module._parse_type_token, "((fp32,i32),i64)")

    assert module.generate_arg_layout(
        ["*fp32:16", "(fp32,i32)", "64"], [2]
    ) == [
        {"type": "ptr", "dtype": "fp32"},
        {"type": "fp32"},
        {"type": "i32"},
    ]
    assert module._normalize_gcu_signature(("fp32", "i32")) == ("fp32", "i32")
    expect_value_error(module._normalize_gcu_signature, ("fp64", "i32"))


if __name__ == "__main__":
    main()
