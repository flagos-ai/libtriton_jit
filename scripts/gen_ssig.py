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
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List

import triton


@dataclass
class Signature:
    num_args: int
    constexpr_indices: List[int]
    non_constexpr_indices: List[int]
    specialised_indices: List[int]
    specialised_no_alignment_indices: List[int]


def static_signature(f: triton.runtime.JITFunction):
    args_names = f.arg_names
    arg_num = len(args_names)
    constexpr_indices = [i for (i, p) in enumerate(f.params) if p.is_constexpr]
    non_constexpr_indices = [i for (i, p) in enumerate(f.params) if not p.is_constexpr]
    specialised_indices = [
        i
        for (i, p) in enumerate(f.params)
        if (not p.do_not_specialize)
        and (not getattr(p, "do_not_specialize_on_alignment", False))
        and (not p.is_constexpr)
    ]
    specialised_no_alignment_indices = [
        i
        for (i, p) in enumerate(f.params)
        if (not p.do_not_specialize)
        and getattr(p, "do_not_specialize_on_alignment", False)
        and (not p.is_constexpr)
    ]
    return Signature(
        arg_num,
        constexpr_indices,
        non_constexpr_indices,
        specialised_indices,
        specialised_no_alignment_indices,
    )


def extract_static_signature(source_path, fn_name):
    source_path = Path(source_path)
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name)

    # unwrap JITFunction from Autotuner or Heuristics, contarct: decorated fn is stored in the fn attribute
    while not (type(fn) is triton.runtime.JITFunction):
        fn = fn.fn

    sig = static_signature(fn)

    # convert to list of int for c++ processing
    arg_types = []
    for i in range(sig.num_args):
        if i in sig.constexpr_indices:
            arg_types.append(2)
        elif i in sig.specialised_indices:
            arg_types.append(1)
        elif i in sig.specialised_no_alignment_indices:
            arg_types.append(3)
        else:  # non-specialzed
            arg_types.append(0)
    return arg_types


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(
        description="generate server-side signature, that is, static part of the full signature"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to Python source containing desired kernel in its scope. File will be executed.",
    )
    parser.add_argument(
        "--kernel-name",
        "-n",
        type=str,
        default="",
        help="Name of the kernel to compile",
        required=True,
    )

    args = parser.parse_args()

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path).expanduser()
    arg_types = extract_static_signature(arg_path, args.kernel_name)

    print(arg_types)
