import importlib.util
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Union

import torch
import triton
from packaging.version import Version

# do not specifier a cache dir for libtriton jit now
# pylint: disable-next=wrong-import-position
triton_version = Version(triton.__version__)

DESC = """
Script to compile Triton Jit functions into Compiled Kernel and cache it into a cache dir.
We return the kernel name and subdir path in which the kernel files site.

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.
"""


# backends/nvidia/driver.py
def ty_to_cpp(ty):
    if ty[0] == "*":
        return "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def parse_bool(s: str) -> bool:
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError(f"{s} is not a boolean")


def constexpr(s: str) -> Union[int, float]:
    """Extract constexpr from signature"""
    if s == "nullopt":
        return None
    try:
        ret = parse_bool(s)
        return ret
    except ValueError:
        pass
    try:
        ret = int(s)
        return ret
    except ValueError:
        pass
    try:
        ret = float(s)
        return ret
    except ValueError:
        pass
    return None


# compiler/code_generator.py
def kernel_suffix(signature, specialization):
    # suffix format:
    # <argid><'c' if equal to 1><'d' if divisible by 16><'e' if divisible by 8>
    suffix = ""
    for i, _ in enumerate(signature):
        suffix += str(i)
        if i in specialization.equal_to_1:
            suffix += "c"
        if i in specialization.divisible_by_16:
            suffix += "d"
    return suffix


def _compile_a_kernel(
    fn: triton.runtime.JITFunction,
    signature: str,
    num_warps: int = 4,
    num_stages: int = 3,
    device_id: int = 0,
) -> Tuple[str, str]:
    """compile a kernel."""
    # static signature
    constexpr_indices = [i for (i, p) in enumerate(fn.params) if p.is_constexpr]
    # non_constexpr_indices = [i for (i, p) in enumerate(fn.params) if not p.is_constexpr]
    # specialised_indices = [
    #     i
    #     for (i, p) in enumerate(fn.params)
    #     if (not p.do_not_specialize) and (not p.is_constexpr)
    # ]

    # validate and parse signature
    # example "*fp32, *fp32:16, i32, 1024"
    # for bool use i1, for boolean values, use 0 or 1.
    # split it

    signature: List[str] = list(map(lambda s: s.strip(" "), signature.split(",")))
    num_args = len(signature)
    assert num_args == len(
        fn.params
    ), f"number of argument mismatch:  Actual({num_args}), Function Definition({len(fn.params)})"

    constants = {
        i: constexpr(s) for i, s in enumerate(signature) if i in constexpr_indices
    }
    assert len(constants) == len(
        constexpr_indices
    ), f"number of constexpr mismatch:  Actual({len(constants)}), Function Definition({len(constexpr_indices)})"

    # signature, no specializations here
    signature_without_spec = {
        i: s.split(":")[0] for i, s in enumerate(signature) if i not in constants
    }

    # specialization: divisibility by 16 or equal to 1
    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = tuple(i for i, h in hints.items() if h == 16)
    equal_to_1 = tuple(i for i, h in hints.items() if h == 1)

    if triton_version.major == 3 and triton_version.minor == 1:
        attrs = triton.compiler.AttrsDescriptor(
            divisible_by_16=divisible_by_16, equal_to_1=equal_to_1
        )
    elif triton_version.major == 3 and triton_version.minor == 2:
        attrs = triton.backends.compiler.AttrsDescriptor.from_dict(
            {
                "arg_properties": {
                    "tt.divisibility": divisible_by_16,
                    "tt.equal_to": equal_to_1,
                },
                "cls": "AttrsDescriptor",
            }
        )
    elif triton_version.major == 3 and triton_version.minor == 3:
        attrs = {(k,): [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    elif triton_version.major == 3 and triton_version.minor == 4:
        attrs = {(k,): [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    elif triton_version.major == 3 and triton_version.minor == 5:
        attrs = {(k,): [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}    
    else:
        raise RuntimeError(
            "Triton may change APIs, we cannot ensure compatibility here now. You can goto https://github.com/flagos-ai/libtriton_jit to raise an issue about supporting your triton version. Triton 3.1/3.2/3.3/3.4/3.5 are supported now."
        )


    # integer 1 in value, but the corresponding ArgType in static signature is not constexpr are added into constants
    for i in equal_to_1:
        constants.update({i: 1})
        # Nones in value, but the corresponding ArgType in static signature is not constexpr are added into constants
    for i, v in signature_without_spec.items():
        if v == "nullopt":
            constants[i] = None
            signature_without_spec[i] = "constexpr"

    if triton_version == Version("3.1.0"):
        src = triton.compiler.ASTSource(
            fn=fn,
            constants=constants,
            signature=signature_without_spec,
            attrs=attrs,
        )
    elif triton_version == Version("3.2.0"):
        arg_names = fn.arg_names
        _constants = {arg_names[i]: v for i, v in constants.items()}
        _signature_without_spec = {
            arg_names[i]: v for i, v in signature_without_spec.items()
        }
        constants, signature_without_spec = _constants, _signature_without_spec

        src = triton.compiler.ASTSource(
            fn=fn,
            constants=constants,
            signature=signature_without_spec,
            attrs=attrs,
        )
    elif triton_version >= Version("3.3.0"):
        arg_names = fn.arg_names
        _constants = {(i,): v for i, v in constants.items()}
        _signature_without_spec = {}
        for i in range(num_args):
            if i in signature_without_spec:
                _signature_without_spec[arg_names[i]] = signature_without_spec[i]
            elif i in constants:
                _signature_without_spec[arg_names[i]] = "constexpr"
            else:
                raise ValueError("wtf")
        constants, signature_without_spec = _constants, _signature_without_spec

        src = triton.compiler.ASTSource(
            fn=fn,
            signature=signature_without_spec,
            constexprs=constants,
            attrs=attrs,
        )

    # STEP1: JITFunction, constants, signature, specialization

    # STEP2: compile options for the backend
    opts = {"num_warps": num_warps, "num_stages": num_stages}

    with torch.cuda.device(device_id):
        # STEP3: ast source, target, compile options
        target: triton.backends.compiler.GPUTarget = (
            triton.runtime.driver.active.get_current_target()
        )
        ccinfo: triton.compiler.CompiledKernel = triton.compile(
            src, target, options=opts
        )

    # kernel's hash may not equals the dir in cache
    from triton.runtime.cache import get_cache_manager

    cache_manager = get_cache_manager(ccinfo.hash)
    return cache_manager.cache_dir


def compile_a_kernel(
    source_path,
    fn_name,
    signature: str,
    num_warps: int = 4,
    num_stages: int = 3,
    device_id: int = 0,
):
    # get jit function
    source_path = Path(source_path)
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name)

    # unwrap JITFunction from Autotuner or Heuristics, contarct: decorated fn is stored in the fn attribute
    while not (type(fn) is triton.runtime.JITFunction):
        fn = fn.fn

    return _compile_a_kernel(fn, signature, num_warps, num_stages, device_id)


if __name__ == "__main__":
    # command-line arguments
    parser = ArgumentParser(description=DESC)
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
    parser.add_argument(
        "--device-id",
        type=int,
        default="",
        help="Targeting device id",
        required=True,
    )
    parser.add_argument(
        "--num-warps",
        "-w",
        type=int,
        default=4,
        help="Number of warps to launch the kernel",
    )
    parser.add_argument(
        "--num-stages",
        "-ns",
        type=int,
        default=3,
        help="Number of stages (meta-parameter of the kernel)",
    )
    parser.add_argument(
        "--signature", "-s", type=str, help="Signature of the kernel", required=True
    )
    args = parser.parse_args()

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path).expanduser()
    kerel_hash = compile_a_kernel(
        arg_path, args.kernel_name, args.signature, args.num_warps, args.num_stages
    )
    print(kerel_hash)
