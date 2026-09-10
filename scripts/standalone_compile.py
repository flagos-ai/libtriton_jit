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
import io
import json
import linecache
import os
import re
import sys
import tokenize
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple, Union

# NPU, MTGPU, MACA, GCU, and MLU require this before importing triton
backend_env = os.environ.get("TRITON_JIT_BACKEND", "").upper()
if backend_env in ["NPU", "MTGPU", "MACA", "GCU", "MLU"]:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import torch  # noqa: E402

# Import backend-specific modules to activate Triton driver
if backend_env == "MTGPU":
    try:
        import torch_musa  # noqa: F401 - Activate MUSA device and Triton mtgpu driver
    except ImportError:
        print("Warning: torch_musa not available, MTGPU backend may not work")
elif backend_env == "GCU":
    try:
        import torch_gcu  # noqa: F401 - Activate GCU device and Triton enflame driver
    except ImportError:
        print("Warning: torch_gcu not available, GCU backend may not work")
elif backend_env == "MLU":
    try:
        import torch_mlu  # noqa: F401 - Activate MLU device and Triton mlu driver
    except ImportError:
        print("Warning: torch_mlu not available, MLU backend may not work")

import triton  # noqa: E402
from packaging.version import Version  # noqa: E402


def get_backend():
    """Get backend from environment variable (set by C++ at runtime)."""
    return os.environ.get("TRITON_JIT_BACKEND", "CUDA").upper()


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


def sig_to_npu_type(sig: str) -> dict:
    """Convert signature type string to NPU arg_layout format.

    Args:
        sig: Type string like "*fp32", "i64", "*fp16:16", etc.

    Returns:
        dict with 'type' and optionally 'dtype' keys, or None for constexpr
    """
    # Remove specialization suffix
    sig = sig.split(":")[0].strip()

    if sig.startswith("*"):
        # Pointer type
        dtype = sig[1:]  # e.g., "fp32", "fp16", "i32"
        return {"type": "ptr", "dtype": dtype}
    elif sig in ("i64", "u64"):
        return {"type": "i64"}
    elif sig in ("i32", "u32"):
        return {"type": "i32"}
    elif sig in ("i16", "u16", "i8", "u8", "i1", "u1"):
        return {"type": "i32"}  # Promoted to i32
    elif sig in ("fp64", "f64"):
        return {"type": "fp64"}
    elif sig in ("fp32", "f32"):
        return {"type": "fp32"}
    elif sig in ("fp16", "f16", "bf16"):
        return {"type": "fp32"}  # Promoted to fp32
    elif sig == "constexpr" or sig == "nullopt":
        return {"type": "constexpr"}
    else:
        # Try to parse as constexpr number
        try:
            int(sig)
            return {"type": "constexpr"}
        except ValueError:
            try:
                float(sig)
                return {"type": "constexpr"}
            except ValueError:
                # Unknown type, default to i64
                return {"type": "i64"}


def _bracket_aware_split(sig: str) -> List[str]:
    """Split on top-level commas while preserving parenthesized tuple groups."""
    parts = []
    current = []
    depth = 0

    for ch in sig:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                raise ValueError(f"unmatched ')' in signature: {sig}")
            depth -= 1

        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if not part:
                raise ValueError(f"empty token in signature: {sig}")
            parts.append(part)
            current = []
        else:
            current.append(ch)

    if depth != 0:
        raise ValueError(f"unmatched '(' in signature: {sig}")

    final = "".join(current).strip()
    if final:
        parts.append(final)
    elif parts:
        raise ValueError(f"empty token in signature: {sig}")
    return parts


def _parse_type_token(token: str):
    """Convert a grouped tuple token into Triton's nested signature form."""
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        inner = token[1:-1].strip()
        if not inner:
            raise ValueError("runtime tuple signature must not be empty")
        elements = _bracket_aware_split(inner)
        if any("(" in element or ")" in element for element in elements):
            raise ValueError("nested runtime tuple signatures are not supported")
        return tuple(elements)
    return token


def _normalize_gcu_signature(value):
    """Normalize GCU signatures without changing runtime tuple ABI widths."""
    if isinstance(value, tuple):
        if "fp64" in value:
            raise ValueError("GCU runtime tuple arguments do not support fp64 elements")
        return value
    return "fp32" if value == "fp64" else value


def _fnv1a64(data: bytes) -> str:
    """Return a stable, lowercase FNV-1a 64 fingerprint."""
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _decode_jitfunction_field(value: str, field_name: str) -> str:
    if not value or len(value) % 2 or re.fullmatch(r"[0-9a-fA-F]+", value) is None:
        raise ValueError(f"invalid hex-encoded JITFunction {field_name}")
    try:
        decoded = bytes.fromhex(value).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"JITFunction {field_name} is not valid UTF-8") from error
    if not decoded:
        raise ValueError(f"JITFunction {field_name} must not be empty")
    return decoded


def _parse_jitfunction_token(token: str) -> Tuple[str, str, str]:
    """Parse @jit:<hex-path>:<hex-name>:<source-fingerprint>."""
    fields = token.split(":")
    if len(fields) != 4 or fields[0] != "@jit":
        raise ValueError(f"invalid JITFunction token: {token}")

    path = _decode_jitfunction_field(fields[1], "module path")
    name = _decode_jitfunction_field(fields[2], "function name")
    fingerprint = fields[3]
    if re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None:
        raise ValueError("JITFunction fingerprint must be 16 lowercase hex characters")
    if not Path(path).is_absolute():
        raise ValueError("JITFunction module path must be absolute")
    return path, name, fingerprint


def _unwrap_jitfunction(value, description: str):
    seen = set()
    current = value
    for _ in range(32):
        if isinstance(current, triton.runtime.JITFunction):
            return current
        identity = id(current)
        if identity in seen:
            raise TypeError(f"cycle while unwrapping {description}")
        seen.add(identity)
        if not hasattr(current, "fn"):
            raise TypeError(f"{description} does not resolve to a Triton JITFunction")
        current = current.fn
    raise TypeError(f"wrapper depth exceeded while unwrapping {description}")


def _load_jitfunction(path: str, name: str, fingerprint: str):
    """Validate a source snapshot and resolve one Triton JITFunction."""
    source_path = Path(path)
    source = source_path.read_bytes()
    actual_fingerprint = _fnv1a64(source)
    if actual_fingerprint != fingerprint:
        raise ValueError(
            "JITFunction source changed after the C++ argument was constructed: "
            f"expected {fingerprint}, got {actual_fingerprint}"
        )

    module_name = (
        f"_triton_jit_callee_{fingerprint}_{_fnv1a64(str(source_path).encode('utf-8'))}"
    )
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load JITFunction module: {source_path}")
    module = importlib.util.module_from_spec(spec)
    linecache_key = str(source_path)
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    previous_linecache = linecache.cache.get(linecache_key, missing)
    sys.modules[module_name] = module
    try:
        # Execute the exact bytes whose fingerprint was validated above.
        # SourceFileLoader may otherwise reuse a same-timestamp, same-size .pyc
        # after an in-place source edit, disconnecting the cache key from code.
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source).readline)
        source_text = source.decode(encoding)
        linecache.cache[linecache_key] = (
            len(source),
            None,
            source_text.splitlines(keepends=True),
            linecache_key,
        )
        code = compile(source, linecache_key, "exec")
        exec(code, module.__dict__)
        value = getattr(module, name)
        return _unwrap_jitfunction(value, f"{source_path}:{name}")
    except Exception:
        if previous_module is missing:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if previous_linecache is missing:
            linecache.cache.pop(linecache_key, None)
        else:
            linecache.cache[linecache_key] = previous_linecache
        raise


def _resolve_jitfunction_constants(signature: List[str]) -> dict:
    resolved = {}
    for index, token in enumerate(signature):
        if token.startswith("@jit:"):
            path, name, fingerprint = _parse_jitfunction_token(token)
            resolved[index] = _load_jitfunction(path, name, fingerprint)
    return resolved


def generate_arg_layout(
    signature: List[str], constexpr_indices: List[int]
) -> List[dict]:
    """Generate arg_layout metadata from signature.

    Args:
        signature: List of signature strings like ["*fp32:16", "*fp32", "i64", "1024"]
        constexpr_indices: List of indices that are constexpr

    Returns:
        List of arg_layout dicts for runtime arguments only (excluding constexpr)
    """
    arg_layout = []

    for i, sig in enumerate(signature):
        # Check if this is a constexpr by index or by value
        if i in constexpr_indices or sig.startswith("@jit:"):
            continue

        # Try to parse as constexpr value
        sig_clean = sig.split(":")[0].strip()
        try:
            int(sig_clean)
            continue  # It's a constexpr number
        except ValueError:
            pass
        try:
            float(sig_clean)
            continue  # It's a constexpr number
        except ValueError:
            pass

        if sig_clean == "nullopt":
            continue  # Skip nullopt

        parsed = _parse_type_token(sig)
        runtime_types = parsed if isinstance(parsed, tuple) else (parsed,)
        for runtime_type in runtime_types:
            type_info = sig_to_npu_type(runtime_type)
            if type_info["type"] != "constexpr":
                arg_layout.append(type_info)

    return arg_layout


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


def _coerce_opt(v):
    """Coerce a string option value coming from the C++ CompileOptions.extra map into
    the type triton.compile expects (int / float / bool), else keep it as a string.
    C++ can only hand us strings, but options like num_ctas are ints and some are bools."""
    if not isinstance(v, str):
        return v
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _merge_compile_options(
    num_warps: int,
    num_stages: int,
    extra_options: dict = None,
) -> dict:
    """Build backend compile options without allowing extras to override core fields."""
    extra_options = extra_options or {}
    reserved = sorted({"num_warps", "num_stages"}.intersection(extra_options))
    if reserved:
        names = ", ".join(reserved)
        raise ValueError(f"extra_options must not override reserved option(s): {names}")

    opts = {"num_warps": num_warps, "num_stages": num_stages}
    opts.update({_k: _coerce_opt(_v) for _k, _v in extra_options.items()})
    return opts


def _compile_a_kernel(
    fn: triton.runtime.JITFunction,
    signature: str,
    num_warps: int = 4,
    num_stages: int = 3,
    device_id: int = 0,
    extra_options: dict = None,
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

    signature: List[str] = _bracket_aware_split(signature)
    num_args = len(signature)
    assert num_args == len(
        fn.params
    ), f"number of argument mismatch:  Actual({num_args}), Function Definition({len(fn.params)})"

    constants = _resolve_jitfunction_constants(signature)
    for i, token in enumerate(signature):
        if i in constexpr_indices and i not in constants:
            constants[i] = constexpr(token)
    missing_constexpr = [i for i in constexpr_indices if i not in constants]
    assert not missing_constexpr, (
        "number of constexpr mismatch: "
        f"missing parameter indices {missing_constexpr}"
    )

    # signature, no specializations here
    signature_without_spec = {
        i: _parse_type_token(s.split(":")[0])
        for i, s in enumerate(signature)
        if i not in constants
    }

    # specialization: divisibility by 16 or equal to 1
    hints = {
        i: constexpr(s.rsplit(":", 1)[1])
        for i, s in enumerate(signature)
        if i not in constants and ":" in s
    }
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
    elif triton_version.major == 3 and triton_version.minor == 6:
        attrs = {(k,): [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    else:
        raise RuntimeError(
            "Triton may change APIs, we cannot ensure compatibility here now. "
            "You can goto https://github.com/flagos-ai/libtriton_jit to raise an issue "
            "about supporting your triton version. Triton 3.1/3.2/3.3/3.4/3.5/3.6 are supported now."
        )

    # integer 1 in value, but the corresponding ArgType in static signature is not constexpr are added into constants
    for i in equal_to_1:
        constants.update({i: 1})
        # An equal-to-one argument is a compile-time constant, so it must also
        # leave the runtime signature. `signature_without_spec` was built before
        # these indices joined `constants`, and the >=3.3 branch below prefers it,
        # so without this the same parameter is declared twice: as "i32" in the
        # signature and as the constant 1. Triton still folds it -- the generated
        # code is identical -- but the resulting ASTSource, and therefore the
        # compilation cache key, no longer matches the one the native Python
        # runtime builds for the same call, so the two paths never share a
        # compiled kernel.
        signature_without_spec.pop(i, None)
        # Nones in value, but the corresponding ArgType in static signature is not constexpr are added into constants
    for i, v in signature_without_spec.items():
        if v == "nullopt":
            constants[i] = None
            signature_without_spec[i] = "constexpr"

    # GCU backend: downcast fp64 to fp32 in signature to avoid arith.extf
    # (GCU hardware/compiler does not support double precision operations)
    if get_backend() == "GCU":
        for k in signature_without_spec:
            signature_without_spec[k] = _normalize_gcu_signature(signature_without_spec[k])

    if Version("3.0.0") <= triton_version < Version("3.2.0"):
        src = triton.compiler.ASTSource(
            fn=fn,
            constants=constants,
            signature=signature_without_spec,
            attrs=attrs,
        )
    elif Version("3.2.0") <= triton_version < Version("3.3.0"):
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
    # Merge backend-specific compiler switches threaded from C++ CompileOptions.extra
    # (e.g. {"opt_level": "O2"} for MLU). They join `opts` before parse_options so the
    # backend's own option validation sees them.
    opts = _merge_compile_options(num_warps, num_stages, extra_options)

    # STEP3: ast source, target, compile options (backend-specific)
    backend = get_backend()
    if backend in ["NPU", "MUSA", "MTGPU", "MACA", "GCU"]:
        # NPU/MUSA/MTGPU/MACA/GCU: no CUDA device context manager
        target = triton.runtime.driver.active.get_current_target()
        ccinfo = triton.compile(src, target=target, options=opts)
    elif backend in ["MLU"]:
        # torch_mlu only registers torch.mlu when initialization sees CPU.
        # The embedded C++ runtime has already activated MLU, so re-run
        # torch_mlu initialization while temporarily masking the accelerator.
        if not hasattr(torch, "mlu"):
            import sys as _sys

            _orig_cur = torch.accelerator.current_accelerator
            torch.accelerator.current_accelerator = lambda *args, **kwargs: None
            try:
                _sys.modules.pop("torch_mlu", None)
                import torch_mlu  # noqa: F401
            finally:
                torch.accelerator.current_accelerator = _orig_cur
            if not hasattr(torch, "mlu"):
                raise RuntimeError(
                    "failed to register torch.mlu for MLU triton driver"
                )

        target = triton.runtime.driver.active.get_current_target()
        opts['is_linear_hint'] = True
        opts['restrict_ptr_hint'] = True
        mlu_backend = triton.compiler.make_backend(target)
        opts = mlu_backend.parse_options(opts)
        ccinfo = triton.compile(src, target=target, options=opts.__dict__)
    else:
        # CUDA / IX: use CUDA device context
        with torch.cuda.device(device_id):
            target = triton.runtime.driver.active.get_current_target()
            ccinfo = triton.compile(src, target=target, options=opts)

    # kernel's hash may not equals the dir in cache
    from triton.runtime.cache import get_cache_manager

    cache_manager = get_cache_manager(ccinfo.hash)
    cache_dir = cache_manager.cache_dir

    # For NPU backend, generate and write arg_layout to metadata JSON
    if backend == "NPU":
        # Generate arg_layout from the original signature
        arg_layout = generate_arg_layout(signature, constexpr_indices)

        # Find the kernel name (usually the function name)
        kernel_name = fn.__name__

        # Look for existing metadata JSON file
        metadata_path = Path(cache_dir) / f"{kernel_name}.json"

        if metadata_path.exists():
            # Read existing metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            # Create new metadata
            metadata = {}

        # Add arg_layout to metadata
        metadata["arg_layout"] = arg_layout

        # Write back
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"[NPU] Generated arg_layout with {len(arg_layout)} runtime args: {arg_layout}"
        )

    # For MTGPU backend: use mtgpu.translate_llvmir_to_mubin() to emit a
    # .mubin when the loaded Triton exposes it. FlagTree 3.6 (mthreads
    # backend) already produces the .mubin inside triton.compile, so this is
    # only a fallback for older mthreads Triton plugin versions.
    elif backend == "MTGPU":
        import shutil

        try:
            from triton._C.libtriton import mtgpu
        except ImportError:
            from triton._C.libtriton import mthreads as mtgpu

        if not hasattr(mtgpu, "translate_llvmir_to_mubin"):
            # FlagTree >= 3.6 / mthreads backend emits the .mubin directly
            # during triton.compile; nothing to do here.
            pass
        else:
            kernel_name = fn.__name__
            llir_path = Path(cache_dir) / f"{kernel_name}.llir"

            if llir_path.exists():
                try:
                    with open(llir_path, "r") as f:
                        llir_content = f.read()

                    # Compilation options (from official Triton MUSA backend compiler.py)
                    opt_option = "-mtgpu-enable-const-calc=1 -mtgpu-tiny-offset-hint=1 -mtgpu-alloc-shared-memory-from-zero=1"

                    # Get capability from target (default to 22 for S5000)
                    capability = getattr(target, "arch", 22)
                    if isinstance(capability, tuple):
                        capability = capability[0] * 10 + capability[1]

                    # Use official compilation function (same as Triton MUSA backend)
                    asm_str, mubin_tmp_path = mtgpu.translate_llvmir_to_mubin(
                        llir_content, opt_option, capability, 0
                    )

                    # Copy mubin to cache directory
                    mubin_path = Path(cache_dir) / f"{kernel_name}.mubin"
                    shutil.copy2(mubin_tmp_path, mubin_path)

                    # Optionally save ASM for debugging
                    if os.environ.get("MUSA_ASM_ENABLE_DUMP", "0") == "1":
                        asm_path = Path(cache_dir) / f"{kernel_name}.asm"
                        with open(asm_path, "w") as f:
                            f.write(asm_str)

                except Exception as e:
                    import sys
                    import traceback

                    sys.stderr.write(
                        f"[MTGPU] Compilation failed: {e}\n{traceback.format_exc()}\n"
                    )

    return cache_dir


def compile_a_kernel(
    source_path,
    fn_name,
    signature: str,
    num_warps: int = 4,
    num_stages: int = 3,
    device_id: int = 0,
    extra_options: dict = None,
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

    return _compile_a_kernel(fn, signature, num_warps, num_stages, device_id, extra_options)


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
