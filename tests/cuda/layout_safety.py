"""Run each case in a fresh process under compute-sanitizer."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import torch
import triton
from vector_kernel import layout_kernel

sys.path.insert(0, sys.argv[1])
resolver = importlib.import_module("tuned_resolver")

spec = importlib.util.spec_from_file_location(
    "fixture", Path(__file__).parents[1] / "test_tuned_resolver.py"
)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
case = sys.argv[2]
base = torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
if case == "contiguous":
    x = base
elif case == "transpose":
    x = base.T
elif case == "slice":
    x = base[::2, ::2]
elif case == "offset":
    x = base[1:6, 2:7]
elif case in ("kwargs", "reset", "restore"):
    x = base
elif case == "alias":
    args, kw = resolver._clone_arguments((base[1:6, ::2],), {"other": base.T})
    assert args[0].storage_offset() == 64
    assert args[0].untyped_storage()._cdata == kw["other"].untyped_storage()._cdata
    args[0].add_(1)
    torch.cuda.synchronize()
    assert torch.equal(
        base, torch.arange(4096, device="cuda", dtype=torch.float32).reshape(64, 64)
    )
    assert kw["other"][0, 1].item() == 65
    print(json.dumps({"case": case, "pass": True}))
    raise SystemExit(0)
else:
    raise ValueError(case)
y = torch.full(x.shape, -17.0, device="cuda")
original = x.clone()
original_y = y.clone()
names = ["x", "y", "M", "N", "sx0", "sx1", "sy0", "sy1", "BLOCK"]


def identity(x, y, M, N, sx0, sx1, sy0, sy1, BLOCK):
    pass


t = f.LibTuner(
    "layout",
    f.JITFunction(identity, names),
    ["M", "N"],
    None,
    [f.Config({"BLOCK": 16}), f.Config({"BLOCK": 32})],
    [1.0, 2.0],
)
lib = f.fake_libentry("flag_fake")
observed = []


def bench(self, *args, config, **kwargs):
    named = {**dict(zip(names, args)), **kwargs}
    expected = named["x"].clone() * 2
    # Exercise the tuner's reset/restore hooks on exactly the arguments _bench sees.
    if case in ("reset", "restore"):
        self.pre_hook(named)
    layout_kernel[(triton.cdiv(named["M"] * named["N"], config.kwargs["BLOCK"]),)](
        named["x"],
        named["y"],
        named["M"],
        named["N"],
        named["sx0"],
        named["sx1"],
        named["sy0"],
        named["sy1"],
        BLOCK=config.kwargs["BLOCK"],
    )
    torch.cuda.synchronize()
    observed.append(bool(torch.equal(named["y"], expected)))
    if case == "restore":
        self.post_hook(named, None)
    return (1.0, 1.0, 1.0)


t._bench = types.MethodType(bench, t)
if case == "reset":
    t.reset_to_zero = ["y"]
    t.pre_hook = lambda named, reset_only=False: named["y"].zero_()
if case == "restore":
    t.restore_value = ["y"]
    t.restore_copies = {}

    def pre(named, reset_only=False):
        if not reset_only:
            t.restore_copies = {"y": named["y"].clone()}

    def post(named, exception=None):
        named["y"].copy_(t.restore_copies["y"])

    t.pre_hook = pre
    t.post_hook = post
args = (x, y, *x.shape, *x.stride(), *y.stride())
kwargs = {}
if case == "kwargs":
    args = ()
    kwargs = dict(zip(names, args or (x, y, *x.shape, *x.stride(), *y.stride())))
answer = resolver.resolve_with_tuner(t, lib, args, kwargs, grid=(1,))
torch.cuda.synchronize()
ok = (
    "unsupported" not in answer
    and len(observed) == 2
    and all(observed)
    and torch.equal(x, original)
    and torch.equal(y, original_y)
)
print(
    json.dumps(
        {
            "case": case,
            "pass": bool(ok),
            "candidate_results": observed,
            "input_unchanged": bool(torch.equal(x, original)),
            "output_unchanged": bool(torch.equal(y, original_y)),
        }
    ),
    flush=True,
)
raise SystemExit(0 if ok else 1)
