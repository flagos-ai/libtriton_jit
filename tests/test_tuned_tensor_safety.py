"""Storage/layout and interleaved invocation regressions."""
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import torch

scripts = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(scripts))
resolver = importlib.import_module("tuned_resolver")

spec = importlib.util.spec_from_file_location(
    "safety_fixture", Path(__file__).with_name("test_tuned_resolver.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
failures = []


def check(name, condition, **details):
    print(json.dumps({"case": name, "pass": bool(condition), **details}), flush=True)
    if not condition:
        failures.append(name)


def kernel(x, M, stride0, stride1, BLOCK):
    pass


def make(names, keys):
    return fixture.LibTuner(
        "safety",
        fixture.JITFunction(kernel, names),
        keys,
        None,
        [fixture.Config({"BLOCK": 16}), fixture.Config({"BLOCK": 32})],
        [1.0, 2.0],
    ), fixture.fake_libentry("flag_fake")


x = torch.arange(32).reshape(4, 8)[:, ::2]
tuner, lib = make(["x", "M", "stride0", "stride1", "BLOCK"], ["M"])
resolver.resolve_with_tuner(tuner, lib, (x, 4, *x.stride()), {}, grid=(1,))
b = tuner.last_bench_args
check(
    "slice_layout",
    b[0].stride() == x.stride() and torch.equal(b[0], x),
    original=list(x.stride()),
    scratch=list(b[0].stride()),
)
check(
    "slice_bounds",
    sum((n - 1) * s for n, s in zip(b[0].shape, b[2:])) + b[0].storage_offset()
    < b[0].untyped_storage().nbytes() // b[0].element_size(),
)

tuner, lib = make(["M", "x", "BLOCK"], ["M"])
x = torch.zeros(4)


def mutate(self, *args, config, **kwargs):
    kwargs["x"].add_(1)
    return (1.0, 1.0, 1.0)


tuner._bench = types.MethodType(mutate, tuner)
resolver.resolve_with_tuner(tuner, lib, (4,), {"x": x}, grid=(1,))
check("keyword_isolation", torch.equal(x, torch.zeros_like(x)), values=x.tolist())

tuner, lib = make(["x", "M", "BLOCK"], ["M"])
x = torch.ones(4)


def reset(self, named, reset_only=False):
    named["x"].zero_()


tuner.pre_hook = types.MethodType(reset, tuner)
resolver.resolve_with_tuner(tuner, lib, (x, 4), {}, grid=(1,))
check("reset_isolation", torch.equal(x, torch.ones_like(x)), values=x.tolist())

root = torch.arange(48).reshape(6, 8)
a = root[1:5, ::2]
b = root.T
args, kw = resolver._clone_arguments((a,), {"b": b})
check(
    "offset_alias",
    args[0].storage_offset() == a.storage_offset()
    and args[0].stride() == a.stride()
    and kw["b"].stride() == b.stride()
    and args[0].untyped_storage()._cdata == kw["b"].untyped_storage()._cdata
    and args[0].untyped_storage()._cdata != a.untyped_storage()._cdata,
)
args[0][0, 0] = 999
check("alias_preserved", kw["b"][0, 1].item() == 999 and root[1, 0].item() == 8)

tuner, lib = make(["M", "BLOCK"], ["M"])
entered = threading.Event()
second_done = threading.Event()
errors = []


def interleave(self, M, config, **kwargs):
    if M == 7:
        entered.set()
        if not second_done.wait(3):
            raise TimeoutError("second call did not finish")
    if self.nargs["M"] != M:
        raise AssertionError("another invocation replaced nargs")
    return (1.0, 1.0, 1.0)


tuner._bench = types.MethodType(interleave, tuner)


def first():
    try:
        resolver.resolve_with_tuner(tuner, lib, (7,), {}, grid=(1,))
    except Exception as error:
        errors.append(repr(error))


worker = threading.Thread(target=first, daemon=True)
worker.start()
if not entered.wait(3):
    raise TimeoutError("first call did not enter")
resolver.resolve_with_tuner(tuner, lib, (9,), {}, grid=(1,))
second_done.set()
worker.join(4)
check("different_keys", not errors and not worker.is_alive(), errors=errors)

# Model ordinary Python run() owning the original tuner while the resolver uses a copy.
tuner, lib = make(["M", "BLOCK"], ["M"])
tuner.nargs = {"M": 123}
resolver.resolve_with_tuner(tuner, lib, (9,), {}, grid=(1,))
check("native_call_state", tuner.nargs == {"M": 123})
raise SystemExit(bool(failures))
