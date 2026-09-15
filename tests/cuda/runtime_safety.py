import importlib
import json
import sys
import threading
from pathlib import Path

import torch

module_dir, fixture_dir, mode = sys.argv[1:4]
if mode not in ("identity", "gil", "graph"):
    raise ValueError(mode)
sys.path.insert(0, module_dir)
sys.path.insert(0, fixture_dir)
probe = importlib.import_module("tuned_runtime_probe")
fake = importlib.import_module("fake_tuned_resolver")

probe.install(fixture_dir)
if mode == "gil":
    errors = []
    answers = []

    def owner():
        try:
            answers.append(probe.resolve("slow_kernel", "/test/a.py", False, 0))
        except Exception as e:
            errors.append(repr(e))

    t = threading.Thread(target=owner, daemon=True)
    t.start()
    assert fake.entered.wait(5)
    answers.append(probe.resolve("slow_kernel", "/test/a.py", False, 0))
    t.join(5)
    assert (
        not t.is_alive()
        and not errors
        and len(answers) == 2
        and answers[0]["calls"] == 1
        and answers[1]["calls"] == 1
    )
    print(json.dumps({"case": "real_gil_wait", "pass": True}))
    raise SystemExit(0)
if mode == "identity":
    a = probe.resolve("mm_kernel", "/test/a.py", False, 0)
    b = probe.resolve("mm_kernel", "/test/b.py", False, 0)
    again = probe.resolve("mm_kernel", "/test/a.py", True, 0)
    ok = a["source"] == "/test/a.py" and b["source"] == "/test/b.py" and a == again
    print(json.dumps({"case": "builtin_source_identity", "pass": ok, "a": a, "b": b}))
    raise SystemExit(0 if ok else 1)

path = str(Path(__file__).with_name("vector_kernel.py"))
x = torch.arange(1024, device="cuda", dtype=torch.float32)
y = torch.full_like(x, -1)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
# Install a direct compiler-call counter before any preparation.
scripts = Path(fixture_dir).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
compile_module = importlib.import_module("standalone_compile")
original = compile_module.compile_a_kernel
calls = []


def counted(*args, **kwargs):
    calls.append(args)
    return original(*args, **kwargs)


compile_module.compile_a_kernel = counted
with torch.cuda.stream(stream):
    assert probe.prepare(path, x.data_ptr(), y.data_ptr(), x.numel(), 128)
assert torch.equal(y, torch.full_like(y, -1)), "prepare must not launch"
stream.synchronize()
assert len(calls) == 1
probe.launch(path, x.data_ptr(), y.data_ptr(), x.numel(), 128, stream.cuda_stream, True)
stream.synchronize()
assert torch.equal(y, x * 2)
try:
    probe.launch(
        path, x.data_ptr(), y.data_ptr(), x.numel(), 256, stream.cuda_stream, True
    )
    raise AssertionError("cold program was accepted while frozen")
except RuntimeError as e:
    assert "frozen" in str(e) or "ScopedFreeze" in str(e)
assert len(calls) == 1
try:
    probe.resolve("cold_kernel", "/test/cold.py", True, stream.cuda_stream)
    raise AssertionError("cold config was accepted while frozen")
except RuntimeError:
    pass
assert "cold_kernel" not in fake._calls
# Warm both states, capture, replay with changed values, and verify no cold callbacks.
probe.resolve("mm_kernel", "/test/a.py", False, stream.cuda_stream)
before = dict(fake._calls)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    probe.resolve("mm_kernel", "/test/a.py", False, stream.cuda_stream)
    probe.launch(
        path, x.data_ptr(), y.data_ptr(), x.numel(), 128, stream.cuda_stream, False
    )
for i in range(5):
    x.fill_(i + 1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, x * 2)
assert len(calls) == 1 and fake._calls == before
# A real captured stream must refuse misses before resolver/compiler entry.
for kind in ["config", "program"]:
    denied = False
    empty = torch.cuda.CUDAGraph()
    with torch.cuda.graph(empty, stream=stream):
        try:
            if kind == "config":
                probe.resolve(
                    "capture_cold", "/test/missing.py", False, stream.cuda_stream
                )
            else:
                probe.launch(
                    path,
                    x.data_ptr(),
                    y.data_ptr(),
                    x.numel(),
                    512,
                    stream.cuda_stream,
                    False,
                )
        except RuntimeError as error:
            denied = "captur" in str(error)
    assert denied
assert len(calls) == 1 and fake._calls == before
print(
    json.dumps(
        {
            "case": "prepare_freeze_capture",
            "pass": True,
            "compiler_calls": len(calls),
            "replays": 5,
        }
    )
)
