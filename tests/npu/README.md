# Ascend runtime regressions

Configure on an Ascend machine with the CANN environment loaded, Torch,
`torch_npu`, and the Ascend Triton/FlagTree backend installed:

```sh
export TRITON_JIT_BACKEND=NPU TORCH_DEVICE_BACKEND_AUTOLOAD=0
cmake -S . -B build-npu -DBACKEND=NPU -DBUILD_TESTING=ON \
  -DTRITON_JIT_BUILD_OPERATORS=OFF -DTRITON_JIT_BUILD_TORCH_RESOLVER=ON \
  -DTRITON_JIT_TEST_NPU=ON
cmake --build build-npu --parallel
ctest --test-dir build-npu --output-on-failure
```

The two `npu_auto_blockify_0` / `npu_auto_blockify_1` tests use the first visible
NPU, serialize access through a CTest resource lock, and keep separate Triton
caches and JSON results under `build-npu/tests/npu-auto-blockify-{0,1}`. All data
is generated in the test. Each mode checks:

- FP16/BF16 GEMV against CPU and native Triton, including grids around and above
  the 910B's 40 AIV-core boundary (32, 40, 41, 64, 128, 256 blocks).
- Atomic per-program counters below, at, and above the physical core count
  (32, 40, 80, 120, 160 on 910B) to detect both missing and duplicate execution.
- Explicit `enable_auto_blockify` options overriding the opposite environment.
- Warm frozen launches after changing the environment, preserving the compiled
  mode; `prepare()` does not execute and cold frozen launches fail.
- Distinct on/off disk cache artifacts and repeat compilation cache hits.
- Plain Triton online autotuning, caller-output isolation, frozen config hits,
  C++ execution, and offline table export/load round trips.

`backend_lazy_init_python` runs without accelerator dependencies and checks
module imports, `--help`, optional table helpers, and required dependency errors.

Launch policy is taken from the compiled JSON's boolean `enable_auto_blockify`.
The compiler helper resolves the environment default before compilation, so the
mode becomes part of Triton's options cache key. Legacy artifacts with missing
or null mode metadata must be recompiled; the runtime reports this explicitly.

The tested FlagTree 0.6.2a1+ascend3.5 Linalg pipeline reads the environment
when emitting the loop, even when the native launcher honors a conflicting
per-kernel option. For such overrides the compiler helper starts a child Python
process with a consistent environment and the caller's visible device index;
it does not change the parent process environment. This adds process startup
cost on a cold conflicting override, but warm C++ launches use the cached kernel.
The effective environment also participates in the disk cache key, separating
these artifacts from ambiguous native/legacy artifacts. Explicit-override tests
compare C++ against exact CPU counts and a native reference compiled with a
consistent environment; the C++ compilation receives the opposite environment.

## Known upstream atomic tail failure

FlagTree 0.6.2a1+ascend3.5 also executes spurious atomics at program ID 0 when
an auto-blockified grid spans a non-integral number of physical-core waves.
Both native Triton and the C++ path reproduce it: native grids 41, 64, and 128
produce counts[0] of 40, 17, and 33 instead of 1 on this 40-AIV-core device.
This is independent of libtriton_jit's launch clamp. The acceptance counters use
complete waves above the core count, which still fail with either the old
unconditional clamp (missing programs) or the unconditional full-grid launch
(duplicate programs with auto-blockify). GEMV tests retain the boundary grids.

Keep auto-blockify off for atomic kernels with such tail grids on this compiler.
The acceptance suite additionally checks the per-kernel opt-out for grids
41, 64, and 128 while the process environment is enabled:
`CompileOptions.extra["enable_auto_blockify"] = "false"`. This exercises the
isolated compiler workaround and verifies every counter against exact CPU
counts. It avoids the compiler defect; it does not repair the upstream pass.
The standalone reproducer below intentionally exits nonzero on affected
compilers; it is retained for an upstream report, outside the passing CTest
acceptance suite. A passing acceptance suite does not mean this upstream defect
is fixed:

```sh
TRITON_CACHE_DIR=/tmp/triton-atomic-tail-repro \
  python tests/npu/reproduce_atomic_tail.py
```
