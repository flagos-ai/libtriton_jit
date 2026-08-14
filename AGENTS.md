<!--
 Copyright 2026 FlagOS Contributors

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 -->

# Repository Guidelines

## Project Structure & Module Organization
- `include/`: public C++ headers for the Triton JIT runtime.
- `src/`: core C++ implementation (runtime, utilities).
- `operators/`: operator implementations and backend-specific code; tests are built per operator.
- `scripts/`: Python helpers used by the runtime (e.g., compilation scripts).
- `tests/`: test runner and configs (`tests/run_all_tests.py`, `tests/configs/`).
- `examples/`, `experiments/`: reference code and experiments (not always built).
- `cmake/`: CMake modules; `build/` is the default build output.
- `assets/`: diagrams and documentation images.

## Build, Test, and Development Commands
- `pip install "torch>=2.5" "triton>=3.1.0,<3.4.0" "cmake" "ninja" "packaging" "pybind11" "numpy"`  
  Installs Python dependencies required for the embedded compiler/runtime.
- `cmake -S . -B build/ -DPython_ROOT="$(which python)/../.."`  
  Configures the project and points CMake at your active Python.
- `cmake --build build/ --parallel`  
  Builds the library and operator tests.
- `cmake --install build/`  
  Installs headers/libs and CMake config (optional).
- `python tests/run_all_tests.py --backend CUDA --build-dir build`  
  Runs all operator tests and prints a summary.

## Coding Style & Naming Conventions
- C++: 2-space indentation, C++20, namespaces in `triton_jit`, functions/variables use `snake_case`.
- Python: 4-space indentation, standard library first; keep scripts importable (no relative imports).
- No enforced formatter/linter in the repo; match existing file style.

## Testing Guidelines
- Tests are compiled executables produced under `build/operators/<category>/<op>/test_<op>`.
- Use `tests/run_all_tests.py` to run the full suite or filter by `--category` or `--operator`.
- Example: `python tests/run_all_tests.py --config tests/configs/quick.json --backend CUDA`.

## Commit & Pull Request Guidelines
- Commit subjects are short and descriptive; existing history uses plain phrases and occasional tags like `[FIX]`.
- Preferred pattern: `<scope>: <action>` or `[TAG] <action>` (e.g., `operators: add topk test`).
- PRs should include: purpose, backend tested (e.g., CUDA/MUSA/NPU/IX), test command output, and linked issues.

## Configuration & Logging
- Backend selection is set at configure time: `-DBACKEND=CUDA|MUSA|NPU|IX`.
- Runtime logging uses torch logging; set `TORCH_CPP_LOG_LEVEL=INFO` to enable.
