"""Initialize vendor Python backends before discovering Triton plugins."""

import os


def initialize_backend():
    if os.environ.get("TRITON_JIT_BACKEND", "").upper() == "NPU":
        # torch_npu imports torch._dynamo, which must see a fully initialized
        # Triton module rather than the one currently discovering its plugins.
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
