"""Standalone native-Triton reproducer for FlagTree's atomic tail-loop defect."""

import json
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"
# Initialize Torch and the Ascend extension before Triton discovers plugins.
import torch  # noqa: E402
import torch_npu  # noqa: F401, E402
from npu_kernels import counter_kernel  # noqa: E402


def main():
    torch.npu.set_device(0)
    failed = False
    for blocks in (41, 64, 128):
        counts = torch.zeros(blocks, dtype=torch.int32, device="npu:0")
        # Pin the option too, to avoid colliding with an environment-only cache.
        counter_kernel[(blocks,)](
            counts, blocks, num_warps=4, num_stages=1, enable_auto_blockify=True
        )
        torch.npu.synchronize()
        actual = counts.cpu()
        wrong = (actual != 1).nonzero().flatten()
        failed |= bool(wrong.numel())
        print(
            json.dumps(
                {
                    "blocks": blocks,
                    "wrong_indices": wrong.tolist(),
                    "actual": actual[wrong].tolist(),
                }
            )
        )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
