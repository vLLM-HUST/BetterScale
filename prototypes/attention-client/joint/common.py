"""Bounded reference neural wire, distinct from the INT32 oracle ABI."""

import importlib.util
from pathlib import Path

H, M, E, K, LAYERS, CAP = 2048, 768, 128, 8, 2, 32
SHARD = E // 2
ALIGN = 2 * 1024**2
W13_BYTES = E * H * (2 * M) * 2
W2_BYTES = E * M * H * 2
LAYER_BYTES = W13_BYTES + W2_BYTES
WEIGHT_BYTES = LAYERS * LAYER_BYTES
INPUT_OFFSET = WEIGHT_BYTES
ALLOCATION_BYTES = ((WEIGHT_BYTES + CAP * H * 2 + ALIGN - 1) // ALIGN) * ALIGN
RESULT_BYTES = CAP * K * H * 2


def acl_api():
    path = Path(
        "/workspace/my-ascend-workspace/stateharbor/src/livemodule/arch/ascend/request_parallel/dsv4/prefix_copy_acl.py"
    )
    spec = importlib.util.spec_from_file_location("joint_acl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrefixCopyACL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")


def recv(pipe, timeout=300):
    if not pipe.poll(timeout):
        raise TimeoutError("neural joint control timeout")
    return pipe.recv()
