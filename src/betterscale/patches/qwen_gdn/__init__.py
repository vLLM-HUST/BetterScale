"""Owned K-V GDN service execution; opt-in and independent of DSV4."""

import hashlib
import json
import os
from pathlib import Path


def check_library():
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise ValueError("Owned GDN raw launch requires TASK_QUEUE_ENABLE=0")
    path = Path(os.environ.get("BETTERSCALE_GDN_LIBRARY", ""))
    contract = json.loads(Path(__file__).with_name("native.json").read_text())
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != contract["sha256"]
    ):
        raise ValueError(
            "BETTERSCALE_GDN_LIBRARY must name the qualified owned-init KV library"
        )
    host = Path(os.environ.get("BETTERSCALE_GDN_HOST_LIBRARY", ""))
    if (
        not host.is_file()
        or hashlib.sha256(host.read_bytes()).hexdigest() != contract["host_sha256"]
    ):
        raise ValueError(
            "BETTERSCALE_GDN_HOST_LIBRARY must name the qualified graph-pool host adapter"
        )
    return str(path.resolve())


def install():
    from vllm_ascend.ops.gdn_attn_builder import (
        AscendGDNAttentionMetadataBuilder as Builder,
    )

    if getattr(Builder, "_betterscale_elastic", False):
        return
    if getattr(Builder, "_betterscale_qwen_prefill", False):
        raise RuntimeError("Cannot mix native V-K and owned K-V Workers in one process")
    check_library()
    from . import metadata, execution, graphs, publication

    metadata.install()
    execution.install()
    graphs.install()
    publication.install()
    Builder._betterscale_elastic = True
