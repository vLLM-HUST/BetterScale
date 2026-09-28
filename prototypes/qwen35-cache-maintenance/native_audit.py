"""Probe-only synchronous byte witness; never enabled by production source."""

import json
import os
from pathlib import Path

import torch
from betterscale.models.qwen35.cache_worker import CacheWorker
from betterscale.live.runtime.host_state import HostStateKey


def install():
    original = CacheWorker._send

    def send(worker, command, error=None):
        if error is None and command["kind"] in ("store", "load"):
            root = worker.runner._live_state_root
            incremental = "pages" in command
            snapshot = (
                None
                if incremental
                else worker.backend._snapshots[HostStateKey(command["key"], 1)]
            )
            span = command["block_size"] // root.capacity.page_tokens
            pages = [b * span + k for b in command["blocks"] for k in range(span)]
            checked, mismatched = [], []
            for name, state in worker.states:
                if not incremental and name not in snapshot.payloads:
                    continue
                ids = [command["seat"]] if state.domain is root.residents else pages
                width = state.physical_blocks_per_logical_block
                rows = [
                    state.tensor.narrow(
                        0, state.leading_physical_blocks + i * width, width
                    ).cpu()
                    for i in ids
                ]
                actual = torch.cat(rows).contiguous().view(torch.uint8)
                if incremental:
                    identities = (
                        ["resident:" + command["key"]]
                        if state.domain is root.residents
                        else command["pages"]
                    )
                    tensors = [
                        worker.page_backend.backend._snapshots[
                            HostStateKey(identity, 1)
                        ]
                        .payloads[name]
                        .tensor
                        for identity in identities
                    ]
                    expected = torch.cat(tensors).contiguous().view(torch.uint8)
                else:
                    expected = (
                        snapshot.payloads[name].tensor.contiguous().view(torch.uint8)
                    )
                checked.append(name)
                if not torch.equal(actual, expected):
                    mismatched.append(name)
            row = dict(
                rank=worker.rank,
                kind=command["kind"],
                checked=len(checked),
                mismatched=mismatched,
                bytes=command.get("_transfer_bytes")
                if incremental
                else snapshot.byte_length,
                operation=command["operation"],
                missing=command.get("missing"),
            )
            (
                Path(os.environ["CAPSULE"])
                / (
                    f"audit-{worker.rank}-{command['kind']}-{command['operation']}.json"
                    if incremental
                    else f"audit-{worker.rank}-{command['kind']}.json"
                )
            ).write_text(json.dumps(row, indent=2) + "\n")
            if mismatched:
                error = repr(row)
        original(worker, command, error)

    CacheWorker._send = send
