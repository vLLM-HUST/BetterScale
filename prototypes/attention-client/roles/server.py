"""Independent expert role: own weights, local IPC discovery, device EOF drain."""

import argparse
import json
from pathlib import Path
import torch
import torch_npu
from common import acl_api, ALIGN
from control import CONTRACT, Channel, listen
from persistent_service import PersistentEngine


def weights(owner, nz=True):
    # Full per-layer geometry. Local generation is independent of client model
    # construction, and does not require an attention-side weight replica.
    torch.manual_seed(812 + owner)
    up = torch.empty((128, 2048, 1536), dtype=torch.bfloat16, device="npu")
    down = torch.empty((128, 768, 2048), dtype=torch.bfloat16, device="npu")
    for tensor in (up, down):
        for expert in tensor:
            expert.normal_(0, 0.01)
    if nz:
        return torch_npu.npu_format_cast(up, 29), torch_npu.npu_format_cast(down, 29)
    return up, down


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--owner", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch_npu.npu.config.allow_internal_format = True
    print("device initialized", flush=True)
    api = acl_api()
    up, down = weights(a.owner)
    print("expert weights ready", flush=True)
    socket_path = a.directory / f"expert{a.owner}.sock"
    listener = listen(socket_path)
    clients = {}
    for _ in range(2):
        ch = Channel(listener.accept()[0])
        hello = ch.expect("hello")
        assert hello["contract"] == CONTRACT
        source = hello["source"]
        assert source in (0, 1) and source not in clients
        output = api.allocate_staging(ALIGN)
        zero = torch.zeros(ALIGN // 4, dtype=torch.int32, device="npu")
        api.copy(torch.npu.current_stream().npu_stream, output, zero.data_ptr(), ALIGN)
        torch.npu.synchronize()
        key = api.export(output, ALIGN, (hello["pid"],))
        ch.send(
            dict(
                op="window",
                owner=a.owner,
                pid=api.pid(),
                key=key.decode(),
                contract=CONTRACT,
            )
        )
        source_key = ch.expect("source")["key"].encode()
        peer = api.import_memory(source_key)
        ch.send(dict(op="registered"))
        clients[source] = dict(
            ch=ch, output=output, key=key, peer=peer, source_key=source_key
        )
    listener.close()
    ordered = [clients[i] for i in range(2)]
    print("clients registered", flush=True)
    engine = PersistentEngine(
        [c["peer"] for c in ordered],
        [c["output"] for c in ordered],
        up,
        down,
        a.owner,
        tasks=32,
        open_service=True,
    )
    engine.replay()
    for c in ordered:
        c["ch"].send(dict(op="ready"))
    expected = [c["ch"].expect("drain")["generation"] for c in ordered]
    receipt = engine.finish()
    assert receipt["completed_counts"] == expected
    for c in ordered:
        c["ch"].send(dict(op="drained"))
    for c in ordered:
        c["ch"].expect("unmapped")
        api.close_mapping(c["source_key"])
        api.close_mapping(c["key"])
        api.free_staging(c["output"])
        c["ch"].send(dict(op="released"))
        c["ch"].close()
    engine.close()
    socket_path.unlink()
    # Never serialize IPC capabilities or raw addresses.
    (a.directory / f"expert{a.owner}.json").write_text(
        json.dumps(
            dict(
                status="pass",
                completed=expected,
                waves=receipt["waves"],
                rolling_trace=receipt["rolling_trace"],
                role_weight_bytes=up.numel() * 2 + down.numel() * 2,
                loaded_without_client=True,
                device_eof=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
