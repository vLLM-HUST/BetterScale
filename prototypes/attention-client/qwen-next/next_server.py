"""Independent expert role: own weights, local IPC discovery, device EOF drain."""

import argparse
import ctypes as C
import json
import os
import threading
from pathlib import Path
import torch
import torch_npu
from common import acl_api, ALIGN
from control import Channel, listen
from settings import CONTRACT, LAYERS
from next_weights import weights
from persistent_service import PersistentEngine
from profile_capture import start as start_profile, stop as stop_profile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--owner", type=int, choices=(0, 1, 2, 3), required=True)
    a = p.parse_args()
    torch.npu.set_device(0)
    # Persistent kernels outlive individual model calls and cold compilation.
    acl = C.CDLL("/usr/local/Ascend/cann-9.0.1/lib64/libascendcl.so")
    acl.aclrtSetOpExecuteTimeOut.argtypes = [C.c_uint32]
    assert acl.aclrtSetOpExecuteTimeOut(1200) == 0
    torch.set_num_threads(2)
    torch_npu.npu.config.allow_internal_format = True
    print("device initialized", flush=True)
    api = acl_api()
    catalog = weights(a.owner)
    up, down = catalog[0]
    weight_table = torch.tensor(
        [[u.data_ptr(), d.data_ptr()] for u, d in catalog],
        dtype=torch.int64,
        device="npu",
    )
    print("expert weights ready", flush=True)
    socket_path = a.directory / f"expert{a.owner}.sock"
    listener = listen(socket_path)
    # Full checkpoint preparation and cold native-op compilation can outlast
    # the small role fixture's 180s rendezvous. Keep startup bounded by the
    # same 600s budget as the Next channels (supervisor remains 1200s).
    listener.settimeout(600)
    clients = {}
    for _ in range(2):
        ch = Channel(listener.accept()[0])
        ch.sock.settimeout(600)
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
        weight_table=weight_table,
    )
    profiler = start_profile(f"expert{a.owner}")
    engine.replay()
    stop_observer = threading.Event()
    observer = None
    if os.environ.get("NEXT_DIAGNOSTICS") == "1":

        def observe():
            torch.npu.set_device(0)
            while not stop_observer.wait(1):
                state = engine.control.cpu().tolist()
                lines = (0, 1, 2, 43, 44, 85, 86, 87, 88, 89, 90, 91, 92)
                print("control", {i: state[i][:4] for i in lines}, flush=True)

        observer = threading.Thread(target=observe)
        observer.start()

    try:
        for c in ordered:
            c["ch"].send(dict(op="ready"))
        expected = [c["ch"].expect("drain")["generation"] for c in ordered]
        receipt = engine.finish()
    finally:
        stop_observer.set()
        if observer is not None:
            observer.join()
    stop_profile(profiler)
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
                diagnostics=(
                    receipt if os.environ.get("NEXT_WIRE_CONCURRENCY") == "1" else None
                ),
                completed=expected,
                admitted_promotions=receipt["admitted_promotions"],
                waves=receipt["waves"],
                rolling_trace=receipt["rolling_trace"],
                role_weight_bytes=sum(
                    u.numel() * 2 + d.numel() * 2 for u, d in catalog
                ),
                loaded_without_client=True,
                device_eof=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
