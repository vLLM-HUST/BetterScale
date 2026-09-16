"""Two physical source NPUs with bulk same/different-layer work, one server.

All inputs are published before server execution: this measures a genuinely
backlogged admission opportunity, not the probability of catching late arrivals.
Timing is device coordinator pull-to-return; client reduction/attention and host
rendezvous are excluded. IPC capabilities never enter result files.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time


def geometry():
    return json.loads(Path(os.environ["PERSISTENT_BUILD"], "geometry.json").read_text())


def cases():
    work = []
    for n in (16, 64, 128, 256, 512):
        if 2 * n > geometry()["rows"]:
            continue
        work.extend(
            [
                (f"solo{2*n}", 2 * n, 0, 0),
                (f"same{n}", n, n, 0),
                (f"different{n}", n, n, 1),
            ]
        )
    n = geometry()["rows"]
    work.extend([(f"same{n}", n, n, 0), (f"different{n}", n, n, 1)])
    if os.environ.get("BULK_SCENARIO") == "priority":
        n = geometry()["rows"]
        work = [
            ("priority_decode_first", n, 32, 0),
            ("priority_decode_other_layer", n, 32, 1),
            ("priority_promoted_prefill", n, 32, 0),
            ("priority_stale_promotion", n, 32, 0),
            ("priority_future_promotion", n, 32, 0),
            ("prefill_pair", n, n, 0),
        ]
    for repeat in range(6):
        for case in work if repeat % 2 == 0 else reversed(work):
            yield repeat, case


def sample_specs(n0, n1, c):
    n = (n0, n1)[c]
    offset = n0 if c else 0
    entries = [
        (
            r * 10 + k,
            ((r + offset) * 13 + k * 53) % 512,
            int(c == 1 or (n1 == 0 and r >= n0 // 2)),
        )
        for r in range(n)
        for k in range(10)
        if ((r + offset) * 13 + k * 53) % 512 < 128
    ]
    return [entries[0], entries[-1]] if entries else []


def recv(pipe):
    if not pipe.poll(180):
        raise TimeoutError("bulk control")
    return pipe.recv()


def worker(role, device, links, out):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    import persistent_service

    expected = Path(os.environ["PERSISTENT_BUILD"], "source", "persistent_service.py")
    assert (
        Path(persistent_service.__file__).resolve() == expected.resolve()
    ), "Admission shadowed the matched bulk runtime"
    g = geometry()
    import torch
    import torch_npu
    from common import acl_api

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format = True
    api = acl_api()
    stream = torch.npu.current_stream().npu_stream
    source_bytes, output_bytes = g["source_bytes"], g["output_bytes"]
    payload = g["payload_words"]
    if role < 2:
        pipe = links[0]
        pipe.send(api.pid())
        server_pid = recv(pipe)
        local = api.allocate_staging(source_bytes)
        key = api.export(local, source_bytes, (server_pid,))
        pipe.send(key)
        frame = torch.zeros(source_bytes // 4, dtype=torch.int32, device="npu")
        for repeat, (label, n0, n1, layer1) in cases():
            assert recv(pipe) == label
            n = (n0, n1)[role]
            layer = layer1 if role else 0
            frame.zero_()
            if n:
                priority = int(
                    label == "prefill_pair"
                    or (label.startswith("priority_") and role == 0)
                )
                frame[8:16] = torch.tensor(
                    [1, layer, n, priority, 0, 0, 0, 0], device="npu"
                )
                if role == 0:
                    frame[16] = {
                        "priority_promoted_prefill": 1,
                        "priority_stale_promotion": -1,
                        "priority_future_promotion": 2,
                    }.get(label, 0)
                ids = (
                    (
                        (torch.arange(n, device="npu")[:, None] + (n0 if role else 0))
                        * 13
                        + torch.arange(10, device="npu")[None, :] * 53
                    )
                    % 512
                ).int()
                frame[64 : 64 + n * 10].copy_(ids.flatten())
                x = torch.full(
                    (n, 2048), 0.01 * (role + 1), dtype=torch.bfloat16, device="npu"
                )
                if role == 0 and n1 == 0:
                    x[n0 // 2 :].fill_(0.02)
                frame[payload : payload + n * 1024].copy_(x.view(torch.int32).flatten())
                frame[0] = 1
            else:
                frame[0] = -1  # explicit inactive source EOF
            torch.npu.synchronize()
            api.copy(stream, local, frame.data_ptr(), source_bytes)
            torch.npu.synchronize()
            pipe.send("published")
            assert recv(pipe) == "finish"
            frame[0] = -2 if n else -1
            torch.npu.synchronize()
            api.copy(stream, local, frame.data_ptr(), 32)
            torch.npu.synchronize()
            pipe.send("eof")
            assert recv(pipe) == "consumed"
            # Prior server is stopped before the next frame can be reused.
        assert recv(pipe) == "unmapped"
        api.close_mapping(key)
        api.free_staging(local)
        pipe.send("released")
        return

    from next_weights import weights
    from persistent_service import PersistentEngine

    for p in links:
        client_pid = recv(p)
        p.send(api.pid())
    keys = [recv(p) for p in links]
    sources = [api.import_memory(k) for k in keys]
    outputs = [api.allocate_staging(output_bytes) for _ in range(2)]
    zeros = torch.zeros(output_bytes // 4, dtype=torch.int32, device="npu")
    catalog = weights(0, layers=[0, 1])
    table = torch.tensor(
        [[u.data_ptr(), d.data_ptr()] for u, d in catalog],
        dtype=torch.int64,
        device="npu",
    )
    # Small selected-route oracle, outside the device timing envelope.
    reference = {}
    for layer, (up, down) in enumerate(catalog):
        u = torch_npu.npu_format_cast(up, 2)
        d = torch_npu.npu_format_cast(down, 2)
        wanted = {
            (scale, expert)
            for repeat, (_, n0, n1, layer1) in cases()
            if repeat == 0
            for c in range(2)
            if (layer1 if c else 0) == layer
            for _, expert, scale in sample_specs(n0, n1, c)
        }
        for scale, expert in wanted:
            x = torch.full(
                (1, 2048), 0.01 * (scale + 1), dtype=torch.bfloat16, device="npu"
            )
            reference[layer, scale, expert] = (
                torch_npu.npu_swiglu(x @ u[expert]) @ d[expert]
            ).clone()
        del u, d
    results = []
    for repeat, (label, n0, n1, layer1) in cases():
        for p in links:
            p.send(label)
        assert all(recv(p) == "published" for p in links)
        for output in outputs:
            api.copy(stream, output, zeros.data_ptr(), output_bytes)
        torch.npu.synchronize()
        engine = PersistentEngine(
            sources,
            outputs,
            *catalog[0],
            0,
            tasks=1,
            open_service=True,
            weight_table=table,
        )
        torch.npu.synchronize()
        header = torch.zeros(8, dtype=torch.int32, device="npu")
        torch.npu.synchronize()
        for c, src in enumerate(sources):
            api.copy(stream, header.data_ptr(), src, 32)
            assert header.cpu()[0].item() == (1 if (n0, n1)[c] else -1), (
                label,
                c,
                "unpublished source",
            )
        # Prepare all observer storage before resident AIV/AIC graphs start.
        # Do not enqueue a fresh zero kernel behind resident compute.
        done = torch.empty(8, dtype=torch.int32, device="npu")
        torch.npu.synchronize()
        print("begin", repeat, label, flush=True)
        engine.replay()
        # Output DONE is read on an independent copy stream, not a polling AIV.
        deadline = time.monotonic() + 30
        for c, n in enumerate((n0, n1)):
            if not n:
                continue
            while True:
                api.copy(stream, done.data_ptr(), outputs[c], 32)
                if done.cpu()[0].item() == 1:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(label)
                if time.monotonic() > deadline - 28:
                    state = engine.control.cpu().tolist()
                    raise RuntimeError(
                        (
                            label,
                            {
                                i: state[i][:4]
                                for i in (0, 1, 2, 43, 85, 86, 87, 88, 89, 90, 91, 92)
                            },
                        )
                    )
                time.sleep(0.0001)
        for p in links:
            p.send("finish")
        assert all(recv(p) == "eof" for p in links)
        torch.npu.synchronize()
        receipt = engine.finish()
        assert receipt["completed_counts"] == [1, int(n1 > 0)]
        assert receipt["waves"] == (
            2 if layer1 or label.startswith("priority_") else 1
        ), (label, receipt["trace"])
        if label.startswith("priority_"):
            admitted = sorted(receipt["trace"], key=lambda row: row[12])
            first = 0 if label == "priority_promoted_prefill" else 1
            assert admitted[0][first] == 1 and admitted[0][1 - first] == 0, (
                label,
                admitted,
            )
            assert admitted[0][11] == int(first == 0), (label, admitted)
        events = receipt["events"]
        span = (max(e[4] for e in events) - min(e[3] for e in events)) / 50
        max_rel = 0
        for c, n in enumerate((n0, n1)):
            if not n:
                continue
            layer = layer1 if c else 0
            for route, expert, scale in sample_specs(n0, n1, c):
                actual = torch.empty((1, 2048), dtype=torch.bfloat16, device="npu")
                api.copy(
                    stream, actual.data_ptr(), outputs[c] + 256 + route * 4096, 4096
                )
                torch.npu.synchronize()
                ref = reference[layer, scale, expert]
                torch.testing.assert_close(actual, ref, rtol=0.02, atol=2e-5)
                rel = (
                    (actual.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-12)
                ).item()
                max_rel = max(max_rel, rel)
        results.append(
            dict(
                case=label,
                repeat=repeat,
                rows=[n0, n1],
                waves=receipt["waves"],
                span_us=span,
                us_per_input_token=span / (n0 + n1),
                selected_route_relative_l2=max_rel,
                trace=receipt["trace"],
                events=events,
            )
        )
        engine.close()
        for p in links:
            p.send("consumed")
    for k, p in zip(keys, links):
        api.close_mapping(k)
        p.send("unmapped")
    assert all(recv(p) == "released" for p in links)
    for output in outputs:
        api.free_staging(output)
    summary = {
        label: dict(
            median_us=statistics.median(
                x["span_us"] for x in results if x["case"] == label and x["repeat"] >= 2
            ),
            median_us_per_token=statistics.median(
                x["us_per_input_token"]
                for x in results
                if x["case"] == label and x["repeat"] >= 2
            ),
        )
        for label, _, _, _ in [case for repeat, case in cases() if repeat == 0]
    }
    Path(out, "result.json").write_text(
        json.dumps(
            dict(scope=__doc__.strip(), summary=summary, samples=results), indent=2
        )
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--devices", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    devices = a.devices.split(",")
    assert len(set(devices)) == 3
    a.out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    pipes = [ctx.Pipe() for _ in range(2)]
    children = []
    try:
        for role in range(3):
            links = [pipes[role][0]] if role < 2 else [p[1] for p in pipes]
            child = ctx.Process(target=worker, args=(role, devices[role], links, a.out))
            child.start()
            children.append(child)
        deadline = time.monotonic() + 300
        while any(p.is_alive() for p in children):
            assert all(p.exitcode in (None, 0) for p in children), [
                p.exitcode for p in children
            ]
            if time.monotonic() > deadline:
                raise TimeoutError("bulk supervisor")
            time.sleep(0.2)
        assert all(p.exitcode == 0 for p in children)
    finally:
        for p in children:
            if p.is_alive():
                p.terminate()
        for p in children:
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join()
