"""Four-card remote control with exactly the synthetic EP2 DFC inputs/routes.

Routing is precomputed in BOTH controls; no gate/top-k timing is included.
"""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace


def child(rank, device, links, root):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    # Non-burst controls consume client_graph_us; make their timing contract explicit.
    os.environ.setdefault("DEVICE_SERVICE_TIMING", "1")
    import torch
    import torch_npu
    import device_joint as service
    from common import H, M, E, K

    torch.set_num_threads(2)
    torch.npu.set_device(0)
    if rank >= 2:
        service.serve(rank - 2, links, str(Path(root, f"expert{rank-2}.json")))
        return
    assert service.PARALLEL
    torch.manual_seed(742)
    base_up = (torch.randn(H, 2 * M) * 0.01).to(torch.bfloat16).npu()
    base_down = (torch.randn(M, H) * 0.01).to(torch.bfloat16).npu()
    factors = torch.tensor([0.5 if e % 2 == 0 else 1.0 for e in range(E)], device="npu")
    up = (base_up.unsqueeze(0) * factors[:, None, None]).to(torch.bfloat16)
    down = base_down.unsqueeze(0).repeat(E, 1, 1)
    expert = SimpleNamespace(w13_weight=up, w2_weight=down)
    model = SimpleNamespace(
        layers=[SimpleNamespace(mlp=SimpleNamespace(experts=expert))] * 2
    )

    class RoutedBank(service.ClientBank):
        def body(self):
            if not hasattr(self, "ids"):
                self.ids = torch.zeros(
                    (self.input.shape[0], K), device="npu", dtype=torch.int32
                )
                self.probs = torch.full(
                    (self.input.shape[0], K), 1 / K, device="npu", dtype=torch.bfloat16
                )
            transport = self.transport
            transport.kernels.call(transport.fn, self.config, self.input, self.ids)
            transport.kernels.call(
                transport.collect, self.config, self.input, self.ids, blocks=16
            )
            transport.kernels.call(transport.retire, self.config, self.input, self.ids)
            return torch_npu.npu_moe_token_unpermute(
                self.raw[0].view(-1, H), self.indices, probs=self.probs
            )

    service.ClientBank = RoutedBank
    if os.environ.get("DEVICE_SERVICE_BURST") == "1":
        from burst_control import run

        run(service, model, links, root, rank, base_up, base_down, RoutedBank)
        return
    audit = []
    remote = service.DeviceExperts(model, links, audit)
    from profile_capture import start, stop

    profiler = start(f"attention{rank}")
    results = []
    for pattern in ("balanced", "hot8"):
        for rows in (1, 16, 32):
            torch.manual_seed(180 + rank + rows)
            x = (torch.randn(rows, H) * 0.1).to(torch.bfloat16).npu()
            if pattern == "balanced":
                ids = (torch.arange(rows * 8).reshape(rows, 8) + rank * 8) % 128
            else:
                ids = torch.tensor([0, 1, 2, 3, 64, 65, 66, 67]).expand(rows, 8).clone()
            ids = ids.to(torch.int32).npu()
            partials = []
            for factor in (0.5, 1.0):
                partials.append(
                    torch_npu.npu_swiglu(x @ (base_up * factor).to(torch.bfloat16))
                    @ base_down
                )
            chosen = torch.stack(partials, dim=1)
            expected = (
                chosen.gather(1, (ids % 2).long()[:, :, None].expand(-1, -1, H))
                .float()
                .mean(1)
                .to(torch.bfloat16)
            )
            if (0, rows) not in remote.banks:
                remote.banks[0, rows] = RoutedBank(remote, 0, None, x)
            remote.banks[0, rows].ids.copy_(ids)
            for repeat in range(4):
                handle = remote.submit(
                    "control", 0, None, SimpleNamespace(normalized=x)
                )
                result = None
                while result is None:
                    result = remote.poll(handle)
                    if result is None:
                        time.sleep(0.0001)
                torch.npu.synchronize()
                torch.testing.assert_close(result, expected, rtol=0.02, atol=2e-5)
                relative = (
                    torch.linalg.vector_norm(result.float() - expected.float())
                    / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
                ).item()
                assert relative < 0.01
                results.append(
                    dict(
                        rows_per_source=rows,
                        pattern=pattern,
                        repeat=repeat,
                        client_us=audit[-1]["client_graph_us"],
                        relative_l2=relative,
                        exact=torch.equal(result, expected),
                    )
                )
    remote.close()
    stop(profiler)
    Path(root, f"client{rank}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    devices = [int(d) for d in args.devices.split(",")]
    assert len(devices) == 4
    Path(args.out).mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    links = [[] for _ in devices]
    for c in range(2):
        for s in range(2):
            a, b = ctx.Pipe()
            links[c].append(a)
            links[s + 2].append(b)
    processes = []
    try:
        for rank, device in enumerate(devices):
            p = ctx.Process(target=child, args=(rank, device, links[rank], args.out))
            p.start()
            processes.append(p)
        for group in links:
            for pipe in group:
                pipe.close()
        deadline = time.monotonic() + 300
        while any(p.is_alive() for p in processes):
            assert all(p.exitcode in (None, 0) for p in processes), [
                (p.pid, p.exitcode) for p in processes
            ]
            if time.monotonic() > deadline:
                raise TimeoutError("remote DFC control")
            time.sleep(0.2)
        assert all(p.exitcode == 0 for p in processes)
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(5)
    print("REMOTE_DFC_CONTROL_PASS", flush=True)
