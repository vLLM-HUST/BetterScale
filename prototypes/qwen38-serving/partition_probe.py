"""Alternate distinct exact partition graphs within one HTTP service; state shadows only."""

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from service_probe import request

root = Path(os.environ["CAPSULE"])
width = 2048
url = "http://127.0.0.1:32181"
prompt = json.loads((root / "prompt.json").read_text())["prompt_token_ids"]
prompts = {n: (prompt * 4)[:n] for n in [512, 514, 1022, 1024, 1536, 2048]}
command = [
    sys.executable,
    "-m",
    "vllm.entrypoints.cli.main",
    "serve",
    "/models/vllm-ascend-models/Qwen3.8-27B",
    "--host",
    "127.0.0.1",
    "--port",
    "32181",
    "--served-model-name",
    "qwen27",
    "--tensor-parallel-size",
    "2",
    "--distributed-executor-backend",
    "mp",
    "--worker-cls",
    "mixed_full_worker.Worker",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "4096",
    "--max-num-seqs",
    "8",
    "--max-num-batched-tokens",
    str(width),
    "--kv-cache-memory-bytes",
    str(1024**3),
    "--seed",
    "17",
    "--no-enable-prefix-caching",
    "--async-scheduling",
    "--shutdown-timeout",
    "60",
    "--additional-config",
    '{"enable_cpu_binding":false}',
    "--limit-mm-per-prompt",
    '{"image":0,"video":0}',
    "--compilation-config",
    json.dumps(
        dict(
            cudagraph_mode="FULL",
            cudagraph_capture_sizes=[1, 2, 4, 8, 513, 517, 2048],
            max_cudagraph_capture_size=2048,
        )
    ),
]

receipt = dict(status="STARTED", command=command, rows=[])
path = root / "receipt.json"
path.write_text(json.dumps(receipt, indent=2))
log = (root / "server.log").open("w")
server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)


def rpc(method, *args):
    req = urllib.request.Request(
        url + "/collective_rpc",
        data=json.dumps(dict(method=method, args=list(args), timeout=120)).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=150) as r:
        return json.load(r)


try:
    deadline = time.monotonic() + 720
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"server exited {server.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("server readiness")
        time.sleep(2)
    import threading

    def joined_cohort(partition):
        decodes = next(i for i, n in enumerate(partition) if n > 1)
        lengths = (
            [partition[decodes], 1536]
            if len(partition) - decodes == 2
            else [partition[-1]]
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(partition) + 1
        ) as pool:
            ready = [threading.Event() for _ in range(decodes)]
            ongoing = [
                pool.submit(request, url, prompts[512], 96, on_first_content=e.set)
                for e in ready
            ]
            assert all(e.wait(120) for e in ready)
            hold = pool.submit(rpc, "hold_for_mixed_inputs")
            time.sleep(0.1)
            joined = []
            for n in lengths:
                joined.append(pool.submit(request, url, prompts[n], 4))
                time.sleep(0.02)
            hold.result()
            return dict(
                ongoing=[f.result() for f in ongoing],
                joined=[f.result() for f in joined],
            )

    order = [
        (2048,),
        (1, 1, 1024, 1022),
        (1, 1, 1022, 1024),
        (1, 512),
        (1, 1, 1, 514),
        (2048,),
        (1, 1, 1024, 1022),
        (1, 1, 1022, 1024),
    ]
    for i, partition in enumerate(order):
        rpc("arm_mixed_shadow", 1, list(partition))
        row = (
            request(url, prompts[2048], 4)
            if len(partition) == 1
            else joined_cohort(partition)
        )
        shadow = rpc("mixed_shadow_result")
        assert all(x["passed"] for x in shadow["results"]), shadow
        for rank in shadow["results"]:
            banks = list(rank["resource_banks"].values())
            assert len(banks) == 5 and len({b["bank_id"] for b in banks}) == 5
            assert all(b["handles"] == b["events"] == 16 for b in banks)
        receipt["rows"].append(dict(partition=partition, requests=row, shadow=shadow))
        path.write_text(json.dumps(receipt, indent=2))
    receipt["status"] = "PASS"
except BaseException as exc:
    receipt.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    raise
finally:
    if server.poll() is None:
        server.send_signal(signal.SIGINT)
    try:
        server.wait(timeout=90)
    except subprocess.TimeoutExpired:
        receipt["shutdown_timeout"] = True
        server.kill()
        server.wait()
    receipt["server_exit_code"] = server.returncode
    path.write_text(json.dumps(receipt, indent=2))
    log.close()

if receipt["status"] != "PASS":
    raise SystemExit(1)
