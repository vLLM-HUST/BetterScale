"""Same-process HTTP ABBA probe of FULL versus ungraphed compiled prefill."""

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
width = int(os.environ.get("FIXED_PREFILL_TOKENS", "512"))
url = "http://127.0.0.1:32181"
prompt = json.loads((root / "prompt.json").read_text())["prompt_token_ids"][:width]
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
    "fixed_full_worker.Worker",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "4096",
    "--max-num-seqs",
    "1",
    "--max-num-batched-tokens",
    str(width),
    "--kv-cache-memory-bytes",
    str(6 * 1024**3),
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
            cudagraph_capture_sizes=[1, width],
            max_cudagraph_capture_size=width,
        )
    ),
]
receipt = dict(
    status="STARTED",
    width=width,
    command=command,
    rows=[],
    cohorts=[],
    scope="fixed pure prefill only, same loaded model + metadata adapter, native compiled NONE versus FULL; not general mixed/GDN support",
)
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
            raise RuntimeError(f"server exited{server.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("server readiness")
        time.sleep(2)
    oracle = None
    for mode in ["none", "full", "none", "full"]:
        rpc("set_prefill_full", mode)
        result = request(url, prompt, 1)
        if oracle is None:
            oracle = result["text"]
        assert result["text"] == oracle, (mode, result)
    for i, mode in enumerate(["none", "full", "full", "none"] * 3):
        rpc("set_prefill_full", mode)
        row = request(url, prompt, 1)
        assert row["text"] == oracle, (mode, row)
        receipt["rows"].append(dict(mode=mode, iteration=i, **row))
        path.write_text(json.dumps(receipt, indent=2))
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for mode in ["none", "full", "full", "none"]:
            rpc("set_prefill_full", mode)
            start = time.perf_counter()
            rows = list(pool.map(lambda i: request(url, prompt, 1), range(8)))
            assert all(r["text"] == oracle for r in rows)
            receipt["cohorts"].append(
                dict(mode=mode, elapsed_s=time.perf_counter() - start, rows=rows)
            )
            path.write_text(json.dumps(receipt, indent=2))
    for mode in ["none", "full"]:
        rpc("set_prefill_full", mode)
        rpc("start_prefill_profile", mode)
        for _ in range(4):
            request(url, prompt, 1)
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
