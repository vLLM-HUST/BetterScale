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
width = 2048
url = "http://127.0.0.1:32181"
prompt = json.loads((root / "prompt.json").read_text())["prompt_token_ids"]
prompts = {n: (prompt + prompt)[:n] for n in [512, 513, 1024, 1536, 2048, 2051]}
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
    "betterscale.qwen_worker.Worker",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "4096",
    "--max-num-seqs",
    "8",
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
            cudagraph_capture_sizes=(
                [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048]
                if os.environ.get("PADDED_PREFILL") == "1"
                else [1, 2, 4, 8, 512, 1024, 1536, 2048]
            ),
            max_cudagraph_capture_size=width,
        )
    ),
]
receipt = dict(
    status="STARTED",
    padded_gdn=os.environ.get("PADDED_PREFILL") == "1",
    width=width,
    command=command,
    rows=[],
    cohorts=[],
    scope="Frozen packaged Qwen Worker acceptance, nonSpec FULL; native mixed fallback; not ABBA timing",
)
path = root / "receipt.json"
path.write_text(json.dumps(receipt, indent=2))
log = (root / "server.log").open("w")
server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

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
    for iteration in range(2):
        for n, tokens in prompts.items():
            row = request(url, tokens, 32)
            receipt["rows"].append(dict(iteration=iteration, **row))
            path.write_text(json.dumps(receipt, indent=2))
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        lengths = list(prompts)
        for concurrency in [4, 8]:
            rows = list(
                pool.map(
                    lambda i: request(url, prompts[lengths[i % len(lengths)]], 32),
                    range(concurrency),
                )
            )
            receipt["cohorts"].append(dict(concurrency=concurrency, rows=rows))
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
