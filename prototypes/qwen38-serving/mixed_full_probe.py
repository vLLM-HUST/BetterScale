"""Two real mixed batches, FULL/NONE hidden and cache shadows; no timing claim."""

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
signature = tuple(int(n) for n in os.environ.get("MIXED_SIGNATURE", "1,512").split(","))
decodes = next(i for i, n in enumerate(signature) if n > 1)
tokens = sum(signature)
width = 2048
url = "http://127.0.0.1:32181"
prompt = json.loads((root / "prompt.json").read_text())["prompt_token_ids"]
if len(signature) == decodes + 1:
    lengths = [signature[-1], width - decodes + signature[-1]]
else:
    assert signature == (1, 1, 1024, 1022)
    lengths = [[1024, 1536], [1024, 1536]]
prompt_lengths = [
    n for value in lengths for n in (value if isinstance(value, list) else [value])
]
prompts = {n: (prompt * 4)[:n] for n in [512, *prompt_lengths]}
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
            cudagraph_capture_sizes=[1, 2, 4, 8, tokens],
            max_cudagraph_capture_size=tokens,
        )
    ),
]

receipt = dict(status="STARTED", command=command, signature=signature, rows=[])
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
    request(url, prompts[512], 4)
    import threading

    def cohort(length):
        ready = [threading.Event() for _ in range(decodes)]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(signature) + 1
        ) as pool:
            ongoing = [
                pool.submit(request, url, prompts[512], 96, on_first_content=event.set)
                for event in ready
            ]
            if not all(event.wait(120) for event in ready):
                raise TimeoutError("ongoing request first token")
            if isinstance(length, list):
                hold = pool.submit(rpc, "hold_for_mixed_inputs")
                time.sleep(0.1)
                joined = []
                for n in length:
                    joined.append(pool.submit(request, url, prompts[n], 4))
                    time.sleep(0.02)
                receipt["correctness_staging_hold"] = hold.result()
            else:
                joined = pool.submit(request, url, prompts[length], 4)
            return dict(
                ongoing=[f.result() for f in ongoing],
                joined=(
                    [f.result() for f in joined]
                    if isinstance(joined, list)
                    else joined.result()
                ),
            )

    if os.environ.get("MIXED_TIMING") == "1":
        assert len(signature) == decodes + 1
        receipt["scope"] = (
            "same-process exact mixed FULL/NONE, identical native scheduling and metadata, no state shadow; profiled cohorts excluded"
        )
        phases = [("full", "warmup"), ("none", "warmup")]
        phases += [(mode, "measure") for mode in ["none", "full", "full", "none"] * 2]
        phases += [("none", "profile"), ("full", "profile")]
        for trial, (mode, phase) in enumerate(phases):
            label = mode if phase == "profile" else None
            rpc("set_mixed_mode", mode, label)
            row = cohort(lengths[0])
            row.update(
                trial=trial, mode=mode, phase=phase, dispatch=rpc("mixed_status")
            )
            expected = "FULL" if mode == "full" else "NONE"
            assert all(
                x["counts"].get(expected, 0) == 1 and x["profile_closed"]
                for x in row["dispatch"]["results"]
            ), row["dispatch"]
            receipt["rows"].append(row)
            path.write_text(json.dumps(receipt, indent=2))
        receipt["status"] = "PASS"
    else:
        rpc("arm_mixed_shadow", 2)
        for trial, length in enumerate(lengths):
            row = cohort(length)
            row["trial"] = trial
            receipt["rows"].append(row)
        receipt["shadow"] = rpc("mixed_shadow_result")
        receipt["status"] = (
            "PASS"
            if all(x["passed"] for x in receipt["shadow"]["results"])
            else "SHADOW_MISMATCH"
        )
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
