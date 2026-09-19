"""Bounded shape-controlled HTTP step evidence; not a new SWE throughput run."""

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from service_probe import request


def cases():
    return (
        [dict(kind="decode", batch=b, context=c) for c in (1024, 4096) for b in (1, 2, 4, 8)]
        + [dict(kind="prefill", batch=1, context=n) for n in (128, 256, 512, 1024, 2048)]
        + [dict(kind="mixed", batch=b, context=1024, joining=n)
           for b in (1, 4) for n in (128, 512, 1024, 1536)]
    )


def main():
    root = Path(sys.argv[1])
    arm = os.environ["COMPARE_NO_MTP"]
    delta_probe = os.environ.get("STEP_PREFILL_DELTA") == "1"
    mc2_probe = os.environ.get("STEP_MC2_QUALIFY") == "1"
    port = 32381
    url = f"http://127.0.0.1:{port}"
    source = Path(__file__).parent
    trace = json.loads((root.parent / "trace.json").read_text())
    seed = trace["sessions"][0]["calls"][0]["prompt_ids"]

    def prompt(length, slot):
        # First-token salt prevents accidental within-cohort prefix reuse. APC
        # remains on. Shape sweep is explicitly cold-cache, not SWE E2E.
        return [500 + slot] + (seed * (length // len(seed) + 1))[:length - 1]

    def post(path, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
            return json.loads(raw) if raw else None

    def rpc(method, *args):
        return post("/collective_rpc", dict(method=method, args=list(args), timeout=150))

    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", os.environ["QWEN_MODEL_PATH"],
        "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "qwen27",
        "--tensor-parallel-size", "2", "--distributed-executor-backend", "mp",
        "--worker-cls", "step_worker.Worker", "--dtype", "bfloat16",
        "--max-model-len", "8192", "--max-num-seqs", "8",
        "--max-num-batched-tokens", "2048", "--kv-cache-memory-bytes", str(6 * 1024**3),
        "--seed", "17", "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--async-scheduling", "--shutdown-timeout", "60",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--additional-config", '{"enable_cpu_binding":false}',
        "--profiler-config", json.dumps(dict(profiler="torch", torch_profiler_dir=str(root / "profiles"))),
    ]
    if arm == "candidate":
        command += ["--compilation-config", json.dumps(dict(
            cudagraph_mode="FULL", cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048],
            max_cudagraph_capture_size=2048))]
    receipt = dict(status="RUNNING", arm=arm, command=command, cohorts=[],
                   scope="Cold-cache shape-controlled HTTP workload; APC and AIV on, MTP off. External preallocated device events; no in-step synchronize. Forward envelope is not graph-body time.")

    def save():
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2))

    def cohort(case):
        n, b = case["context"], case["batch"]
        if case["kind"] == "prefill":
            return [request(url, prompt(n, 0), 4)]
        with concurrent.futures.ThreadPoolExecutor(b + 1) as pool:
            ready = [threading.Event() for _ in range(b)]
            lead = [pool.submit(request, url, prompt(n, slot), 48, on_first_content=e.set)
                    for slot, e in enumerate(ready)]
            join = []
            if case["kind"] == "mixed":
                assert all(e.wait(120) for e in ready), "lead did not begin decoding"
                join = [pool.submit(request, url, prompt(case["joining"], 9), 4)]
            return [f.result() for f in lead + join]

    save()
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("readiness")
                    time.sleep(2)
            # The first full pass warms every workload shape; never enters results.
            matrix = ([dict(kind="prefill", batch=1, context=n) for n in (1024, 2048)]
                      if delta_probe else cases())
            if mc2_probe:
                matrix = ([dict(kind="prefill", batch=1, context=n)
                           for n in (16, 32, 128, 1024, 2048)]
                          + [dict(kind="decode", batch=b, context=128) for b in (1, 4, 8)]
                          + [dict(kind="mixed", batch=1, context=128, joining=128)])
            if os.environ.get("STEP_REVERSE") == "1":
                matrix.reverse()
            for repeat in range(5 if delta_probe else 3):
                for index, case in enumerate(matrix):
                    post("/reset_prefix_cache")
                    label = f"r{repeat}-case{index}"
                    if repeat:
                        rpc("begin_step_measurement", label)
                    responses = cohort(case)
                    measured = rpc("end_step_measurement") if repeat else None
                    row = dict(label=label, warmup=not repeat, case=case,
                               requests=[{k: v for k, v in r.items() if k not in ("text", "chunk_times_s")} for r in responses],
                               measurement=measured)
                    receipt["cohorts"].append(row)
                    save()
                    print(arm, label, case, "done", flush=True)
            # Only six diagnostic steps, after unprofiled measurements. Profile
            # data never contributes to the curve, and event collection is off.
            post("/reset_prefix_cache")
            post("/start_profile")
            if mc2_probe:
                # Small/large prefills plus two decode replays each. Deliberately
                # verify actual captured routing, not only source-level intent.
                responses = []
                for n in (32, 1024):
                    post("/reset_prefix_cache")
                    responses.append(request(url, prompt(n, 0), 3))
                receipt["profile_prompt_lengths"] = [32, 1024]
            elif delta_probe:
                # Six actual forwards:1024 /1536+512 /1536+512 /1024.
                # One output avoids adding decode steps to the short capture.
                responses = []
                for n in (1024, 2048, 2048, 1024):
                    post("/reset_prefix_cache")
                    responses.append(request(url, prompt(n, 0), 1))
                receipt["profile_prompt_lengths"] = [1024, 2048, 2048, 1024]
            else:
                responses = cohort(dict(kind="mixed", batch=1, context=1024, joining=512))
            receipt["profile_requests"] = [
                {k: v for k, v in r.items() if k not in ("text", "chunk_times_s")}
                for r in responses
            ]
            post("/stop_profile")
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
                server.kill()
                server.wait()
                receipt["shutdown_timeout"] = True
            receipt["server_exit_code"] = server.returncode
            save()


if __name__ == "__main__":
    main()
