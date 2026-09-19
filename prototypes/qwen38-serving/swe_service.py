"""Closed-loop original-history SWE HTTP replay; dataset tool calls stay inert."""

import argparse
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


def barrier(root, phase):
    (root / f"{phase}.ready").touch()
    deadline = time.monotonic() + 1800
    while not (root.parent / f"{phase}.go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"paired barrier: {phase}")
        time.sleep(0.1)


def replay(url, sessions, concurrency):
    start = time.perf_counter()

    def session(item):
        index, trace = item
        rows = []
        for turn, call in enumerate(trace["calls"]):
            submitted = time.perf_counter() - start
            row = request(url, call["prompt_ids"], call["output_tokens"])
            row.update(session=index, turn=turn, submitted_s=submitted)
            rows.append(row)
        return rows

    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        rows = [
            r
            for session_rows in pool.map(session, enumerate(sessions))
            for r in session_rows
        ]
    elapsed = time.perf_counter() - start
    return dict(
        concurrency=concurrency,
        elapsed_s=elapsed,
        output_tokens=sum(r["output_tokens"] for r in rows),
        requests=rows,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    p.add_argument("--arm", choices=["baseline", "candidate"], required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    root = a.capsule
    trace = json.loads((root.parent.parent / "trace.json").read_text())
    sessions = trace["sessions"]
    assert len(sessions) == 8
    assert all(
        0 < len(c["prompt_ids"]) + c["output_tokens"] <= 8192
        for s in sessions
        for c in s["calls"]
    )
    url = f"http://127.0.0.1:{a.port}"
    apc = os.environ.get("SWE_PREFIX_CACHING", "1") == "1"
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        os.environ["QWEN_MODEL_PATH"],
        "--host",
        "127.0.0.1",
        "--port",
        str(a.port),
        "--served-model-name",
        "qwen27",
        "--tensor-parallel-size",
        "2",
        "--distributed-executor-backend",
        "mp",
        "--worker-cls",
        "concurrency_worker.Worker",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "2048",
        "--kv-cache-memory-bytes",
        str(6 * 1024**3),
        "--seed",
        "17",
        "--enable-prefix-caching" if apc else "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--async-scheduling",
        "--shutdown-timeout",
        "60",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--additional-config",
        '{"enable_cpu_binding":false}',
        "--profiler-config",
        json.dumps(dict(profiler="torch", torch_profiler_dir=str(root / "profiles"))),
    ]
    if a.arm == "candidate":
        command += [
            "--compilation-config",
            json.dumps(
                dict(
                    cudagraph_mode="FULL",
                    cudagraph_capture_sizes=[
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                        1536,
                        2048,
                    ],
                    max_cudagraph_capture_size=2048,
                )
            ),
        ]
    receipt = dict(
        status="STARTED",
        arm=a.arm,
        command=command,
        rounds=[],
        devices=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        communication="native AIV enabled on both arms",
        source_commit=os.environ.get("SWE_SOURCE_COMMIT", "unrecorded"),
        prefix_caching=apc,
        cache_start="empty before each cohort" if apc else "disabled",
        scope="Closed-loop whole SWE sessions over HTTP. Original recorded history; fixed recorded response-token budgets; no tool execution, MTP or accuracy claim. APC configuration recorded separately. Dormant identical observation hooks in timed runs; profile separate.",
    )
    path = root / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2))
    with (root / "server.log").open("w") as log:
        env = os.environ.copy()
        if apc:
            # Loopback-only dev endpoint is used outside timing to equalize
            # initial cache state, never to alter the scheduler during replay.
            env["VLLM_SERVER_DEV_MODE"] = "1"
        server = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 720
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError("server readiness")
                    time.sleep(2)
            with concurrent.futures.ThreadPoolExecutor(8) as pool:
                list(
                    pool.map(
                        lambda s: request(url, s["calls"][0]["prompt_ids"], 16),
                        sessions,
                    )
                )
            for concurrency in map(int, os.environ["SWE_CONCURRENCIES"].split(",")):
                if apc:
                    req = urllib.request.Request(
                        url + "/reset_prefix_cache", method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        assert response.status == 200
                barrier(root, f"c{concurrency}")
                cohort = replay(url, sessions, concurrency)
                if apc:
                    hits = [
                        row["usage"]["prompt_tokens_details"]["cached_tokens"]
                        for row in cohort["requests"]
                    ]
                    assert sum(hits) > 0, "APC enabled but no prefix tokens reused"
                    cohort["cached_prompt_tokens"] = sum(hits)
                receipt["rounds"].append(cohort)
                path.write_text(json.dumps(receipt, indent=2))
            if a.profile:
                barrier(root, "profile")
                # Preserve original prompt/budget pairs. Stage a joining prefill
                # while another SWE request decodes, to expose the mixed hinge.
                calls = sorted(
                    (c for s in sessions for c in s["calls"]),
                    key=lambda c: len(c["prompt_ids"]),
                )
                lead = next(c for c in calls if c["output_tokens"] >= 128)
                join = calls[0]
                ready = threading.Event()
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    first = pool.submit(
                        request,
                        url,
                        lead["prompt_ids"],
                        lead["output_tokens"],
                        on_first_content=ready.set,
                    )
                    if not ready.wait(120):
                        raise TimeoutError("profile lead first token")
                    with urllib.request.urlopen(
                        urllib.request.Request(
                            url + "/start_profile", data=b"", method="POST"
                        ),
                        timeout=60,
                    ):
                        pass
                    second = pool.submit(
                        request, url, join["prompt_ids"], join["output_tokens"]
                    )
                    receipt["profile_requests"] = [first.result(), second.result()]
                with urllib.request.urlopen(
                    urllib.request.Request(
                        url + "/stop_profile", data=b"", method="POST"
                    ),
                    timeout=120,
                ):
                    pass
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
            path.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
