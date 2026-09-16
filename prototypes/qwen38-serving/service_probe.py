"""Bounded real HTTP/SSE frontier; profile data never enters timing statistics."""

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def request(url, prompt, budget, *, stream=True):
    payload = dict(
        model="qwen27",
        prompt=prompt,
        temperature=0,
        max_tokens=budget,
        ignore_eos=True,
        stream=stream,
    )
    if stream:
        payload["stream_options"] = dict(include_usage=True)
    req = urllib.request.Request(
        url + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    events = []
    text = ""
    usage = None
    finish = None
    with urllib.request.urlopen(req, timeout=240) as response:
        if not stream:
            body = json.load(response)
            return dict(
                text=body["choices"][0]["text"],
                usage=body["usage"],
                latency_s=time.perf_counter() - start,
            )
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            body = json.loads(data)
            usage = body.get("usage") or usage
            for choice in body.get("choices", []):
                piece = choice.get("text", "")
                if piece:
                    events.append(time.perf_counter() - start)
                    text += piece
                finish = choice.get("finish_reason") or finish
    elapsed = time.perf_counter() - start
    if not usage or usage["completion_tokens"] != budget or finish != "length":
        raise RuntimeError(f"wrong terminal response: {usage}, {finish}")
    return dict(
        prompt_tokens=len(prompt),
        output_tokens=budget,
        text=text,
        usage=usage,
        latency_s=elapsed,
        ttft_s=events[0] if events else None,
        chunk_times_s=events,
        finish_reason=finish,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capsule", type=Path, required=True)
    p.add_argument(
        "--arm", choices=["sync", "async", "mtp1", "mtp2", "mtp3"], required=True
    )
    p.add_argument("--model", default="/models/vllm-ascend-models/Qwen3.8-27B")
    p.add_argument("--port", type=int, default=32181)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    out = a.capsule
    url = f"http://127.0.0.1:{a.port}"
    # Freeze semantically valid token IDs from the same tokenizer's accepted prompt.
    raw = json.loads((out / "prompt.json").read_text())
    prompt = (
        raw["prompt_token_ids"]
        if isinstance(raw, dict) and "prompt_token_ids" in raw
        else raw
    )
    if isinstance(prompt, dict):
        raise ValueError(f"unknown prompt schema: {list(prompt)}")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        a.model,
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
        "observe_worker.Worker",
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
        "--no-enable-prefix-caching",
        "--shutdown-timeout",
        "60",
        "--no-async-scheduling" if a.arm == "sync" else "--async-scheduling",
        "--additional-config",
        '{"enable_cpu_binding":false}',
        "--profiler-config",
        json.dumps(
            dict(
                profiler="torch",
                torch_profiler_dir=str(out / "profiles"),
                torch_profiler_with_stack=False,
            )
        ),
    ]
    if a.arm.startswith("mtp"):
        command += [
            "--speculative-config",
            json.dumps(dict(method="mtp", num_speculative_tokens=int(a.arm[-1]))),
        ]
    if os.environ.get("FULL_MTP"):
        assert a.arm == "mtp" + os.environ["FULL_MTP"]
        command[command.index("observe_worker.Worker")] = "bucket_full_worker.Worker"
        q = int(os.environ["FULL_MTP"]) + 1
        prefill_sizes = [
            ((n + q - 1) // q) * q for n in [64, 128, 256, 512, 1024, 1536, 2048]
        ]
        command[command.index("--max-num-batched-tokens") + 1] = str(prefill_sizes[-1])
        command += [
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
            "--compilation-config",
            json.dumps(
                dict(
                    cudagraph_mode="FULL_AND_PIECEWISE",
                    cudagraph_capture_sizes=[q, q * 2, q * 4, q * 8] + prefill_sizes,
                    max_cudagraph_capture_size=prefill_sizes[-1],
                )
            ),
        ]
    receipt = dict(
        status="STARTED",
        arm=a.arm,
        pack_conv=os.environ.get("SERVING_PACK_CONV") == "1",
        full_mtp=os.environ.get("FULL_MTP"),
        padded_gdn=os.environ.get("PADDED_PREFILL") == "1",
        command=command,
        cohorts=[],
        scope="HTTP SSE fixed English prompt; not SWE task accuracy",
    )
    path = out / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2))
    log = (out / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 720
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("readiness deadline")
            time.sleep(2)
        request(url, prompt[:512], 16)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda i: request(url, prompt[: 512 + i], 16), range(8)))
            for concurrency in [1, 4, 8]:
                for repeat in range(2):
                    # Identical population across arms; arrivals concurrent within cohort.
                    prompts = [
                        prompt[: [512, 2048, 1024, 1536][i % 4]]
                        for i in range(concurrency)
                    ]
                    start = time.perf_counter()
                    rows = list(pool.map(lambda ids: request(url, ids, 64), prompts))
                    elapsed = time.perf_counter() - start
                    receipt["cohorts"].append(
                        dict(
                            concurrency=concurrency,
                            repeat=repeat,
                            elapsed_s=elapsed,
                            tokens_per_s=64 * concurrency / elapsed,
                            requests=rows,
                        )
                    )
                    path.write_text(json.dumps(receipt, indent=2))
        if a.profile:
            with urllib.request.urlopen(
                urllib.request.Request(url + "/start_profile", data=b"", method="POST"),
                timeout=60,
            ):
                pass
            request(url, prompt[:2048], 20)
            with urllib.request.urlopen(
                urllib.request.Request(url + "/stop_profile", data=b"", method="POST"),
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
            receipt["shutdown_timeout"] = True
            server.kill()
            server.wait()
        receipt["server_exit_code"] = server.returncode
        path.write_text(json.dumps(receipt, indent=2))
        log.close()


if __name__ == "__main__":
    main()
