"""Bounded public-HTTP E2E cohorts, under the existing NPU lease supervisor.

Both policies run identical synthetic token requests; generation and acceptance
remain native, not forced identical. Record token counts, TTFT, inter-chunk gaps,
wall throughput and native speculative metrics. Loading/warmup is excluded.
This measures synthetic service cohorts, not original agent trajectories.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def completion(base, prompt, tokens, identity):
    start = time.monotonic()
    payload = dict(
        model="dsv4",
        prompt=prompt,
        temperature=0,
        max_tokens=tokens,
        ignore_eos=True,
        stream=True,
        stream_options={"include_usage": True},
        return_token_ids=True,
    )
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    arrivals, ids, usage = [], [], None
    done = False
    with urllib.request.urlopen(request, timeout=240) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event.get("choices", []):
                assert choice["index"] == 0
                new = choice.get("token_ids") or []
                if new or choice.get("text"):
                    arrivals.append(time.monotonic() - start)
                ids.extend(new)
    elapsed = time.monotonic() - start
    assert done and usage is not None and arrivals, "Incomplete SSE response"
    assert usage["prompt_tokens"] == len(prompt)
    assert usage["completion_tokens"] == tokens, "Fixed output budget not completed"
    assert not ids or len(ids) == tokens
    return dict(
        request=identity,
        input_tokens=len(prompt),
        output_tokens=tokens,
        token_ids=ids,
        ttft_s=arrivals[0],
        elapsed_s=elapsed,
        chunk_arrivals_s=arrivals,
        usage=usage,
    )


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert command
    (args.output / "command.json").write_text(json.dumps(command, indent=2))
    with (args.output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 540
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server failed: {server.returncode}")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=2) as r:
                        if r.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server readiness")
                time.sleep(3)
            rows = []
            with ThreadPoolExecutor(max_workers=args.seats * 2) as clients:
                # Warm target, producer, metadata and draft shapes before timings.
                tasks = [
                    clients.submit(completion, base, [17 + i % 4] * 128, 32, i)
                    for i in range(args.seats)
                ]
                for task in tasks:
                    task.result()
                for repeat in range(args.repeats):
                    for label, lengths, output in (
                        ("decode", [128] * args.seats, 256),
                        ("prefill", [4096] * args.seats, 128),
                        ("turnover", [8192] + [256] * (args.seats * 2 - 1), 128),
                    ):
                        tag = f"{repeat}-{label}"
                        with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
                            (args.output / f"{tag}-metrics-before.txt").write_bytes(
                                r.read()
                            )
                        start = time.monotonic()
                        tasks = [
                            clients.submit(
                                completion, base, [17 + i % 4] * n, output, i
                            )
                            for i, n in enumerate(lengths)
                        ]
                        results = [task.result() for task in tasks]
                        elapsed = time.monotonic() - start
                        with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
                            (args.output / f"{tag}-metrics-after.txt").write_bytes(
                                r.read()
                            )
                        rows.append(
                            dict(
                                label=tag,
                                elapsed_s=elapsed,
                                requests=results,
                                output_tps=sum(r["output_tokens"] for r in results)
                                / elapsed,
                            )
                        )
                        (args.output / "cohorts.json").write_text(
                            json.dumps(rows, indent=2)
                        )
                        print(tag, round(elapsed, 3), "seconds", flush=True)
            if args.quality_requests:
                quality_rows = json.loads(args.quality_requests.read_text())
                assert len(quality_rows) == 32
                (args.output / "quality-inputs.json").write_text(
                    json.dumps(quality_rows)
                )

                def quality(row):
                    request = urllib.request.Request(
                        base + "/v1/completions",
                        data=json.dumps(
                            dict(
                                model="dsv4",
                                prompt=row["prompt_token_ids"],
                                temperature=0,
                                max_tokens=row["max_new_tokens"],
                                stop_token_ids=[row["eos_token_id"]],
                                return_token_ids=True,
                            )
                        ).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=240) as response:
                        reply = json.load(response)
                    assert len(reply["choices"]) == 1
                    choice = reply["choices"][0]
                    assert reply["usage"]["prompt_tokens"] == len(
                        row["prompt_token_ids"]
                    )
                    assert len(choice["token_ids"]) <= row["max_new_tokens"]
                    return dict(
                        request_id=row["request_id"],
                        prompt_tokens=len(row["prompt_token_ids"]),
                        token_ids=choice["token_ids"],
                        finish_reason=choice["finish_reason"],
                        stop_reason=choice.get("stop_reason"),
                    )

                with ThreadPoolExecutor(max_workers=args.seats) as clients:
                    results = list(clients.map(quality, quality_rows))
                (args.output / "quality-result.json").write_text(
                    json.dumps(
                        dict(
                            status="COMPLETED_UNSCORED",
                            scope="Public HTTP; retained32 OpenCompass English retrieval",
                            requests=results,
                        ),
                        indent=2,
                    )
                )
            (args.output / "complete.json").write_text(
                json.dumps(dict(status="PASS", cohorts=len(rows)))
            )
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30880)
    parser.add_argument("--seats", type=int, choices=(4, 16), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quality-requests", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    main(parser.parse_args())
