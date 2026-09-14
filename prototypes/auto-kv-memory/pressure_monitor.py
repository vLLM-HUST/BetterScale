"""Persist partial HTTP capacity evidence even if a cohort is interrupted."""

import json
import time
import urllib.request


def collect(futures, base, output, tag, start):
    seen = set()
    with (
        (output / f"{tag}-pressure.jsonl").open("w", buffering=1) as samples,
        (output / f"{tag}-completed.jsonl").open("w", buffering=1) as completed,
    ):
        while True:
            for index, future in enumerate(futures):
                if index in seen or not future.done():
                    continue
                try:
                    record = dict(index=index, result=future.result())
                except Exception as exc:
                    record = dict(index=index, error=repr(exc))
                completed.write(json.dumps(record) + "\n")
                seen.add(index)
            if len(seen) == len(futures):
                break
            record = dict(elapsed_s=time.monotonic() - start)
            try:
                with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
                    metrics = response.read().decode()
                record["metrics"] = [
                    line
                    for line in metrics.splitlines()
                    if not line.startswith("#")
                    and any(
                        key in line
                        for key in (
                            "kv_cache_usage_perc",
                            "num_requests_running",
                            "num_preemptions_total",
                        )
                    )
                ]
            except Exception as exc:
                # Observation failure is evidence, not license to restart a job.
                record["observation_error"] = repr(exc)
            samples.write(json.dumps(record) + "\n")
            time.sleep(1)
    # Preserve the harness's original error semantics after saving partial work.
    return [future.result() for future in futures]
