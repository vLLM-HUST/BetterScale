"""Compare equal-work replay arms, preserving cold-start and profiling exclusions."""

import argparse
import json
import math
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def stats(row):
    result = dict(
        elapsed_s=row["elapsed_s"],
        output_tokens=row["output_tokens"],
        output_tokens_s=row["output_tokens"] / row["elapsed_s"],
        hit_tokens=row["hit_tokens"],
    )
    for key in ["ttft_s", "tpot_s", "latency_s"]:
        values = [x[key] for x in row["records"]]
        result[key] = {
            label: percentile(values, p)
            for label, p in [("p50", 0.5), ("p95", 0.95), ("p99", 0.99)]
        }
    result["memory"] = row.get("memory")
    if "waves" in row:
        result["waves"] = row["waves"]
    return result


def compare(capsule):
    engine = capsule / "engine"
    native = json.loads((engine / "native.json").read_text())
    config = json.loads((engine / "config.json").read_text())
    owned = [
        json.loads((engine / f"candidate-rank{rank}.json").read_text())
        for rank in range(config["tensor_parallel_size"])
    ]
    assert all(r["status"] == "PASS" for r in owned), "owned protocol failed"
    excluded = (capsule / "timing-exclusion.json").exists()
    config = json.loads((engine / "config.json").read_text())
    result = dict(
        capsule=str(capsule),
        model=config["model"],
        weights=config.get("load_format", "real"),
        scope="closed-loop input replay, native LLMEngine versus owned in-worker execution; not HTTP serving",
        latency_boundary={
            "native": "client engine outputs",
            "owned": "worker all-rank receipt retirement",
        },
        timing_eligible=not excluded and not native["profiled"],
        rounds=[],
    )
    if excluded:
        result["timing_exclusion"] = json.loads(
            (capsule / "timing-exclusion.json").read_text()
        )
    for nr, *peers in zip(native["rounds"], *(r["rounds"] for r in owned), strict=True):
        cr = peers[0]
        index = lambda r: {(x["session"], x["turn"]): x for x in r["records"]}
        n, c = index(nr), index(cr)
        assert n.keys() == c.keys()
        for peer in peers[1:]:
            p = index(peer)
            assert c.keys() == p.keys()
            assert all(
                c[k]["token_ids"] == p[k]["token_ids"] for k in c
            ), "TP disagreement"
        mismatches = []
        for key in n:
            assert n[key]["prompt_tokens"] == c[key]["prompt_tokens"]
            assert n[key]["output_tokens"] == c[key]["output_tokens"]
            if n[key]["token_ids"] != c[key]["token_ids"]:
                first = next(
                    i
                    for i, (x, y) in enumerate(
                        zip(n[key]["token_ids"], c[key]["token_ids"])
                    )
                    if x != y
                )
                mismatches.append(
                    dict(
                        session=key[0],
                        turn=key[1],
                        first_divergent_token=first,
                        native=n[key]["token_ids"][first],
                        owned=c[key]["token_ids"][first],
                    )
                )
        result["rounds"].append(
            dict(
                repeat=nr["repeat"],
                cache_start="cold" if nr["repeat"] == 0 else "retained",
                native=stats(nr),
                owned=stats(cr),
                exact_calls=len(n) - len(mismatches),
                calls=len(n),
                mismatches=mismatches,
            )
        )
    result["token_comparison"] = (
        "PASS" if not any(r["mismatches"] for r in result["rounds"]) else "DIVERGED"
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("capsule", type=Path)
    a = p.parse_args()
    result = compare(a.capsule)
    (a.capsule / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}, indent=2))
    for r in result["rounds"]:
        print(
            json.dumps(
                dict(
                    repeat=r["repeat"],
                    exact_calls=r["exact_calls"],
                    calls=r["calls"],
                    native_seconds=r["native"]["elapsed_s"],
                    owned_seconds=r["owned"]["elapsed_s"],
                    mismatches=r["mismatches"][:3],
                )
            )
        )
