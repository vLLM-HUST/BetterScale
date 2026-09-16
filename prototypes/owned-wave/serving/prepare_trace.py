"""Pin whole Open-SWE trajectories and render their original history for Qwen.

Dataset content is inert input, never executed. Selection excludes WHOLE
trajectories; no prompt, response or turn is truncated to fit the benchmark.
"""

import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--sessions", type=int, default=4)
p.add_argument("--scan-rows", type=int, default=5000)
p.add_argument("--max-context", type=int, default=32768)
p.add_argument("--max-output-total", type=int, default=4096)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
source = Path("/root/my-ascend-workspace/datasets/nvidia/Open-SWE-Traces")
model = "/data/shared_models/Qwen3-30B-A3B"
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
selected, rejected = [], []
seen = 0
for shard in sorted((source / "data/openhands/deepseek_v4_flash").rglob("*.parquet")):
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=8):
        for row in batch.to_pylist():
            seen += 1
            messages = row["messages"]
            turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            reason = None
            if not 2 <= len(turns) <= 12:
                reason = "whole_trajectory_turn_count"
            calls = []
            if reason is None:
                tools = [
                    json.loads(x) if isinstance(x, str) else x for x in row["tools"]
                ]
                try:
                    for i in turns:
                        prefix = tok.apply_chat_template(
                            messages[:i],
                            tools=tools,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        full = tok.apply_chat_template(
                            messages[: i + 1],
                            tools=tools,
                            add_generation_prompt=False,
                            tokenize=False,
                        )
                        if full[: len(prefix)] != prefix:
                            raise ValueError(
                                "generation prefix is not assistant serialization prefix"
                            )
                        budget = len(
                            tok.encode(full[len(prefix) :], add_special_tokens=False)
                        )
                        if budget < 1:
                            raise ValueError("empty assistant token continuation")
                        calls.append(
                            dict(
                                message_index=i,
                                prompt_ids=tok.encode(prefix, add_special_tokens=False),
                                output_tokens=budget,
                            )
                        )
                    if (
                        max(len(c["prompt_ids"]) + c["output_tokens"] for c in calls)
                        > a.max_context
                    ):
                        reason = "whole_trajectory_context"
                    elif sum(c["output_tokens"] for c in calls) > a.max_output_total:
                        reason = "whole_trajectory_output_budget"
                except (ValueError, TypeError, KeyError) as exc:
                    reason = f"template:{type(exc).__name__}:{exc}"
            identity = dict(
                trajectory_id=row["trajectory_id"],
                instance_id=row["instance_id"],
                source_shard=str(shard.relative_to(source)),
            )
            if reason:
                rejected.append(dict(**identity, reason=reason))
            else:
                selected.append(dict(**identity, calls=calls))
                print(
                    json.dumps(
                        dict(
                            **identity,
                            turns=len(calls),
                            max_prompt=max(len(c["prompt_ids"]) for c in calls),
                            output_tokens=sum(c["output_tokens"] for c in calls),
                        )
                    ),
                    flush=True,
                )
            if len(selected) >= a.sessions or seen >= a.scan_rows:
                break
        if len(selected) >= a.sessions or seen >= a.scan_rows:
            break
    if len(selected) >= a.sessions or seen >= a.scan_rows:
        break
manifest = dict(
    dataset="nvidia/Open-SWE-Traces",
    revision="fb0c0dccc7a5cce79b3f6de891848acdede36685",
    license="CC-BY-4.0",
    model=model,
    sessions=selected,
    scanned=seen,
    selection=dict(
        min_turns=2,
        max_turns=12,
        max_context=a.max_context,
        max_output_total=a.max_output_total,
    ),
    transforms=dict(
        truncated=False,
        history="original recorded messages, not generated responses",
        output_budget="Qwen-tokenized complete assistant serialization suffix (BPE at suffix boundary is independent)",
        tool_execution=False,
        tool_wait_seconds=0,
        arrival_policy="one initially ready call per session; next turn after previous completion",
    ),
    scope="closed-loop concurrent trace-input replay; not SWE task solving, recorded arrival timing or population-wide throughput",
)
(a.output / "trace.json").write_text(json.dumps(manifest))
(a.output / "rejections.json").write_text(json.dumps(rejected, indent=2))
assert (
    len(selected) == a.sessions
), f"only {len(selected)} complete trajectories selected; inspect rejections"
