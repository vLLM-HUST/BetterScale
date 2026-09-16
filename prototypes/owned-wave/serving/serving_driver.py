"""Matched native/owned concurrent input replay; profiled runs are separate."""

import json
import os
import time
from pathlib import Path
from vllm import LLM, SamplingParams


def native_round(llm, sessions):
    engine = llm.llm_engine
    active = {}
    records = []
    begin = time.perf_counter()

    def submit(session, turn):
        call = sessions[session]["calls"][turn]
        key = f"s{session}t{turn}"
        active[key] = dict(
            session=session, turn=turn, call=call, start=time.perf_counter(), first=None
        )
        engine.add_request(
            key,
            dict(prompt_token_ids=call["prompt_ids"]),
            SamplingParams(
                temperature=0,
                max_tokens=call["output_tokens"],
                ignore_eos=True,
                detokenize=False,
            ),
        )

    for session in range(len(sessions)):
        submit(session, 0)
    while active:
        outputs = engine.step()
        now = time.perf_counter()
        for output in outputs:
            r = active[output.request_id]
            tokens = output.outputs[0].token_ids
            if tokens and r["first"] is None:
                r["first"] = now
            if output.finished:
                assert len(tokens) == r["call"]["output_tokens"]
                records.append(
                    dict(
                        session=r["session"],
                        turn=r["turn"],
                        prompt_tokens=len(r["call"]["prompt_ids"]),
                        output_tokens=len(tokens),
                        token_ids=list(tokens),
                        hit_tokens=output.num_cached_tokens,
                        ttft_s=r["first"] - r["start"],
                        latency_s=now - r["start"],
                        tpot_s=(now - r["first"]) / max(1, len(tokens) - 1),
                    )
                )
                del active[output.request_id]
                if r["turn"] + 1 < len(sessions[r["session"]]["calls"]):
                    submit(r["session"], r["turn"] + 1)
    return dict(
        elapsed_s=time.perf_counter() - begin,
        records=records,
        output_tokens=sum(x["output_tokens"] for x in records),
        hit_tokens=sum(x["hit_tokens"] or 0 for x in records),
    )


out = Path(os.environ["SERVING_OUTPUT"])
out.mkdir(exist_ok=True)
arm = os.environ.get("BENCH_ARM", "owned")
dummy = os.environ.get("QWEN_DUMMY") == "1"
rounds = int(os.environ.get("BENCH_ROUNDS", "2"))
profile = int(os.environ.get("PROFILE_STEPS", "0"))
if dummy:
    sessions = [
        dict(
            trajectory_id=f"synthetic{s}",
            calls=[
                dict(prompt_ids=[17 + s] * 257, output_tokens=8),
                dict(prompt_ids=[17 + s] * 257 + [80 + s] * 132, output_tokens=7),
            ],
        )
        for s in range(4)
    ]
    maximum, chunk, kv = 1024, 256, 256 * 1024**2
else:
    manifest = json.loads(Path(os.environ["SWE_TRACE"]).read_text())
    sessions = manifest["sessions"]
    maximum, chunk, kv = 32768, 1024, 6 * 1024**3
maximum = int(os.environ.get("BENCH_MAX_LENGTH", maximum))
chunk = int(os.environ.get("BENCH_CHUNK", chunk))
kv = int(os.environ.get("BENCH_KV_BYTES", kv))
chunks = [2**i for i in range(chunk.bit_length())]
if os.environ.get("PROFILE_GRAPH_CHUNKS"):
    if not profile or arm != "owned":
        raise ValueError("reduced graph catalog is only for owned diagnostic profiles")
    chunks = [int(x) for x in os.environ["PROFILE_GRAPH_CHUNKS"].split(",")]
    if (
        not chunks
        or chunks != sorted(set(chunks))
        or chunks[0] < 1
        or chunks[-1] != chunk
    ):
        raise ValueError("invalid profile graph catalog")
config = dict(
    model=os.environ.get("BENCH_MODEL", "/data/shared_models/Qwen3-30B-A3B"),
    tensor_parallel_size=int(os.environ.get("BENCH_TP", "2")),
    enable_expert_parallel=os.environ.get("BENCH_EP", "1") == "1",
    dtype="bfloat16",
    distributed_executor_backend="mp",
    worker_cls="serving_worker.ServingWorker",
    max_model_len=maximum,
    max_num_batched_tokens=chunk,
    max_num_seqs=len(sessions),
    kv_cache_memory_bytes=kv,
    enable_prefix_caching=True,
    skip_tokenizer_init=True,
    seed=123,
    disable_log_stats=False,
    compilation_config=dict(
        cudagraph_mode="FULL",
        cudagraph_capture_sizes=chunks,
        max_cudagraph_capture_size=chunk,
    ),
    additional_config=dict(
        enable_cpu_binding=False,
        ascend_compilation_config=dict(
            enable_npugraph_ex=True, enable_static_kernel=False
        ),
    ),
)
if dummy:
    config.update(load_format="dummy", hf_overrides=dict(num_hidden_layers=2))
(out / "config.json").write_text(json.dumps(config, indent=2))
llm = LLM(**config)
order = ["native", "owned"] if arm == "both" else [arm]
if os.environ.get("BENCH_ORDER") == "owned-first":
    order.reverse()
for stage in order:
    if stage == "native":
        results = []
        for repeat in range(rounds):
            llm.collective_rpc("reset_memory_peaks")
            if profile and repeat == 0:
                llm.collective_rpc("begin_native_profile", args=(profile,))
            row = native_round(llm, sessions)
            if profile and repeat == 0:
                llm.collective_rpc("end_native_profile")
            row["repeat"] = repeat
            row["memory"] = llm.collective_rpc("memory_receipt")
            results.append(row)
            (out / "native.json").write_text(
                json.dumps(dict(rounds=results, profiled=bool(profile)))
            )
    if stage == "owned":
        result = llm.collective_rpc(
            "run_serving", args=(sessions, rounds, chunks, profile)
        )
        (out / "owned.json").write_text(json.dumps(result))
comparison = None
if arm == "both":
    from report import compare

    comparison = compare(out.parent)
    (out.parent / "comparison.json").write_text(json.dumps(comparison, indent=2))
    # Equal workload and TP agreement remain mandatory. Cross-arm token identity
    # is observational, not a performance gate (Fletcher, 2026-09-16).
(out / "complete.json").write_text(
    json.dumps(
        dict(
            status="PASS",
            arm=arm,
            profiled=bool(profile),
            token_comparison=comparison["token_comparison"] if comparison else None,
            acceptance="protocol and equal work; cross-arm tokens observational",
        )
    )
)
