# Equal-eight-device agent-load comparison

Status (2026-09-17): preparation, **not a performance result**. The earlier
short fixed-batch decode receipts do not implement this experiment.

## Question and controls

Compare two TP2 attention sources + four dedicated expert devices with four
TP2 attention groups sharing a colocated EP8 plane. Both use eight 910B devices,
the same repaired Eco-Tech checkpoint, tokenizer, quantization and MTP setting.
Keep shared experts on the attention devices in both cases. Do not count TP
replicas as independent requests or compare six-card controls as equal hardware.

The question is whether pooling experts improves *completed agent work under
irregular demand*, not whether a larger resident request count is possible.
Measure fixed-work makespan, committed output tokens/s, TTFT and inter-output
latency distributions, and prefill/decode service time. Record admission wait
separately from engine queueing. Report model loading/capture separately from
steady service, without silently excluding per-request cold prefill.

Use one frozen set of distinct complete trajectories for both topologies, with
stable session placement within each run. Replaying more sources must partition
that same work, not multiply it. Sweep total active conversations rather than
using the same per-source concurrency, which would give the four-source control
twice the work. A session submits its next turn only after its previous output
budget has completed. New sessions replace completed sessions until the fixed
set drains. Also inspect server co-batching by layer and phase; throughput gains
alone do not demonstrate cross-source batching.

## Data, preparation and limitations

Download directly on hw0, not by copying local model/data payloads:

- Dataset: `nvidia/Open-SWE-Traces` (CC-BY-4.0).
- Revision: `fb0c0dccc7a5cce79b3f6de891848acdede36685`.
- Cohort: `openhands/deepseek_v4_flash/scale-swe`, nine Parquet shards,
  previously inventoried as 21,208 trajectories / 2,520,770,368 bytes.
- Remote root: `/workspace/betterscale-hw0/datasets/Open-SWE-Traces`.
- Download log/driver: `/workspace/betterscale-hw0/runs/swe-traces-20260917/`.
- Direct HF connectivity failed; download uses the HF mirror with the same
  immutable revision. A receipt is written only after all nine shards finish.

Reuse the existing `prototypes/agent-trace-serving/prepare.py` from the workspace
research repository as a disconnected experiment artifact, and retokenize with
the hw0 Qwen3.8 tokenizer. Old DSV4 token IDs must not be reused. Sample distinct
trajectories without replacement (seed 20260917); inspect the resulting turn,
new-input, output and context distributions before selecting the run budget.

The existing format supplies recorded system/user/tool observations and tool
schemas as role-labelled deltas; each engine appends its own generated history.
Output lengths are fixed by retokenized recorded assistant text. Tools are data,
never executed. Missing tool wait times are zero, not invented authentic arrival
timestamps. This is **SWE-derived agent-load replay**, not adaptive agents,
SWE-bench Verified task accuracy, original chat-template execution, or original
cache-hit counts. Actual incremental state reuse must be measured, not assumed.
Do not silently truncate long turns to the current prototype's small capacity.

## Readiness gaps found before launching

1. The current Qwen38 remote source accepts at most 32 token rows. It has a
   fixed-membership short-window runner, not yet a full trajectory scheduler.
   Long prefill, session admission/retirement and prefix continuation need their
   own integration gates. This is separate from expert-only batch microbench
   results in the older BF16 implementation.
2. hw0's installed donor reports vLLM 0.23.0 and vLLM-Ascend 0.23.0rc1; it is
   not a qualified native Qwen4Exp serving baseline. The inspected upstream vLLM
   source at `5690b02c03832a4ac3af231d3ecebe649c188095` selects NVIDIA/AMD model
   implementations in `vllm/models/qwen4_exp/__init__.py`, not an Ascend lane.
3. Our owned Qwen38 MoE binding explicitly requires `EP == TP` in
   `livemodule/arch/ascend/llm/qwen38/moe.py`. TP2 x DP4 with EP8 is therefore
   not a launcher-only change. Merely deleting that guard is incorrect: token
   views, routing/exchange, local expert ownership and output return must agree.

A same-implementation colocated EP8 control is a useful topology experiment,
but must be labelled as such, not as unmodified vLLM. Choosing that integration
versus switching to an already supported model is pending Fletcher's decision.
No donor pins, installed donor runtime or public defaults were changed.

## Prepared workload receipt

The nine-shard download finished at 2026-09-17T13:44:00Z. Parquet footer census
matches 21,208 trajectories / 2,520,770,368 bytes. The disconnected preparation
script comes from workspace commit `cd1b55ef3e8905ebc3e3b80d6bdc55380fe70d1c`.
PyArrow25.0.1 was installed in the private hw0 experiment venv; no donor upgrade.

Frozen plan on hw0:
`/workspace/betterscale-hw0/runs/swe-traces-20260917/qwen38-swe32.json`.
It contains 32 distinct full sessions, 1,445 turns, 837,431 output tokens and
1,872,933 recorded new-input tokens (1,874,346 with continuation anchors).
Maximum complete horizon is180,143 tokens; the inherited summary adds a K5
margin and reports180,148, **not a chosen MTP setting**.

See `swe-workload-stats.json` for nearest-rank quantiles:

| Tokens per turn | P50 | P95 | P99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| New prefill | 575 | 4,732 | 11,314 | 28,861 |
| Output | 343 | 1,818 | 3,011 | 5,457 |
| Context before generation | 49,133 | 131,051 | 159,301 | 179,189 |

1,420 of1,445 turns exceed the current32-row source window. This measured shape
makes a large-prefill integration gate essential; feeding everything through
32-row fragments would benchmark that artificial limit, not the intended
expert-pooling design. No accelerator run or topology performance comparison
has been launched for this workload yet.
