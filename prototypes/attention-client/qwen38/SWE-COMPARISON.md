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
but must be labelled as such, not as unmodified vLLM. Fletcher approved that integration on September17; see
`CAPACITY-AND-COLOCATED.md` for progress and passed leaf gates.
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

## Retained-prefix integration (in progress)

`trace_plan.py` owns fixed output budgets and pending-token accounting: the last
emitted token is not yet target encoded and becomes a single continuation anchor
before the next recorded input delta. The CPU test protects this from double
prefill or output-budget overrun. `trace_client.py` is the first fixed-resident
multi-turn pilot, not yet a qualified performance lane. It captures decode,
chunks real input, and preserves request GDN/PLE/QSA between turns. The native
EP arm votes a common phase before collective entry. Pending prefill currently
has priority; this is a simple shared policy, not a tuned vLLM scheduler.

Before continuation prefill, GDN's accepted recurrent row and convolution slice
must become canonical row0. The plain prefill reader does not select the decode
candidate automatically. PLE already reads its own accepted endpoint and resets
the selector after successful non-speculative forward. These boundaries need a
hardware continuation gate before whole-workload timings can be trusted.

First admitted gate: four distinct sessions, first two turns, complete input
(4.6–4.9K initial prompts,679–4204 next-turn deltas), output capped8/turn.
This intentionally truncated gate is **not throughput evidence**. Whole trace
runs must omit both truncation options. No repeated session copies are used.

The first retained gate `qwen38-model-20260917T143653Z` failed PLE publication
on its first real512-token/query chunk after decode warmup; no SWE timing is
accepted from it. Static inspection found scalar-by-scalar response packing:
`bytes(cpu_uint8_tensor_row)` creates one Python scalar per byte. A CPU test of
1024x1024 BF16 responses on hw0 measured4.513s versus5.773ms for bulk byte-copy,
with byte-exact equality and an empty-wave check (`ple-codec-result.json`). This
is a codec microtest, not proof of the original failure's complete cause.
`trace_ple.py` scopes that change to this runner, retains the native byte ABI,
and acknowledges empty masked waves so idle EP members can participate.

`trace_commit.py` also preserves the *bounded* endpoint: the inherited helper
limits output count but returns pending/multi from the untruncated speculative
cabin. A terminal request could discard that State; a retained agent session
cannot. The CPU probe forces accepted K1 with remaining1 and checks first-output
pending/multi and the corresponding GDN/PLE selector. No donor/global patches.

The repaired smoke gate `qwen38-model-20260917T144438Z` completes all four
sessions' two turns (16 output tokens/session), with actual previous-turn
prefix reuse4615/4871/4732/4636 tokens. See `retained-gate.json`. Its variable
prefill tails trigger first-use Triton compilation (47–52s outliers), so those
timings are **not steady-serving measurements**. The next runner uses a fixed
512-row/query bucket, warms fresh and continuation prefill before the window,
compares final token IDs within TP, and checks that retained first history
pages remain byte-identical. Native EP routes only valid token rows, including
idle sources. The first matched pilot keeps the same four sessions and two
turns but restores all1408 recorded output tokens; it is still not the full
185-turn workload represented by these four sessions.

The separated matched pilot `qwen38-model-20260917T145121Z` completes all1408
outputs,26391 prefill tokens (including four pending-token anchors), and19310
reused-prefix tokens. All four attention ranks pass first-history-page retention
and final TP output-ID agreement. Maximum source duration is40.5824s; this is a
single four-session/two-turn pilot, not a complete-trajectory or quality result.

The first colocated attempt `qwen38-model-20260917T145508Z` fails during warmup
with HCCL communicator initialization error9, before a measured wave. Inspection
found that the legacy QSA-island constructor calls `new_group(local_tp_ranks)`
under WORLD8: the four DP groups supply different member lists at the same
creation position. For TP2 the island is exactly the existing TP pair.
`bootstrap_groups()` now seeds its cache with that deterministic, warmed group;
there is no reason to create another communicator. Also rendezvous after every
expert catalog has loaded, before data-plane warmup. Do not score that failed
startup as the control's throughput or erase it from the record.

## First matched pilot result

`qwen38-model-20260917T145121Z` (separated) versus
`qwen38-model-20260917T150050Z` (colocated), on the same hw0 eight910B2 cards,
repaired full48+MTP K1, State8GiB/attention rank, four **total** concurrent
sessions, first two complete turns each. Both process26391 prefill tokens,
reuse19310 prefix tokens, and emit1408 tokens. Both pass retained first-page
and final within-TP output-ID checks. All processes exit successfully; final
npu-smi shows eight healthy idle cards and no NPU processes.

| Metric | two TP2 sources + E4 | four TP2 groups + EP8 |
| --- | ---: | ---: |
| Maximum source duration | 40.5824s | 47.6982s |
| Committed output / duration | 34.6948tok/s | 29.5189tok/s |
| Attention-rank peak allocated HBM | 15.631GiB | 29.140GiB |
| Attention-rank final reserved HBM | 16.254GiB | 29.953GiB |

The observed throughput ratio is1.1753 in this **single pilot**, not a robust
scaling claim. Durations follow a shared host warmup rendezvous, not exact global
start/end timestamps. Raw receipts and `analyze_traces.py` preserve the denominator.

Most importantly, this is our same-model native-operator control, **not vLLM**.
Its initial scheduler votes a global prefill/decode phase and idles decode while
another group prefills. It does not yet use mixed target waves to hide that
stall. The separated side can progress independently. Therefore neither the
throughput ratio nor the large P99 interval difference can establish superiority
over a mature mixed DP+EP scheduler. Few prefill interruptions can also move
across the P99 cutoff in this small sample. Consult both maximum and P99 in
`swe-two-turn-result.json`; do not headline P99 as a stable SLO result.

No full-trajectory run, largest-context fill, maximum-concurrency sweep, full
quality evaluation or repeatability interval is claimed. Both model contracts
allow262144 context tokens with lookahead space required for MTP, but this pilot
configures16384 and activates8GiB State. Peak measurements at its short contexts
are not evidence that the same remaining HBM is free at262K context. The
physical maximum State allocation remains unqualified.
