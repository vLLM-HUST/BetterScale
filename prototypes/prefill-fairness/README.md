# Chunked-prefill fairness investigation

Scope: investigate BetterScale issue #4 without changing runtime scheduling or
rerunning C16. This branch is separate from Conv Draft PR #5. No NPU was used.

## What can be recovered from the old run?

The Conv C16 arms used `apc_boundary.BoundaryScheduler(AsyncScheduler)` from
`runs/operator-response/20260927-qkv-serving01/service-v6`. Their retained
server directories contain HTTP/engine logs, prefix-hit records and aggregate
metrics, but no per-step scheduling sidecar. The launch sets `MTP_PROFILE=0`;
`timeline_probe.install` returns immediately. The available prefix-hit rows
contain request/prompt/hits, not timestamps or per-step grants. Historical
steps cannot be retrospectively exported from those records.

The previous HTTP timeline establishes overlapping cold-prefill/short-turn
intervals, not which step admitted the short request. Do not invent a historical
schedule from that overlap or equate a CPU fixture to those exact requests.

## Executed donor policy

Read-only source copied from hw3:
`/workspace/my-ascend-workspace/runs/qwen35-moe-mtp-256k/candidate-runtime1/`
`vllm/v1/core/sched/{scheduler,async_scheduler}.py`.
Local retained copies and full CPU step results are in workspace
`runs/operator-response/20260927-prefill-fairness01/`.

Relevant `scheduler.py` locations in this snapshot:

- 396+: each schedule creates a fresh4096-token budget.
- 442–624: RUNNING requests are visited in list order; grants are remaining
  demand limited by optional long-prefill threshold and remaining budget.
- 590: the grant is immediately deducted. There is no round-robin cursor that
  rotates a long prefill behind later arrivals after each chunk.
- 638–640: WAITING is considered only after running grants, with positive budget
  and no preemption in that step. It is not enough that request seats are free.
- 828–842: waiting grants also use the threshold/budget; 864–872 aligns hybrid
  chunks and breaks when a waiting request receives zero after alignment.
- 338–394: Mamba `align` can round a nonfinal grant down to a2048 multiple.

The launch does not set `long_prefill_token_threshold`; the pinned config
default is0 (no per-request cap). `AsyncScheduler` adds speculative bookkeeping,
not a scheduling-order override. The task's `BoundaryScheduler` changes prefix
publication and disables Eagle retreat inside the alignment helper; it does
not override `schedule`. Source inspection of this capsule found no grant-order
or threshold patch. The optional Ascend balance scheduler is a separate DP
mechanism, not a TP2/DP1 fairness fix.

**Mechanism:** chunking limits one wave's work, but does not itself share
successive waves fairly. An early running long prefill can monopolize the
budget. With alignment, a positive leftover can still be unusable to a waiting
prefill. These mechanisms fit the tail observation; their exact contribution
to its23s wait has not been measured.

## CPU evidence, deliberately narrower than serving

`probe_donor_budget.py` AST-extracts the unchanged leading statements and entire
RUNNING loop, plus the original alignment helper. No Torch/vLLM/NPU imports.
Allocation always succeeds; the waiting loop, native async completion,
preemption, connectors and model execution are NOT simulated. After each grant,
the fixture advances computed counts and asks whether the waiting request could
receive a positive aligned grant. These step counts are not latency estimates.

Fixture: long prompt143971 already in running; waiting prompt29399 with26624
reusable tokens; budget4096; optionally an earlier ready3-token decode.

| Fixture | First-step grants | Remaining | Waiting short usable grant |
|---|---|---:|---:|
| Old aligned, no cap | long4096 |0|0|
| Old aligned, cap2048 | long2048 |2048|2048|
| Old aligned, decode first | decode3 + long2048 |2045|0|
| Old aligned, cap2048 + decode first | decode3 + long2048 |2045|0|
| No alignment, decode first | decode3 + long4093 |0|0|
| No alignment, cap2048 + decode first | decode3 + long2048 |2045|2045|

With the controlled unchanging demand/ready assumptions, the uncapped aligned
fixture first offers the short request usable tokens at step36; with a decode
it is step71, unchanged by cap2048. These are synthetic CPU schedules, not
reconstructed C16 steps. `cpu-summary.json` retains the concise results.

Reproduce (absolute paths should be adapted to the source snapshot location):

```sh
python3 prototypes/prefill-fairness/probe_donor_budget.py \
  --scheduler /root/my-ascend-workspace/runs/operator-response/20260927-prefill-fairness01/donor/scheduler.py \
  --out /tmp/donor-budget.json
```

## Current integration seam, not the retired live loop

This worktree starts at `eee35fd`, after the resident-State default change.
`src/betterscale/models/qwen35/seat_scheduler.py:LiveStateScheduler.schedule`
calls `super().schedule` and attaches lease/generation metadata. It retains
the native asynchronous request selection and token grants. Its constructor
requires a pure-attention KV pool (`has_mamba_layers == False`), so the native
`need_mamba_block_aligned_split` condition is false. The no-alignment fixtures
above isolate this relevant difference, but are NOT full resident admission
or serving tests. Resident writer fences/page availability remain independent
reasons a request can wait.

Do not patch `src/betterscale/live/llm/qwen35/scheduler.py` merely because its
name looks right: that independent wave loop has oldest-ready family selection
and is not the current native-async resident-State entry. Replacing it would
not fix the active path's grants.

## What to record next, and when

No full msprof capture or new900s C16 replay is needed to establish this source
mechanism. Before choosing/accepting a fairness change, use one bounded serving
reproduction with fixed long prompt, warmed short continuation and fixed
arrival offset, recording scheduler-side steps:

- request arrival plus stable request ID; no prompt text/token IDs needed;
- step sequence and monotonic timestamp, running order and waiting IDs;
- per-request prompt/computed/output-placeholder counts and scheduled tokens;
- total budget/unused remainder, alignment mode and any preemption;
- admission/hit result and allocator/fence denial reason if admission fails;
- corresponding completion sequence and first output publication, to separate
  planned grants from async execution and HTTP-visible first token.

Save a bounded CPU-side buffer outside the measured region, not synchronous
file writes or tensor reads every step. Existing worker dispatch-only profiling
cannot explain a request absent from a submitted batch; observe EngineCore's
scheduler, not just the NPU graph. This trace should decide whether a cap is
sufficient or explicit age/rotation/admission reservation is needed. Preserve
existing decode service and state fences. Do not increase global query budget,
relax APC validity or assume two2048 prefills plus decode fit in4096.

Implementation and NPU validation are not performed by this investigation.
