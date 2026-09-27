# Chunked-prefill fairness investigation

Scope: investigate BetterScale issue #4 and implement the subsequent simple
rotating-start policy. This branch is separate from Conv Draft PR #5. The
initial investigation was CPU-only; later implementation evidence is separated
below. No new C16 throughput comparison is claimed.

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
At that base, `src/betterscale/models/qwen35/seat_scheduler.py:LiveStateScheduler.schedule`
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

The initial investigation above (7672dea) did not change scheduling or run NPUs.
The implementation below is its subsequent, separately authorized continuation.

## First implementation: rotating opportunity, greedy grants

`models/qwen35/prefill_round_robin.py` owns the small circular request list.
Every scheduling attempt rotates its START by one request, not by the last
request served. Finished prefills leave; new/waiting prefills join. Known
writer-fence/seat blocks stay in the ring but receive no grant while blocked.
Decode/MTP verification demand is reserved first. Remaining tokens are greedily
assigned in circular order, respecting free execution slots for new admission.
The request's current exact resident checkpoint is previewed without claiming
or mutating it; actual allocation/hit publication still belongs to native code.

The pinned donor has no general per-request grant hook. Rather than copy its
complete schedule body or rewrite async scheduling, `fair_schedule.py` binds a
class-local adaptation of that exact method: two grant limits and one native
waiting-queue skip. The method source is checksum-pinned before AST adaptation;
unknown versions fail closed. The exact Ascend balance wrapper is unwrapped
only on its disabled forwarding path; enabled balance is rejected. Donor module globals and installed files are NOT
patched. Native State callbacks, output schema, async updates, allocation and
preemption remain in the original body. Waiting entries lacking a grant are
parked on its existing skipped queue, not mislabeled RUNNING or PREEMPTED.

A grant preview is not an allocation reservation. Unexpected allocation failure
can strand budget for one attempt; the cursor advances even then to avoid
retrying an impossible head forever. This first cut does not promise strict
work conservation, equal token/time shares, bounded waits under resource
pressure, or admission while all execution seats remain occupied. The follow-up
[issue6](https://github.com/vLLM-HUST/BetterScale/issues/6) owns aging, capacity
windows and HBM-holding-time tradeoffs. No such advanced policy is implemented.

`probe_native_round_robin.py` executes the FULL pinned native schedule with CPU
request/allocation/completion doubles (not device execution). Under4096 tokens,
long143971 + cached short29399/26624 + decode3, old scheduling leaves short
waiting for all four observed steps. The new schedule grants short2775 on step2,
long1318 and decode3 in the same wave; subsequent waves preserve decode service.
This complements, rather than replaces, native-page/State and NPU qualification.


## Current qualification and remaining gate

`qualification.json` records **24 passed CPU tests**, the full-native schedule
fixture, and successful real Ascend platform-loading preflight. That preflight
observes AsyncScheduler -> BalanceScheduler -> core Scheduler and resolves the
source-pinned disabled forwarding path; unknown/enabled variants fail closed.

The retained real-model attempts are not a passing serving result:
- hw3 model01: selected-card physical activity appeared after admission without
  visible process rows; startup memory gate rejected before weights.
- local model02: redundant controller FIA preload broke CANN environment
  re-source before server Python. Removed only that controller preload.
- local model03:32 FULL graphs captured, then the original method guard rejected
  the Ascend wrapper. This exposed the platform-loading gap above.
- local model04: corrected preflight passed, but an external SIGTERM reached the
  admission supervisor during warmup, before any HTTP request. Signal sender is
  unknown; cleanup completed with server exit0 and both cards at idle baseline.
  Do not reinterpret the resulting cancellation stack as a scheduler failure.

No real-request correctness, scheduler-step overlap, throughput or TTFT gate is
claimed. The prepared bounded controller checks serial/C16 retrieval, exact hot
continuation, and actual grants to that short request while a143971-token cold
request remains prefilling. Resume only after clarifying the external stop and
fresh shared-resource admission; do not automatically retry into another task.
The final policy files match model04's frozen capsule (apart from its trace
observer). Source adaptation, MTP-row indivisibility and startup traps are also
preserved in the repository knowledge scenario.
