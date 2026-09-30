# A2E2 revival command board

Date: 2026-09-30. Branch: `codex/a2e2-revival-20260930`, based on
`archive/ae-final@d0ebe3a` (`397efcf` execution source).

## Objective and first acceptance

Restore and optimize Qwen3.5-35B-A3B BF16/MTP2 A2E2 without changing model
quality, token accounting or latency policy. The first performance gate is a
four-card A2E2 deployment matching or exceeding one two-card TP2 instance's
**total** output throughput under a matched SWE prefix-reuse protocol.

Report all three views:

1. A2E2 total output tokens/s and tokens/s/chip;
2. one TP2 replica total output tokens/s and tokens/s/chip;
3. two TP2 replicas on four cards as a separate reference, measured when a
   safe four-card matched run is warranted, never silently projected as data.

Historical orientation is not acceptance: final A6E2 produced 1,388.608 total
tokens/s (173.576/chip) under its 2026-09-24 C64 protocol. Latest independent
TP2 C16 produced 885.274 total (442.637/chip), and C32 1,227.760 total
(613.880/chip), under a different 2026-09-28 protocol.

## Matched protocol to freeze before performance claims

- same model snapshot, BF16, native MTP2 and natural acceptance;
- identical prepared SWE prefix-reuse workload and tokenizer fingerprint;
- exact generated-token continuation, cold client start and cache-reset policy;
- same global concurrency, query budget, context limit, max sequence policy,
  KV/state budgets, request routing and measurement window;
- output tokens observed inside the fixed window; report total and per chip;
- zero failed requests, exact generation drain, role exit0 and device release;
- report TTFT/ITL distributions and SLO-goodput thresholds without relaxing
  them to manufacture a win.

The first smoke uses C16 and a short predeclared window; no 900-second campaign
until correctness and causal measurements pass.

## Evidence chains

| ID | Claim / question | Smallest discriminator | Artifact / acceptance |
|---|---|---|---|
| H0 | Archived A2E2 lifecycle still works on pinned donor | CPU contracts, exact build, real-weight target/draft leaf | all contracts pass; changed-input L2 gate; exact generations; clean release |
| H1 | Cross-source ready work can form useful waves | two sources, cap1 versus cap7, matched arrival scripts and route populations | frames/waves/paired waves plus ready-to-admit and queue timing; no wait-to-fill |
| H2 | Single-Cube scheduling causes actionable HOL | delayed large source A plus already-ready short source B, serial control A/B/A | B READY-to-CCMD/DONE tail, Cube wait/active intervals and exact outputs |
| H3 | Shared expert hides part of routed wait | timestamp submit, shared part1/part2, collect/DONE on the executed A2E2 path | measured overlap and exposed wait; no claim from source order alone |
| H4 | A2E2 trend can reach TP2 total throughput | matched short C16 serving smoke after H0-H3 | total/per-chip throughput, TTFT/ITL, MTP counters, cache hits and correctness |

## Validation ladder and stop conditions

1. Static ABI/topology/source review and CPU tests.
2. Existing real-weight layer0/draft40 multi-source leaf; no full service.
3. Instrumented two-source queue/HOL leaf, with unchanged arithmetic.
4. Small real-model retrieval/MTP/lifecycle qualification.
5. Matched short C16 serving smoke.
6. Only after a credible trend and explicit resource plan: longer repeat or
   four-card dual-TP2 control.

Stop escalation on numerical mismatch, generation/slot ownership failure,
unclean release, foreign occupancy, unstable admission, or a lower gate that
does not support the current hypothesis. Kernel/synchronization changes require
an isolated commit/rollback point, full diff review and fresh lower-level gates.

## Current state

- Main checkout is dirty with unrelated user work and is out of scope.
- Isolated worktree created from clean archive; exact
  `vllm-ascend@9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` initialized.
- Baseline CPU suite: 20 passed, 15 warnings.
- hw2 inventory at intake: eight 910B2 cards idle, no NPU processes. This is an
  observation, not a reservation; admission must be repeated before each run.
- Shared expert is already attention-local. Historical TP2 traces show the
  default branch serialized; the donor multistream switch increased mixed AIV
  contention. A2E2 overlap must be observed, not assumed.

## 2026-09-30 staged gate receipts

- Final cap1/push real-weight leaf remains the rollback baseline. Stage7b used
  two sources plus one owner on devices0/1/2, layer0 and physical draft40,
  rows3 and one graph replay. All12 changed-input comparisons passed; maximum
  relative L2 was `0.004516422`, both client generations were10, owner
  completion was10/10, and the admitted job exited0 with all devices released.
- `Deployment` now accepts an explicit `--max-model-len` for a bounded smoke;
  the unchanged default and performance protocol remain262144. CPU tests prove
  a32768 override reaches both attention commands and reject values above the
  historical default.
- Stage6 did not establish a draft40 bug. Source1 first failed its262144/2GiB
  KV gate at06:42:23; partial-startup teardown stopped both owners around
  06:42:32; source0 only then reported draft40 L2=1.319 at06:42:48. Stage8 is
  the healthy-sibling discriminator.
- Stage8 artifact:
  `runs/a2e2-revival/20260930-stage8-a2e2-smoke`. A2E2/MTP2/final-plan/push,
  max-seqs4, explicit short-smoke max length32768 and2GiB KV. Both ranks reached
  READY with41/41 native shadows: maximum relative L2
  `0.00024064371245913208`, draft40 `0.00012095794954802841`, and no failure
  diagnostics. Concurrent real requests to both ports returned HTTP200, four
  completion tokens each and identical text. Final generations were owner0=160
  and owner1=140 for both sources; server receipts matched, used two direct
  resident kernels/owner, and all four roles exited0. Deployment and admission
  are PASS and the release receipt has no NPU processes.
- The Stage8 result confirms the Stage6 draft40 mismatch was teardown-derived,
  not a healthy full-serving numerical regression. It does not establish
  throughput: the short request elapsed times include first-request work.
- CPU expert regression after making the AST route-plan/EP mocks independent of
  real vLLM import order: `35 passed in 7.86s`. The earlier cold failure came
  from a fake `torch_npu` without `__spec__`; the later combined failure came
  from a partial real-vLLM import and duplicate Torch test-operator
  registration, not the route-plan implementation.

Next gate is a separately admitted, short matched C16 run. Before launching,
freeze the exact TP2 comparison context/token/cache/window/SLO contract and
increase A2E2 max-seqs, KV budget and graph capture envelope accordingly. Do not
reuse the Stage8 max-seqs4/32768 smoke as a performance result.

## Recovered frontier: Stage9 / Stage10 (2026-09-30)

The preceding “next gate” was the Stage8 checkpoint, not the latest state.
Stage9 completed C16/D1/60s cold SWE prefix-reuse with the same prepared workload,
natural MTP2 and exact generated-ID continuation. Both clients valid, zero failed
requests, complete drain. A2E2: **352.8 tok/s total, 88.2/chip**, TTFT p95 7.5056s;
TP2: **715.2333 total, 357.6167/chip**, TTFT p95 1.2155s. A2E2 all four roles
exit0 and owner/source generations agree. These are best-supported-system
configurations, NOT architecture-only A/B: A2E2 FDO/32GiB per A differs from TP2
FULL/LiveStateScheduler/full state cache/24.25GiB per rank. The immutable Stage9
`capsule/manifest.json` records that distinction (its old `prepared` label does
not override actual completed client/service receipts). Do not exclude cold
startup or replace the measurement window. Two TP2 replicas were not measured.

Stage3/4 already tested nonplanned cap2: 432 frames became328 waves versus432
for cap1. Three-row bracketing bursts did not establish repeatable latency gain;
start skew and run variation remain. Fewer waves is not a serving speedup.

Stage10 did not produce accepted profile evidence. Initial HTTP404, then an
AF_UNIX control path overflow, then two `no valid pid values` attach failures
are distinct harness failures. The latter launches omitted early
`PROFILING_MODE=dynamic`; changing host/container PID alone did not fix that.
Do not repeat the old launchers or treat these failures as expert-kernel results.

## Stage11: recovered profiling and first A2E2 cost evidence

Artifact root: workspace `runs/a2e2-revival/20260930-stage11-profile-recovery/`.
No serving algorithm, native binary, sampling policy or benchmark window changed.
`intake.diff` preserves the inherited worktree delta; `capsule/` owns the corrected
controller, known-good interactive msprof helper and direct native-DB analyzer.

1. Model-free attach discriminator on the intended local runtime: absent early
   dynamic initialization gives exit255; `PROFILING_MODE=dynamic` before imports
   plus the same-namespace worker PID gives acknowledged start/stop/exit0.
   Both workers exit0. Exported native DB integrity and199 compute tasks pass.
2. `p4` A2E2 retains Stage9 serving geometry. Sixteen fixed first-turn prompts,
   warm8 then diagnostic128 outputs each, separate salts; this is deliberately
   NOT an official SWE measurement or cold-start result. Capture starts only
   after both profiler acknowledgements and stops after the requests finish.
3. Two attention-owned captures/DBs pass PID/device ownership, positive TASK
   populations and integrity. All16 requests return exact128 output IDs,
   all four roles exit0, and owner/source final generations match
   (source0:1506/1360; source1:1484/1340). Selected0–3 release to idle over30s.
   hw3 had only three baseline cards, so the four-card diagnostic used local
   fallback with all-or-none selected leases, fresh30s admission and owner guard.
4. Full capture sessions, **no edge/start exclusion**:

| Rank | Capture s | Compute-kernel union s | Collect union s | Collect / compute union | Collect median µs |
|---|---:|---:|---:|---:|---:|
| attention0 |4.161944|3.313882|1.529266|46.15%|484.76|
| attention1 |4.167599|3.302547|1.522006|46.09%|496.50|

Collect occupies36.74%/36.52% of full capture time; denominators are distinct.
There are2268/2226 complete same-stream publication/collect pairs. Median
publish-end→collect-start intervals are36.38/36.82µs. Source places the local
shared MLP there: this is an observed opportunity to hide remote latency, NOT a
measurement of actual remote compute overlap. Noncollect compute union remains
about1.78s/rank; gaps without observed kernels are not proven hardware idle.

**Interpretation:** exposed remote-service completion is the largest individual
observed cost. Collect includes owner queue/compute/transport/local copy, not
network-only time. Persistent expert kernels predate dynamic attachment; this
capture intentionally does not claim their internal compute timeline. Native DB
model IDs are unavailable, so no exact graph membership or pure-decode split is
claimed. Do not extrapolate the diagnostic fractions to the cold SWE score,
subtract owner shutdown rings from this capture, or claim a speedup ceiling.

Next causal discriminator: instrument a bounded real-weight two-source leaf to
separate READY→admission, owner Cube service and completion, using same-device
intervals and exact generations. Reuse existing owner events where sufficient;
add only the missing READY/queue boundaries. Freeze both ready-short versus
large-source interference and a bracketing serial control before changing
admission or kernel scheduling. H2/H3 remain open; no900s campaign, cap2 default
promotion, dual-TP2 result, or public performance claim follows from Stage11.

## Fletcher's selected direction: finite EP waves (2026-09-30)

The next-discriminator suggestion above is superseded by Fletcher's new design:
stop refining the persistent server and implement a host-loop EP scheduler.
Primary clients rotate; only already-ready same-layer decode can hitchhike.
Prefill/mixed are singleton waves; cap is one maximum-prefill token width.
Gate precedes next PULL, which overlaps current DOWN; accepted prefill bubbles
need no priority mechanism. Finite PUSH can overlap next UP. Alternative client
PULL retains owner output slots until consumption ACK; no winner is selected.

Source contract and implementation boundary:
`src/betterscale/patches/expert_service/WAVE_PROTOCOL.md`, `wave_protocol.py`,
`wave_server.py`. Fifteen CPU tests pass, including NumPy matrix-oracle EP2/EP4,
target/draft identities, changing routes/generations, empty owners, both return
modes and independent owner progress. This is executable scheduling/lifetime
policy, **not a native device Backend or switched serving deployment**. No new
NPU job was launched for this protocol step. Preserve existing persistent runtime
and stage11 timelines as rollback/evidence rather than silently replacing them.

## Remote evacuation checkpoint (2026-09-30)

Backup branch: `codex/experimental-ep-waves-20260930` on
`git@github.com:vLLM-HUST/BetterScale.git`, forked from the revival worktree.
The commit message contains the handoff and resume gates. The original revival
branch and superproject pins are not advanced by this checkpoint.

The sibling `profile-recovery/` directory now preserves the Stage11 capture/
analysis source and compact Stage9/11 evidence previously located only in runs.
Those historical scripts require explicit path/runtime relocation and fresh
admission on a replacement host. Raw profiling DBs/timelines, model weights,
compiled native builds and full run logs are **not backed up by this Git commit**.
`fusion_result.json` is generated CANN output, not source, and is excluded.

For CPU wave tests, use Python with NumPy (validated here with Python3.11 and
NumPy2.4.6); production serving still uses its separately pinned Python3.12
runtime. Read `WAVE_PROTOCOL.md` before implementing the missing native Backend.
Do not resume persistent-scheduler tuning or AgentX instead of the accepted
finite-wave EP design. No PR, merge, package release, or new NPU run is part of
this evacuation.
