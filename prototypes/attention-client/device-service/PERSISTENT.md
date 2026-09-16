# Work-conserving persistent expert service — implementation contract

Goal (2026-09-16): replace wait-to-coalesce with collection during useful work.
The initial contract is below; qualified evidence and remaining gaps follow.

Two independent sources, two expert servers; 910B2 BF16 Qwen H2048/M768,
128 experts split64/server, two layers, topk8,1–32 rows/source,24 jobs/source.
Same differential oracle as remote_dfc_control; no real-weight serving claim.

A server has two slots. AIV pulls unique input rows into a source-local staging
area, then checks mailboxes again at that completed AIV boundary. Ready new
sources can join before grouping. It never waits for an absent source to enlarge
a batch. Freeze the batch before publishing group counts and packed data to AIC.
Different (layer,expert) values remain separate groups.

One persistent vector team (coordinator plus movers) and one persistent cube
team communicate through generation-tagged cache-line-isolated commands and
per-core completion lines. AIC computes native CATLASS tiles with real row counts;
AIV performs live-row SwiGLU and return copies. While AIC computes, AIV may pull
and pack the other slot. No host per-wave decisions or finite unrolled GMM graph.

Original serial lifecycle: EMPTY -> PULL -> PACK -> READY_UP -> UP -> READY_ACT ->
ACT -> READY_DOWN -> DOWN -> READY_RETURN -> RETURN -> EMPTY. The optional
segmented successor uses independent compute-phase counters; see `SEGMENTED.md`.
Only the coordinator changes lifecycle. At most one vector command and one cube command in flight.
Stage completions join every participating core before publishing successors.
Sources are claimed until return writes finish; clients still wait for BOTH
servers before reusing input/output frames. Per-source generations prevent ABA.
Outputs in unowned expert slots remain unread.

Workspace/slot: raw[2,32,2048] BF16256KiB; packed[512,2048]2MiB;
up[512,1536]1.5MiB; activation[512,768]768KiB; down[512,2048]2MiB;
maps[2,264]INT32, groups[128]INT64. Two slots ~13MiB, independent of weights.
Worst-case all routes to one server is512: never use average256 as capacity.
Each copy is one aligned row, at most4096 bytes. No dynamic allocation in kernels.

Bounded inactivity/error watchdogs are still required; persistent means one
runtime invocation, not authorization for an immortal process. Invalid metadata
must terminate with a recorded status, never silently truncate. No polling core
may occupy the engine needed by its producer. Separate AIV/AIC admission is a
hardware gate, not an assumption from compilation.

Small-expert investigation: existing CATLASS already rotates starting cores
between expert groups (as DFC does). Do not claim that policy is newly missing.
Broad64-local-expert BF16 weights total576MiB/layer (384MiB up,192MiB down).
Compare matrix bytes, actual work and caching before attributing all cost to
scheduler overhead. More sources can amortize weights without reducing absolute
single-batch weight traffic.

Gates: leaf AIV/AIC handshake and real GEMM; changing-count/canary numerical
oracle; four-card changing-route joint protocol; staggered arrivals and zero-
local-work/skew; matched stage timings and compressed timeline. Default remains
unchanged until evidence warrants adoption.

## Implemented shape and qualified gates

The implementation uses **two coupled persistent kernels**, one AIV team and one
AIC team, launched once each through two captured graphs. It is not a single
mixed-engine binary, and it is no longer an unrolled graph of per-wave GMM ops.
There are17 vector blocks (one coordinator,16 movers),24 cube blocks, two slots.
Admission remains bounded to24 tasks/source. No production worker default changed.

AIV pulls unique source rows before expert-major local packing. After that DMA
completes it takes another mailbox snapshot; a newly ready source can join before
the catalog freezes. No sleep/extra-poll budget is used to enlarge a batch.
Already frozen batches are immutable; no claim of arbitrary late fusion is made.
A future source-depth/queue policy is separate from forcing sources to wait.

Evidence under /workspace/strengthen-dsv4/runs:

- `persistent-control-20260916T061905Z`: initial real-math engine-coexistence gate.
- `persistent-control-20260916T063029Z`: hot/broad/zero-local-work, one-row,
 512-route single-expert skew; independent random expert weights, output guards,
 input immutability and unowned-output poison preserved. Invalid layer terminates
 with device error; absent publication exits through bounded watchdog.
- `remote-dfc-control-20260916T062025Z`: four-card,48 changing-input outputs;
 24 paired waves/server, without an intentional batching wait.
- `remote-dfc-control-20260916T062255Z`: source1 delayed0.4ms;48 outputs pass,
 48 single-source waves. Coordinator-observed preparation/cube overlap totals
 1619.86us and708.68us/server. Those are command envelopes, not exact hardware
 instruction overlap; optional per-core timestamps provide a stricter view.
- Full native attention/KV model acceptance is NOT newly established here.
 The client wire/retirement contract is reused, but this turn's four-card oracle
 uses synthetic neural inputs, explicit routes and independent BF16 arithmetic.

Numerical gate: rtol.02/atol2e-5 and relativeL2<.01 for four-card controls.
Observed changing-input burst maximum relativeL2 was0.000293, not bitwise equality.

## Matched native-NZ / persistent burst: do not hide phase sensitivity

Same local2,3,5,7; same changing inputs, alternating two layers,32rows/source,
24 jobs/source. No forced-pair mode, no coalescing poll budget, no profiler.
Both begin their timed client episode AFTER initialization acknowledgement
(`DEVICE_SERVICE_BURST_PREQUEUE=0`). Report slower source's episode, not a
serving throughput number.

| Order/run | Native NZ | Persistent | Natural paired waves/server |
| --- | ---: | ---: | --- |
| A/B062743 /062811 |25.062ms|19.160ms|native0; persistent24|
| B/A062957 /063024 |24.977ms|24.408ms|native0; persistent0|

The first pair is23.6% shorter; the reversed pair is only2.3% shorter.
**There is no stable24% claim.** With only two one-outstanding sources, arrival
phase determines whether weights can be amortized across sources. The unpaired
persistent run still overlaps preparation with cube work, but that alone does
not remove most matrix time.

## Small experts and the remaining DFC gap

The pinned BF16 DFC tile is L1(128,256,256), L0(128,256,64), exactly the initial
CATLASS geometry. Both rotate the starting core across expert groups and disable
weight L2 caching for a single-M-tile expert. Missing either feature is not our
explanation.64 active local experts require576MiB of BF16 up/down weights per
layer; isolated, repeatedly warmed down GEMM is not the same cache workload as
alternating up/down and alternating layers.

A legal wider-N trial, L1(64,512,128)/L0(64,512,32), passed all14 leaf cases in
`actual-gmm-control-20260916T062538Z`, but broad gate/up remained~280us and down
~71us. No useful gain established; default geometry retained. Initial128×512
choices were rejected by explicit L0B/L0C resource assertions, not run on NPU.
The new build knobs are experimental, not auto-tuning or a selected optimization.

Fresh DFC on the same expert devices5,7:
`dfc-ep2-control-20260916T063400Z`.
Persistent independent-source stage:
`remote-dfc-control-20260916T063333Z`.

| Route / rows per source | EP2 DFC | Persistent client stage |
| --- | ---: | ---: |
| broad /1 |~281us|~255us|
| broad /16 |~416us|~694us|
| broad /32 |~419us|~750us|
| hot8 /1 |~274us|~251us|
| hot8 /16 |~286us|~250us|
| hot8 /32 |~317us|~281us|

DFC uses synchronized two-source EP waves and two devices; separated service
uses four devices and independently arriving tasks. This is stage diagnosis,
NOT equal-resource throughput. The broad-route gap remains, and this persistent
version is not a universal improvement over the earlier native-GMM service.
Keep the opt-in path; do not replace the shipped worker or claim DFC parity.

## Reproduce and inspect

Build `bash prototypes/attention-client/device-service/build_persistent.sh`.
It freezes the compile source closure and records tile options and binary hashes.
Use `PERSISTENT_BUILD=$PWD/runs/attention-persistent-build`.
Leaf: `bash prototypes/attention-client/device-service/run_persistent.sh 1`.
Four-card: export DEVICE_SERVICE_PARALLEL=1, DEVICE_SERVICE_PERSISTENT=1,
DEVICE_SERVICE_BURST=1, DEVICE_SERVICE_BURST_PREQUEUE=0, and point
DEVICE_SERVICE_SOURCE_BUILD at the qualified queue/client binary; run_remote_dfc.sh
takes the four idle device IDs. Do not set a coalescing or forced-pair gate.

Set DEVICE_SERVICE_PROFILE=1 for native msprof, and
DEVICE_SERVICE_INTERNAL_TIMING=1 for per-core work timestamps.
`persistent_analyze.py RUN` audits each slot's stage order and exports a compressed
per-device-relative phase timeline; it is not a distributed clock fit.
`profile_export.py RUN` supplies the separate native TraceLoom provider-clock
view. Neither raw JSON nor process pointers belong in the tracked result.

The first leaf attempt launched outside the graph's internal capture stream
and captured no tasks; fixed by using current_stream INSIDE torch.npu.graph.
The first non-burst run063204 failed a harness KeyError for disabled timing,
then its orphaned server wait hit runtime failure. The harness now enables its
required timing before launching. These rejected runs are not acceptance.

## Stricter overlap evidence and heterogeneous arrivals

The final instrumented run
`remote-dfc-control-20260916T063751Z` uses32 tokens/source0 and1 token/source1,
with a2ms host delay before source1 submission. Both still complete24 changing
jobs; no artificial wait-to-batch is enabled. Server0 forms40 waves (8 paired),
server1 forms35 (13 paired). Independent servers need not make identical batches.

Per-core timestamps bracket the actual transfer/activation and matrix routines,
excluding mailbox polling. Each core owns a64-byte timestamp line; the analyzer
joins exact command generations and verifies every interval lies inside its
coordinator-observed command interval. Unioned preparation/matrix-routine overlap
is239.70us/server0 and146.42us/server1. These are routine intervals, not direct
memory-bus transactions or a claim that all transfer cost is hidden.

`analysis/persistent-work-relative.json.gz` shows these per-core intervals;
`analysis/persistent-phases-relative.json.gz` shows coordinator phase envelopes.
Each server has a separate local clock origin. The separate native TraceLoom
`analysis/attention2-expert2-provider-clock.json.gz` retains provider timestamps.
Do not align internal clocks across servers by simply zeroing their first event.

The instrumented homogeneous0.4ms-delay run063512 formed24 paired waves/server
and showed **zero** preparation/matrix overlap: with both sources waiting for
the same batch, no next input exists to prepare. Retain this counterexample.
Outstanding work and natural arrival phase, not just available double buffers,
determine pipeline utilization.

Remaining scope: more independent lanes/queue depth, trained weights and full
attention/KV/model correctness, production cancellation and replenishment, and
equal-resource serving throughput. The small-expert/fused-DFC performance gap is
still open. None of those follow automatically from this bounded protocol pass.

For the opt-in intra-wave two-segment implementation and its **negative net-speedup**
control, read [SEGMENTED.md](SEGMENTED.md). Real phase overlap is qualified; it does
not justify changing the unsegmented default.

The next opt-in [internal publication prototype](INTERNAL-PIPELINE.md) keeps one
CATLASS up tile object alive across the prefix boundary. Its half-cut shows a
small positive matched-control result; tail-two still misses the consumer window.
Read its scoped evidence before equating either variant with full DFC fusion.

For interruptible-at-boundary pull/pack, read [YIELDING-MOVES.md](YIELDING-MOVES.md).
It reduces measured mover overlap with activation waiting but loses net episode
performance; it remains an explicit diagnostic, not the default or a full ready-queue design.

For descriptor/cursor retention across urgent AIV work, see
[RESIDENT-MOVES.md](RESIDENT-MOVES.md). Real in-command suspension/resumption is
qualified, but reversed controls do not show a stable net throughput benefit.
