# Recover the expert-service implementation, not just its last recipe

Use before changing Qwen35 batching or calling DFC-style execution missing.
Read `expert-profile-handoff.md` for profile identity and local artifact root.

## Exact migration seam (observed 2026-09-24)

Legacy BetterScale worktree `../expert-serving`, relative to this worktree, owns
`prototypes/attention-client/v2lite/build_persistent.py`. Its CLI defaults
`--sources-per-wave 7`, with1 as an explicit experiment. Sourcec8d6c55 introduced
the qualified DFC-aligned optimized recipe; sourcebf32672 adds the external
watchdog. MOD migrationbe475d3 copied the emitted **cap1 recipe**, not the builder's
capabilities. At the initial audit, the MOD CLI had no cap selector and Accept had a hardcoded1.

A CPU-only emit of the legacy Qwen35 recipe with cap7, batched activation,
pipelined copy, combined return,4096rows and the external watchdog produced
11 native files byte-identical to installed5 (including GEMM, streaming GEMM,
Cube, movement, activation, combine and client). Only persistent_vector.cpp
changed: the admission-count guards and increment implementing cap1. These are
40-layer source templates; serving adds physical draft layer40 during emission.
`lineage-cap7-vs-mod.{json,diff}` and `lineage-cap7/source` in the profile capsule
retain the actual comparison. This is source evidence, not cap7 qualification.

## Earlier work that exists locally

Under the legacy `prototypes/attention-client/` tree:

- `device-service/persistent_vector.cpp::Accept` and post-FETCH admission collect
  compatible ready sources, then freeze Group before Cube sees it. No forced
  wait-to-fill. `device-service/PERSISTENT.md` records actual two-source natural
  coalescing and delayed-source controls in an expert-sharded BF16 engine.
- `device-service/streaming_gmm.hpp` and the packaged equivalent use real group
  counts, rotating cross-expert core assignment, small-M weight-L2 bypass,
  CATLASS preload, and prefix/tail C/V handoff. They are DFC-aligned mechanisms,
  not proof of matched DFC latency. Current MOD still executes streaming mode2.
- `v2lite/bf16_move.hpp`, `bf16_activate.hpp`, `bf16_combine.hpp` supply token-owned
  fanout, pipelined DMA, batched activation and fixed-order whole-layer combine;
  these survived migration. Old per-row fine-pack polling is disabled on the
  qualified pipelined-copy path because its measured cost was bad.
- `qwen38/native_service/` demonstrates opportunistic same-layer multi-client
  batching (+30–44% in selected homogeneous cases, mixed case regressed), but
  uses host control/native W8A8 graphs, not this BF16 persistent MOD.
- `qwen38/route_stream/` demonstrates client metadata and tile-ready ingress/up
  overlap; no complete activation/down/return or persistent multi-client closure.
  Broad shapes improved, hot shapes regressed, and client plan cost remains.
  These two directories are uncommitted foreign work: preserve them, do not
  silently move, commit or import them into the MOD.

## Shared engine versus topology-specific completion

Both layer-owner placement and expert-sharded placement can share ready-source
admission, real local-expert counts, CATLASS compute and C/V pipelines. Batched
rows must keep (layer,expert) identity; current SINGLE_LAYER wave admission
requires same-layer compatible sources. Independent sources need not synchronize
at a global barrier to batch already-ready work.

Whole-layer ownership can combine every route in canonical order on the owner.
EP distributes a token's top-k contributions across owners, so return placement,
fixed-order reduction and all-reader retirement cannot be blindly identical.
The pre-migration staged EP2/EP4 candidate changed ownership/maps/client collect,
retained cap1, disabled early return and had NO device acceptance. Its receipt
expression `early_return=not combined_return` falsely implied early return for EP
despite cfg22=0. The migration below fixes that expression; hardware acceptance
remains outstanding.

Do not reintroduce forced batch barriers or declare cap7 universally faster.
The V2Lite cap1 result was genuine but workload-specific; it is not authority to
permanently disable batching for Qwen35. Recover the reusable capability and
validate bounded policy/complete-service behavior instead of rewriting GEMM or
promoting an unqualified W8A8 prototype wholesale.


## First MOD migration after Fletcher's instruction

The source-owned build now accepts `--sources-per-wave 1..7` for either placement,
binds it to `SOURCES_PER_WAVE` in the packaged protocol and the binary ABI, rejects
invalid ABI caps at admission, and reports the selected cap at drain. Default1
preserves the qualified control while cap>1 stays explicitly experimental.
The same Accept implementation serves both placements; no prototype import or
host per-wave dispatch is added. No forced batch delay is introduced.

`tests/test_expert_batching.py` compiles and executes the actual device Accept
body as host C++ at caps1/2/7, covering ready/late/incompatible/claimed/stale/EOF
sources and invalid metadata. It also exercises layer/EP2/EP4 source emission,
ABI rejection and actual configuration-derived early-return receipts. This is
logical admission validation, not visibility, DMA or numerical device evidence.
The existing full CPU suite passed134 tests before the extra receipt test; the
final focused suite passed3. Fresh CANN9.0.1 cap7 builds succeeded for layer,
EP2 and EP4, all41 physical layers; artifact directories are
`batch7-{layer,expert2,expert4}-v1` in the profile capsule. No NPU was used.

Hardware gates remain before promoting cap>1 or expert placement: actual
co-batch observations, changing full outputs, same-graph reuse, empty owners,
max4096/skew, multi-source retirement, exact-build model shadows and matched
service cost. Do not use the old cap1 or W8A8 receipts as those gates.


## Local three-card hardware follow-through (2026-09-24)

Fletcher restricted validation to2–3 local cards while benchmarks continued.
The queued selected-card job waited for foreign leases; no lock was replaced or
foreign process interrupted. Runtime source782c9e2, unchanged cap7 binaries from
the profile capsule, local real Qwen35 weights, layers0/40. Evidence root:
`runs/qwen35-expert-mtp/20260924-local-leaves`; read `README.md` and `results.json`.

- `ep2-v2`: physical client4/owners5,6,28 eager/FULL changing-route cases,
  max relativeL2 .0043242,30 generations per owner; includes empty owner and4096.
- `batch1-v1`/`batch7-v1`: physical clients4,5/owner6,432 total source frames;
  cap1 has432 waves, cap7 has326 (106 actual double-source waves). Both clients'
  complete changed-input outputs pass, worst relativeL2 .0045712. Same-layer
  real-weight hot8/broad mutations and FULL replay, no host per-wave scheduler.
- `ep2-multi-v2`: two clients share physical4, owners5/6.29 cases/client,
  nonuniform BF16 probabilities and an independent broad4096 FULL burst.
  Each owner completes126 source frames, in94/95 waves (32/31 paired).
  Max relativeL2 .0043969. This is three-card protocol coverage, NOT four-card
  attention throughput or a proof that colocated clients have no interference.

All accepted admissions/roles exit0 and exact generations agree. Final30s probe
shows4/5/6 IDLE, with no owned worker/queue left. Complete model/MTP-state,
EP4 and performance gates remain; defaults still cap1. Fewer waves is not a
measured speedup. `ep2-multi-v1` was a fixture startup failure (missing owner
launches), not a kernel failure; preserved evidence and CPU launch-order regression
precede the successful v2. The original30-minute queue expiry launched no worker.

## Whole-model and SWE cap comparison on hw3 (2026-09-24)

hw0 was reclaimed; do not reconnect or schedule there. Fletcher accepts local
and hw3 as equivalent8x910B2/HCCS for comparisons, retaining host provenance.
Evidence: workspace `runs/qwen35-expert-mtp/20260924-hw3-e2e`, remote
`/workspace/my-ascend-workspace/runs/qwen35-expert-mtp-hw3-20260924`.
Runtime is clean archivedd1d67fa source-tree execution, not a new installed-wheel
qualification; all22 donor source pins match. Reused qualified hw3 model/runtime.

Layer placement cap7 whole-model gates passed A4E4 and A6E2:128 mixed-length
requests each, realMTP2/C64,164/246 exact-build native layer shadows,
max relativeL2 .000240644/.000245942, all eight roles exit0. These are bounded
execution/numerical gates, not semantic SWE-solving quality or EP qualification.

Four SWE Prefix Reuse C64/900s points then passed with zero failed requests,
realMTP2, query4096/max-seqs32/KV32GiB per chip/native FULL decode, same prepared
corpus and seed20260924. `analyze-swe.py` verifies client validity, exact owner
retirement generations, direct resident launch counts, all exits and release.

| Layer placement | cap | Output tok/s/chip | TTFT p95 seconds | Lifetime frames/waves |
|---|---:|---:|---:|---:|
| A4E4 |1|141.932639|4.522088|1238880/1238880|
| A4E4 |7|143.928750|4.832207|1262106/1242972|
| A6E2 |1|134.023889|2.832773|1834422/1834422|
| A6E2 |7|138.175972|2.792350|1833162/1639981|

Use `swe-comparison.json` for exact numbers; table values are rounded. Cap7
observed throughput improvements about1.4%/3.1%, not a dramatic missing-engine
recovery. A4 TTFT worsened; A6 improved slightly. Lifetime coalescing reduced
wave count relative to its own source frames by1.52%/10.54%. These counters
include startup/drain, not only the900s window; they are not per-wave latency
or matched-work cost. Single closed-loop samples reach different turn mixes;
no significance claim or default promotion follows. Keep defaultcap1 until
broader evidence supports a policy. Historical hw0 DP8EP8/TP8EP8 remain labeled
historical controls; this round did not rerun them or publish the leaderboard.

Migration pitfalls: the first launcher missed `idle_gate.py`, failing on CPU
before admission; copy the supervisor dependency closure. The first SWE name
made AF_UNIX control paths too long; it failed before attention startup. Use
short IDs (`a4e4-c1-swe2`, etc.), check encoded socket-path length before load,
and retain those failures rather than attributing them to kernel numerics.

For the subsequent communication/serial-routing investigation and native client
plan's prototype versus MOD acceptance boundary, read [expert-transport.md](expert-transport.md).
