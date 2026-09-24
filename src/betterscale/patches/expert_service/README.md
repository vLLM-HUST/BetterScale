# Persistent routed-expert execution closure (under integration)

This module is the source owner for the qualified Qwen35 BF16 persistent
client/server program. It must not import prototypes, reach into a task run
folder, or generate code by reading an unrelated experiment tree.

The initial native sources are the exact emitted closure of BetterScale
`bf32672`, `build_persistent.py --model-family qwen35 --event-capacity 512
--batch-activate --pipeline-copy --sources-per-wave 1 --combine-on-server
--frame-rows 4096 --lifetime-seconds 10800 --external-watchdog`, used by the
accepted 2026-09-24 C64 MTP0 matrix. The 801-line vector kernel remains together
because its coordinator/worker handshake is tightly coupled. Its arithmetic and
protocol have not been refactored while moving source ownership.

Native build depends on the selected CANN9.0.1/AscendC/CATLASS toolchain,
not installed prototype modules or another checkout. Build artifacts stay out
of Git. The explicit zero kernel deadline is only allowed behind a bounded
external supervisor; it does not authorize an unbounded serving launch.

**Status:** source-closure migration is in progress. The single Worker and physical
draft-layer paths are wired, but fresh hardware acceptance is not yet complete.
A package directory or CPU build test is not a serving qualification or release.

## Explicit experimental entry

Attention processes use the single `betterscale.worker.Worker`, with native
TP1/DP1/EP-off configuration and `additional_config.betterscale_experts`:

```json
{"experimental":true,"control":"/absolute/private/control","build":"/absolute/build",
 "owners":2,"sources":6,"source":0}
```

Each attention process has a different `source` and one visible device; expert
roles use separate devices. Build with
`python -m betterscale.patches.expert_service.build /absolute/new-build --draft-layers 1`.
Start each expert role via `python -m betterscale.patches.expert_service.server
--help`. The caller owns device admission, the shared/selected lease, a bounded
external watchdog and all-role cleanup. This package never steals a lease or
silently initializes extra devices. `BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG=1`
asserts that an actual external supervisor is present; it is not a watchdog.

No-MTP requires a 40-layer build; native MTP2 requires a 41-layer build. Target
layers use IDs0..39 and the one physical draft layer uses40, including its second
drafting step. Every layer has exactly one owner (`layer % owners`); all256
experts of that layer reside there. The checkpoint has **two fused tensors per
target layer but768 separate tensors in the draft layer**. Native loader hooks
skip both formats and retain only two2-byte CPU metadata parameters per layer.
Attention, shared MLP, routing and real acceptance remain native. The pinned
asynchronous Mamba feedback uses a private host mailbox so InputBatch row
changes cannot overwrite the previous-order state-selection receipt; see
`../qwen_mtp_feedback.py`. Native APC reset semantics are preserved.

By default each global layer is compared once with the donor routed FFN
(relative L2<=.02) during untimed startup. A subsequent deployment can supply
`qualification` pointing to that complete exact-build receipt to disable
shadows. MTP0 evidence cannot qualify a41-layer binary. Call `expert_receipt`
through native worker RPC to retrieve the proof. All attention ranks must call
`close_expert_service` **concurrently** before ordinary service termination:
servers wait for every registered source, verify completed generations, then
release IPC mappings. Worker shutdown invokes the same idempotent drain.

The initial port admits eager or native FULL_DECODE_ONLY with compiler mode0,
4096 query budget,<=32 requests and configured context<=262144. It does not
inherit the separate experimental FULL-prefill/GDN/FIA compiler stack or claim
its memory/performance results. Capture buffers include MTP verification rows,
not merely the live-request count. Qwen0.25 donor seams are pinned explicitly in
`expert_pins.json`; the older successful separated benchmarks used0.23.

## Acceptance boundary

- CPU contracts cover explicit admission, physical draft ownership, immutable
  emitted kernels, checkpoint names and pre-load/drain lifecycle ordering.
- The41-layer native closure builds independently with CANN9.0.1, one compiler
  at a time; this is **compile-only** evidence.
- Fresh real-weight shadows, MTP acceptance, changing-input state/replay,
  concurrent requests and clean device reclamation are required before this
  migration is called serving-qualified. Existing no-MTP throughput is not a
  performance claim about this new entry.

## Owned group lifecycle

The package also owns startup and concurrent drain; no prototype launcher is
needed for a real deployment:

```bash
# Under the caller's admitted-device lease and bounded external watchdog:
python -m betterscale serve-experts /absolute/model \
  --output /absolute/new-run --build /absolute/build \
  --devices 0,1,2,3,4,5,6,7 --sources 6 --owners 2 --mtp-tokens 2
```

This starts six independent native attention HTTP endpoints on loopback
ports32510..32515 and two expert owners. `ready` lists the endpoints;
`deployment.json` binds package location, commands, startup/final receipts,
generation counts and exits. A `stop` file under that run directory, or SIGINT
to the foreground launcher, drains all clients concurrently before terminating
HTTP services. Admission/foreign-owner guards remain the external host
supervisor's responsibility; a package launcher is not a device reservation.

The source fixes expert owners to **direct** launches of the two resident
kernels. Attention FULL decode graphs are unaffected. The first migration had
retained the prototype's default server-graph branch: short MTP gates exercised
that branch, not the older matrix's direct launch. It is removed rather than
relying on a deployment script to remember an environment override. Full Qwen35
placement admits E2/E4, not E1: the whole BF16 expert catalog plus service scratch
cannot fit one910B2. Independent workload/acceptance tests live in
`prototypes/expert-service-qualification/`; they consume this packaged group,
not private execution code.
The first bounded real-MTP gate now passes eight cold/warm/concurrent retrievals
with A1E2 and native FULL decode after the feedback ownership correction; the
unpatched native TP2 control reproduced the semantic failure and the repaired
control also passed. This is not yet a multi-attention-rank or high-concurrency
performance qualification. Details and retained failed controls live in the
`adapt-qwen35-moe` repo-knowledge scenario.


Installed-source acceptance now includes repaired real-MTP2 A4E4 and A6E2 with
C64 SWE exact-token continuation for900 measured seconds, query4096, max-seqs32,
32GiB KV/chip and native FULL decode. Both complete without protocol failures,
with verified prefix hits, direct resident owners, exact generation drains and
clean eight-role exits. This qualifies the bounded execution/lifecycle envelope,
not arbitrary model quality, full256K request coverage or peak throughput. See the
repo-knowledge adaptation scenario for artifact identities and native EP controls.


## Restored opportunistic batching (experimental)

The MOD now owns the legacy ready-source coalescing option for both `layer`
and `expert` placement; it no longer needs the prototype builder. For example:

```bash
python -m betterscale.patches.expert_service.build /absolute/new-build \
  --draft-layers 1 --placement layer --sources-per-wave 7
# Use --placement expert --owners 2 (or 4) for expert-sharded builds.
```

The cap is an integer1..7 bound into the source, binary ABI and server receipt.
Default1 retains the previously qualified control, not a universal scheduling
recommendation. At admission and after useful FETCH, the owner takes already-ready
same-layer/same-class sources up to the cap. It never waits to fill a batch;
Group freezes membership before expert computation. Generation, claimed-source,
priority/fairness and per-source lifetime rules remain unchanged. Seven is the
compiled source capacity, not a requirement for seven live attention ranks.

The compute engine, token fanout, DMA pipeline and batched activation are shared
across placements. Whole-layer return combines on the owner; EP preserves
route-valued return and client fixed-order reduction. Neither currently uses
early return; receipts read the actual configuration rather than inferring it
from return format. This does not restore the separate experimental online-return
or W8A8 route-stream paths.

CPU tests execute the actual Accept body with ready/late/absent sources, layer
and class incompatibility, claimed/stale/EOF generations and invalid metadata.
They establish admission logic only, not device visibility or GEMM correctness.
Cap>1 and expert placement require fresh multi-source changing-input, graph,
empty-owner/skew and exact-build shadow gates before performance/default promotion.

## Native client route preparation (opt-in, layer/cap1 only)

The source-owned build can enable a native routing plan above an explicit frame
size threshold; default0 keeps the existing wire and execution path:

```bash
python -m betterscale.patches.expert_service.build /absolute/new-build \
  --draft-layers 1 --route-plan-min-rows 1024
```

This experimental mode currently rejects expert-sharded placement and cap>1.
All registered attention ranks may still submit independently; cap1 limits one
admitted source per server wave, not deployment source count.

The client invokes `npu_moe_init_routing_v2` on a32-wide dummy hidden input and
its actual int32 route IDs. The resulting inverse route map and256 int64 counts
are packed into existing aligned IPC padding, after the maximum hidden payload.
Actual hidden is still sent only once. READY follows all metadata/payload writes.
The server validates original IDs/count totals, preserves the full expert catalog
and segmented boundary, and DMA-copies the prepared map rather than constructing
it route by route. Small frames retain the old path. Frame sizes, planner choice
and binary threshold are bound by the shared ABI, not environment heuristics.

Plan tensors remain alive through native enqueue; the original source-window,
DONE, retirement and drain ownership remain in force. No per-frame host RPC or
server-side host operator launch is introduced. Runtime source needs no prototype
module. Receipts include the selected threshold and bank memory including dummy
plan inputs. This is not communication/compute overlap or a public release.

The package-owned implementation passes the real-weight two-layer leaf and
141 CPU contracts; whole-model qualification is still pending. Track fresh acceptance in the
repo-knowledge expert-transport scenario before recommending the option.
