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

The first bounded real-MTP gate now passes eight cold/warm/concurrent retrievals
with A1E2 and native FULL decode after the feedback ownership correction; the
unpatched native TP2 control reproduced the semantic failure and the repaired
control also passed. This is not yet a multi-attention-rank or high-concurrency
performance qualification. Details and retained failed controls live in the
`adapt-qwen35-moe` repo-knowledge scenario.
