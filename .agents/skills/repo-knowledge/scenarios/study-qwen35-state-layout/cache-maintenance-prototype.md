# Scheduler-issued State cache maintenance (2026-09-27)

Enter before extending offload over SIMD lanes / seats. Fletcher authorized a
small Qwen3.5 TP1 prototype on hw3, not automatic policy or production 35B/TP2
rollout. Keep the accepted native State-only substitution boundary intact.

## Existing backend, new execution seam

`src/betterscale/live/runtime/host_state.py` was already inside BetterScale,
relocated from LiveInference, with logical-domain selection, candidate-span
lowering, contiguous page-run copies and host snapshot handles. Reuse it rather
than build another cache pool. It previously only supported CPU/CUDA and charged
completed snapshots, not pending stores. This cut adds NPU streams/pinned host
memory, preallocation budget reservation, copy-stream failure drain, and retained
buffers when drain cannot establish quiescence.

The explicitly optional integration is in the **small-model research**
`live/llm/qwen35/{scheduler,residents}.py`. It is NOT the native
`models/qwen35/{seat_scheduler,resident_leases}.py` adapter. Command/receipt policy
and the NPU worker live in `prototypes/qwen35-cache-maintenance/`; that README
owns invocation, source navigation and the supported boundary.

Accepted prototype invariants:

- One scheduler writer submits and reaps. The worker never edits residency.
- `(operation, seat, incarnation)` receipts authorize publication/recycling.
  Stable host keys are independent of device placement. I/O pins are separate
  from execution width and exclude both seat eviction and page reclamation.
- Store is allowed only for a completed, idle resident, after the final writer
  on the declared producer stream. Store completion publishes the host identity
  before source eviction. Merely enqueueing a store frees **zero** pages.
- Restore reserves shared FA pages and an entire seat before copying; prefix
  identity is published only on completion. Requests awaiting that prefix yield
  admission to unrelated requests. Page-pressure requests can wait for I/O
  without prematurely recomputing/preempting the cohort.
- Cancellation waits for the completion receipt. Unknown/stale completion keeps
  resources quarantined. This is fail-closed, not live device-error recovery.
- Maintenance futures can wake the driver with no model work. The probe uses an
  explicit host wait plus scheduler tick; native EngineCore wakeup plumbing is
  still future integration, not something this test has established.

## Bounded real-weight observation

Capsule, both local workspace and hw3:
`runs/qwen35-state-lanes/20260927-cache-maintenance/`.
The initial run is `admission1/`; `package/`, `maintenance.py`, `probe.py` freeze
its code. Final-source confirmation uses `final/`. Full logs remain outside Git.

Hardware: hw3 physical NPU0 / logical NPU0, Ascend910B2, selected-card lease
`/home/jingyuan/npu-device-0.lock`, 30s under-lease admission and continuing
foreign-owner guard. Do not reuse that card without fresh admission. New hw3
jobs follow the Sept27 selected-card policy, not the retired global-lock rule.

Model: real BF16 Qwen3.5-0.8B, snapshot
`2fc06364715b967f1860aea9cf38778875588b17`; copied weights/config verified on both
hosts (`model.sha256`). E2/R3/P128, page128, context8192, FULL prefill256, MTP2.
The normal research bootstrap still checks every `live_qwen_pins.json` donor
source/version gate; no bypass. Remote Python:
`runs/liveinfer-online/20260907-donor-dspark-runtime/env/bin/python`, with the
qualified `qwen35-moe-mtp-256k/{candidate-runtime1,runtime-source}` source overlay.

Both initial and final-source real-NPU runs passed (process exit0). Final
`result.json` preserves the complete receipt; the final copied implementation
was compared directly with the committed source. Observed results:

- Host payload **61,401,132 bytes**, all GDN candidates/conv, target + draft FA,
  and numerical continuation. Destination keeps its new resident epoch.
- A stored from seat0/pages[0,1]; C then overwrote the original seat; A restored
  to seat2/pages[4,5]. **70 selected views byte-identical** after relocation.
- Exact152-token hot hit; nine output IDs identical across restored MTP,
  cold MTP, and independent target-only/nonchunked prefill. MTP accepted5/6
  proposed draft tokens on the resumed request.
- Store was issued while an unrelated request was active. Maintenance-only
  restore completion and cancellation after store/load enqueue passed.
- All128 shared pages free at the final request boundary; host bytes zero
  after maintenance close. Probe process exit0.

**28 focused CPU tests passed.** They cover delayed/out-of-order identity,
no early free, cancellation,
shared-page pressure waiting, unrelated admission behind a pending prefix,
submission rollback, unknown DMA failure quarantine, span/offset relocation,
and in-flight host budget. Run the README's focused test command.

## Do not overgeneralize

The copy worker uses independent direction streams and no device-wide barrier.
However, this research root completes compute waves synchronously and still has
`clear_seat`/generation synchronizations. This establishes transport, logical
identity and scheduler ownership semantics—not overlap, performance, native
async scheduling, TP2 quorum, or 35B serving equivalence. The logged
`store_and_other_request_seconds` includes B's remaining computation and Python
work; it is **not a D2H bandwidth/latency benchmark**. No leaderboard update.

Before native integration, map the receipt to its processed-step frontier and
idle engine wakeup; replace its immediate eviction/retry assumption without
changing donor numerics, graphs or async execution. TP2 must join successful
completion for the same operation on every required rank before host/device
publication. Do not infer that from this TP1 run.

For the subsequent native AsyncScheduler/EngineCore and TP2 expansion, read
[native-cache-maintenance.md](native-cache-maintenance.md). The observations
above remain evidence for the original research route only.
