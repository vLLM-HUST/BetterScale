# Observe native cache actions on the device timeline

Enter after [native cache qualification](native-cache-maintenance.md), before
assuming asynchronous receipts imply kernel/DMA overlap. The measured first-use
counterexample below changes that interpretation; it is not an optimization.

## Reuse the capture

Capsule on workspace and hw3:
`runs/qwen35-state-lanes/20260927-cache-timeline/attempt1/`.
Production source is unchanged from5c591b4 (qualification driver57a8a66 plus
frozen profiling hooks). Same35B TP2/MTP2/FULL/native async,6GiB State budget,
E16/R20,512MiB host budget per rank. hw3 physical4/5, logical0/1.

`CACHE_PROFILE=1 CACHE_BYTE_AUDIT=0` enables the prototype's `native_profile.py`.
Native start/stop-profile RPC controls only already rank-bound workers. Level1
raw CANN, no stack/shapes/memory collection, native host MSTX markers at submit
begin/end and receipt-ready. No diagnostic CPU readback or added DMA barrier.
Capture starts after initialization/FULL capture/fill20/twin8/A8; measures B256,
store A, source overwrite C8, idle load A, unmoved hot and restored continuation;
then stops before independent cold oracle. First pinned-host allocation is
intentionally INCLUDED, not warmed away. Both-rank app oracle passed, exit0.

`analysis/` contains copied-provider native DB exports, TraceLoom augmented DBs,
full Perfetto outputs and provider-clock State annotations. Analyzer is pinned
ab8b5131191c6d5aeee2dd8566c34411f49ceab0 with qwen35-serving rules. `analyze.py`,
`summarize.py`, `annotate.py`, `plot.py`, protocol and raw hashes live with the
immutable capsule. Reuse/export retained raw data; do not recapture to fix a
viewer or annotation. Stock exporter omits MSTX labels; annotate.py inserts
those exact provider-clock intervals using the exporter's time_origin_ns.
Rank files remain separate; there is no fitted cross-rank clock alignment.

## What was actually observed

Each direction/rank has90 payload copy tasks and118,673,460 payload bytes.
Tasks join submission-window CANN memcpy APIs by connectionId, then TASK IDs;
source semantics identify D2H(store, stream32) and H2D(load, stream31).

| Phase | Rank0 | Rank1 |
|---|---:|---:|
| Store submit→rank receipt |23.689 ms|28.551 ms|
| Store device-copy envelope |6.158 ms|9.967 ms|
| Store main-thread host allocation APIs (83 calls) |11.358 ms|11.520 ms|
| Load submit→rank receipt |10.689 ms|10.258 ms|
| Load device-copy envelope |5.837 ms|5.009 ms|

No kernel-type TASK overlaps either D2H copy envelope on either rank. Prior
compute finishes about4.1–4.4ms after store submission begins; copies start
about17.3–17.4ms after submission begins. **Async completion works, but this
first-use D2H was not hidden by model computation.** The measured host allocation
cost and synchronous submission interval are real; their removal is not a
measured improvement. Copy envelopes include launch gaps and are not summed
DMA busy time. Rank1's envelope9.967ms contains4.254ms copy busy time.

H2D was deliberately issued with no model wave, so absence of overlap there is
expected, not an overlap-performance test. Subsequent measured model work is
first the unmoved control, then restored A; do not label the first post-load
kernel as restored-request execution. Overview compute unions exclude HCCL but
include small metadata kernels such as epoch initialization. White space on
that filtered lane alone is not proof of total device idle.

## Acceptance and limits

Both rank owners match raw PIDs and admitted devices. Native/derived DB checks,
positive task/API/graph populations,90 copies in each phase and normal-chat
hot/restored/cold exact outputs pass. Host/pending charge drains to0. Native
shutdown logged grace-period forced worker termination after profiler stop and
the successful cold oracle; app exit0 and a subsequent30s hardware window show
both selected cards back at driver baseline (3419–3421MiB), no residual workers.
Preserve that teardown detail rather than claim every child exited voluntarily.

Full timeline plus annotations/overview are duplicate views, not additive costs.
Host operation spans are not CPU-busy measurements. No throughput, steady-state
bandwidth, universal non-overlap, or unprofiled latency claim follows from this
single first-valid capture. The earlier post-EOS numerical counterexample is
unchanged and remains in the qualification note.
