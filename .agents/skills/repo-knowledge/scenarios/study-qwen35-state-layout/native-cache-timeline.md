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

## Continuous compute control (2026-09-27)

Do not infer DMA serialization from the first-use trace above. Fletcher asked
for preallocated copies while kernels actually keep running. Capsule:
`runs/qwen35-state-lanes/20260927-continuous-state-dma/attempt1/`.
Driver `prototypes/qwen35-cache-maintenance/continuous_dma_probe.py`, CPU-only
packet exporter `continuous_dma_declare.py`. Source baseline d74317f; no runtime
change. hw3 physical4, selected lease +30s admission/foreign-owner supervision,
exit0 and baseline release. Torch2.10/torch-npu2.10post2; seven shuffled timing
rounds and a separate Level1/MSTX profile, same pinned buffers throughout.

This is a **single-device hardware control**, not a native serving/TP2 run.
Actual observed TP2-per-rank grouped MLP shapes:192x2048 with256x2048x512,
SwiGLU,192x256 with256x256x2048. Synthetic random BF16 weights, one row each
for experts0..191. A captured64-block graph replayed4 times keeps computation
queued for~108ms. Prefix/gate/body are queued before waiting for the gate and
submitting DMA. No global synchronization/allocation inside the timed window.
Both directions use separate host/device buffers,90 packets118,673,460B each,
derived from actual State declarations (one resident plus16 kernel FA pages).
CPU numerical oracle, eager/graph exact comparison and every transferred byte
pass. Native/provider DB integrity and full exported JSON parse pass.

Unprofiled seven-round medians, milliseconds:

| Condition | Compute body | D2H event envelope | H2D event envelope |
|---|---:|---:|---:|
| compute only |107.738|—|—|
| D2H only |—|5.823|—|
| H2D only |—|—|5.417|
| both DMA only |—|6.516|6.372|
| compute + D2H |107.831|5.791|—|
| compute + H2D |108.087|—|5.459|
| compute + both |108.130|6.426|6.315|

Raw profile: kernel duty~98.82%;~98.85–98.88% of copy busy time intersects
actual kernels, and all copies fit within the compute envelope. In the both
condition D2H and H2D overlap2.766ms. They are not serialized, but host submits
all90 D2H calls before H2D, so their starts are staggered. This is not proof of
sustained fully simultaneous bidirectional peak bandwidth.

**Do not turn +0.36% whole-body time into zero local interference.** In the
single profile sample, GMM kernels touching DMA are+1.6%(D2H),+4.9%(H2D),
+3.9%(both) versus matching kernel ordinals in compute-only. These local numbers
are profiler-on observations, not statistically qualified penalties; the long
body dilutes short-window effects. The defensible finding is real compute/DMA
and directional overlap with modest observed whole-body cost in this setup,
not universal noninterference or end-to-end cache speedup. Event envelopes also
include submission gaps; never sum duplicated timeline projections.

`analysis/` retains raw-provider DBs, TraceLoom ab8b513 full annotated timeline,
small four-lane overview, plot and machine-readable summary. `analyze.py` and
`deliver.py` replay analysis without touching NPU. Profile teardown warns about
RECORD-state stop; all seven marked conditions have their expected771 kernels
or90 copies/direction, complete end markers and validated exported evidence.


## Aligned bidirectional bandwidth control (2026-09-28)

Fletcher asked whether demand H2D and background D2H should compete. Do not
infer direction independence from compute/DMA overlap. The follow-up capsule
`runs/qwen35-state-lanes/20260928-bidirectional-dma/` retains `probe.py`,
`packets.json`, protocol, all63 unprofiled trials, summary and admission/release.
Local910B2 physical2, same Torch2.10/torch-npu2.10post2/CANN9.0.1, shared host.
Other cards carried serving startup/foreign work; this is not isolated-host peak
bandwidth or a TP2 end-to-end scheduling result.

The same90 State packets (118,673,460B) repeat8 times per direction, using
independent preallocated buffers. Both DMA streams wait on ONE device event
behind >=200ms of queued grouped-MLP prefix. Every measured trial verifies that
ALL copies were submitted before gate release. Prefix work is excluded from
timings; compute-background trials queue another~207ms body after that gate.
Nine shuffled rounds; no profiler. CPU numerical oracle, graph/eager equality
and every transferred byte pass; supervisor exit0. Direction-start skew median
~0.14us, and almost the entire shorter DMA event interval overlaps the other.
This removes the historical D2H-first host-submission stagger.

Median GB/s, decimal payload/event-duration:

| Background | D2H alone | H2D alone | D2H with H2D | H2D with D2H |
|---|---:|---:|---:|---:|
| No timed compute |17.609|21.869|14.337|14.244|
| Continuous GMM |17.684|21.172|13.799|13.786|

With compute, standalone D2H53.687ms and H2D44.842ms become68.799/68.866ms
together: demand-load duration+53.6%, while combined makespan is shorter than
serializing both (~98.5ms). Compute body207.142ms alone versus208.380ms with
both (+0.60%) is diluted over the longer body, not a zero-interference claim.
These packet-stream rates include packet/event scheduling, not PCIe line rate.

**Inference:** simultaneous directions can improve aggregate throughput yet
hurt a latency-critical restore. This supports investigating demand-load
priority if parallel directions are enabled; it does not identify the contended
hardware resource or select an optimal scheduling policy. Current incremental
cache transactions remain globally single-flight; no new direction-priority
policy was introduced from this control.
