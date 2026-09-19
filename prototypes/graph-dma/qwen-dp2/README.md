# Full Qwen prefill with graph-external bulk DMA

This is an isolated two-card native Qwen3-30B-A3B/DP2-TP1-EP2 experiment,
not a BetterScale serving patch or a claim of solving DP skew. Only local0/1
are requested under the existing home lease; no8-card or hw3 allocation.
The model comes from `/data/shared_models/Qwen3-30B-A3B` with real BF16 weights.

## Fixed-work protocol

Two native offline clients each submit one4096-token prompt. Native FULL
capture has buckets1/4096, one seat/engine,2GiB KV/rank, no speculation/APC.
After native preparation and one real forward, the worker freezes that exact
input, attention metadata and graph entry. Repeated native forward calls use the same complete
48-layer graph, including attention and EP; scheduler and sampler are excluded from the timed forward. Native attention
parameter-update/event publication remains part of the invocation.
No manual shadow graph is built; native FULL must actually be selected.

One independent stream issues `aclrtMemcpyAsync` OUTSIDE capture:
256MiB,1GiB and4GiB; H2D, D2H and same-device D2D. H2D/D2H use4GiB of
`aclrtMallocHost` backing per rank; device source/destination buffers each4GiB.
This is contiguous bulk traffic, not sparse KV gathering, remote peer HCCS,
network RDMA, or a proof of the physical transfer-engine identity from a trace.

For each size/direction, test rank0-only traffic and both-ranks traffic, with
three alternating-order trials of compute alone, copy alone, serial, overlap.
Both ranks always replay the model together, preserving native EP ordering.
The CPU barrier runs before timing; fork/join events order main and DMA streams.
Overlap issues DMA before replay so a long transfer can span the whole model.
The final join waits for both; buffers are never freed or overwritten in flight.

Record compute event span, copy span, joint completion and per-rank outcomes.
Compare model output against a frozen same-state ordinary replay (record exact
match; fail if beyond rtol=.001/atol=.001). Check all KV backing bytes after each
size/active-rank group: repeated prefill at the same positions must be idempotent.
Copy checks sample each4KiB page and final byte; full copy-byte correctness is
separately covered by the earlier local DMA oracle. No checks are inside timing.

The same-state replay is deliberately not an online workload, original eager
reference, accuracy benchmark, or state migration implementation. Rank0-only
traffic measures how a slowed peer can affect an EP partner, not offloaded
attention balancing. `TASK_QUEUE_ENABLE=0` prevents raw ACL calls bypassing an
independent Python submission queue; it is an experimental setting, not a
recommended serving configuration.

## Resource / artifact boundary

Raw capsule: `/workspace/strengthen-dsv4/runs/qwen-dma-local-20260914/first`.
The frozen `source` directory precedes other imports; do not mutate a live
capsule to repair code. Native launch uses the existing CANN9.0.1/torch-npu2.10
runtime without installing packages or editing donor. Admission waits are
bounded and fail closed; the supervisor owns descendant cleanup.

Output: rank manifests, line-flushed per-trial measurements, explicit per-rank
KV/output gates, client completion tokens and whole-job completion receipt.
An exit without both per-rank gates is not a successful interference result.
Do not use profiled timings for speedup or multiply one-card bandwidth by8.

First fixture correction: direct `entry.aclgraph.replay()` stalls because native
Qwen FULL attention consumes/resets ExternalEvents. Each invocation must also
run `_update_full_graph_params_if_needed`; frozen tensor values do not eliminate
this signal protocol. The first attempt was stopped by its owned supervisor,
without a successful measurement. V2 repeats the original `_model_forward`
call with frozen args/context, retaining native parameter-update/replay ordering.
Consequently host issue and native metadata-update delays within the measured
event span remain part of the practical forward cost. No donor hooks are changed.
A separate four-forward native profile follows the unprofiled matrix (compute,
H2D, D2H, localD2D at1GiB), to inspect actual task types and overlap. Profiled
forwards never contribute to timing medians.

## Completed bulk run (second capsule)

The native FULL envelope passes on both ranks:216 measurements/rank, all outputs
exact (no tolerance exceptions used), all KV backing bytes unchanged after each
case, one real completion per client. Measured KV backing2,139,095,040bytes/rank;
with reference snapshots and8GiB synthetic device buffers, allocated~45.0GB.
Raw `second/measurements` retains all trials; `bulk-result.json` reports max-rank
event durations per trial before taking medians (not a cross-rank clock span).

One representative result: baseline~287ms; rank0-only4GiB H2D gives288.0ms
joint completion. Both-rank4GiB H2D gives314.8ms median, but its three joint
trials span290.7–323.0ms. Both-rank4GiB D2H spans287.6–314.7ms (median309.2).
The host-transfer variation is material and non-monotonic; do not turn a median
into a zero-interference guarantee or attribute the variation to one cause.
Other cards on this shared host are not isolated from host-memory activity.

Both-rank4GiB local D2D is stable at289.5ms joint,~1% compute slowdown, but the
copy itself lasts only6.6ms. It does NOT establish sustained D2D interference
throughout a287ms prefill. The next `sustained` capsule queues16 or64 copies of
the same4GiB buffers (64/256GiB traffic/rank), both ranks, with the same controls
and native protocol. It is intentional repeated-address bandwidth stress, not
64/256GiB of unique resident KV or one enormous allocation.

## Native profile

Separate four-forward profiling passes. Both ranks contain three explicit
`aclrtMemcpyAsync` calls on independent stream36: H2D, D2H, localD2D. At1GiB,
H2D/D2H become16 native memcpy tasks (consistent with64MiB transport chunks),
whereas local D2D is one task lasting~1.7ms. The API calls themselves take only
~16–85us in this profile; do not blame long unprofiled copy spans on API CPU
work without matching evidence. These tasks are native MEMCPY_ASYNC, not a
custom AIV gathering kernel; exact hardware engine/PMU attribution is separate.

TraceLoom37323af analyzed both native DBs. Display-only affine alignment uses
unique identical provider collective identities and passes the unchanged50us
gate: holdout P95~0.477us, drift~0.049ppm. Compressed timeline:
`runs/qwen-dma-local-20260914/second/measurements/analysis/qwen-dma-dp2-end-aligned.json.gz`
(~1.2MiB). The four profiled waves are compute-only, then1GiB H2D/D2H/localD2D;
never substitute these profiler timings for unprofiled medians.

## Sustained D2D result

The follow-up passes both ranks:24 measurements/rank, all output tensors exact,
all KV backing bytes exact. `sustained-result.json` retains all three trials.

| Per-rank traffic, both ranks | Compute-only | Serial compute+copy | Overlap joint | Compute slowdown |
|---|---:|---:|---:|---:|
|16×4GiB =64GiB|287.1ms|389.0ms|344.1ms|19.8%|
|64×4GiB =256GiB|287.2ms|694.6ms|509.5ms|77.1%|

Copy-only spans101.9/407.7ms; concurrent copy spans~112/443ms (slower rank).
Thus the large-copy route has genuine resource competition, not just a small
setup charge. Joint completion still improves~11.5%/~26.7% over serial, but an
online prefill pays a substantial latency penalty. Do not call that a free
state service. Physical cause (memory bandwidth, transfer-engine scheduling,
existing EP/metadata traffic) is not uniquely attributed by this measurement.

The bulk host-transfer result and sustained local-D2D result are different
traffic regimes. One4GiB local copy finishes in~6.6ms; repeated copies occupy a
large fraction of the forward. Host DMA is far slower and stayed active for
~200ms at4GiB, with substantially lower observed prefill cost in this bounded
run. Do not extrapolate either result to cross-device HCCS or network RDMA.

## What this establishes

Graph-external bulk transfer can run alongside a full native prefill *including
its parameter-update protocol*, with exact model outputs/KV in these fixtures.
It supports exploring bounded asynchronous KV preparation. It does not yet
redistribute a request's attention or eliminate DP skew; ownership, metadata,
helper readiness and safe buffer reuse remain separate integration work.
The measured policy question is prefill latency cost versus bytes prepared,
not maximizing the copy engine's utilization regardless of service quality.

The sustained follow-up's separate two-forward profile also completes and
exports via the same frozen TraceLoom path. Holdout P95~1.894us, drift~-0.344ppm;
compressed file (~601KiB):
`runs/qwen-dma-local-20260914/sustained/measurements/analysis/qwen-dma-dp2-end-aligned.json.gz`.
It shows compute-only then64×4GiB local-copy overlap, not the unprofiled timing
matrix. Both completed capsules have released their selected-device leases and
owned processes. Other tasks on other devices were not modified.

## Sustained-DMA operator attribution

[Matched native operator analysis](OPERATOR-INTERFERENCE.zh-CN.md) separates the
profile pair from unprofiled timing medians. `operator_interference.py` verifies
matching shapes/dtypes/task types/counts and reports task union coverage. The
largest increases are attention and expert GEMM (~2.4x), then projections (~2x);
AllGather is nearly unchanged. Model-task gaps rise only ~2.4ms, not ~221ms.
This supports device-task interference, not a new large host submission bubble;
physical bandwidth-resource attribution still requires counters.

## Sustained host DMA (September 15)

[H2D/D2H operator attribution](HOST-DMA-INTERFERENCE.zh-CN.md) uses the same
real-model fixture on local physical6/7. `DMA_SUSTAINED=host` selects16/32GiB
per rank and separate32GiB profiles. Compute slows~9%, but attention/GEMM task
durations barely change; most added time is a first-attention metadata-readiness
gap, not stretched compute kernels. Keep the exact cause unproven. Neither the
full transfer volume nor its joint completion time is hidden by the prefill.
