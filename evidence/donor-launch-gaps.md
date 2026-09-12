# Native donor launch gaps (2026-09-12)

Fletcher observed sparse task execution on rank 3 while other ranks looked
replay-like. Analysis below uses original native provider DB timestamps, NOT
collective-end-aligned cross-rank timestamps.

Reproduce (CPU-only):
```
python3 evidence/inspect_launch_gaps.py \
  /workspace/my-ascend-workspace/runs/donor-c32-diagnosis/20260912-c32-profile/engine/profile \
  evidence/donor-launch-gaps.json
```

## Observed

All eight ranks have 20,727 COMPUTE_TASK_INFO-linked tasks, each linked to a
CANN launch API through connectionId. Union these intervals with COMMUNICATION_OP
intervals across streams. Interior uncovered intervals >=50us:

| Rank | Window ms | Uncovered >=50us, ms |
|---|---:|---:|
|0|3670.2|51.3|
|1|3674.6|708.2|
|2|3680.1|57.1|
|3|3659.9|632.6|
|4|3673.2|56.6|
|5|3669.2|113.0|
|6|3698.6|592.8|
|7|3666.9|67.2|

Rank 3's sparse layout is real; ranks 1 and 6 also have substantial uncovered
intervals. This is not evidence that only rank 3 uses eager mode. Existing API
counts match across ranks; no graph execute/replay API was recorded in any rank.

Rank 3 has 2,479 gaps >=50us. For 2,395 (total gap duration 572.0ms), the next
compute task's launch API STARTS after the gap begins. This category is NOT an
estimate of host-removable time: launch may happen early within the gap, then
execution still waits on something else. 58 gaps (55.5ms) have that API already
finished before gap start; 26 overlap the launch interval (5.1ms).

## Two concrete rank-3 examples

Times below are relative to first rank-3 compute task (not combined viewer origin).

1. At +1546.16572ms, a 5.55316ms gap ends at HcPost. Its launch begins
   +5.458995ms after gap start. The PyTorch `_C_ascend::npu_hc_post` scope
   starts at +0.8161ms and lasts 0.2056ms; the CANN `aclnnHcPost` execution
   call on a different host thread starts only at +5.4453ms. Thus looking
   only at the frontend/Python call misses delayed asynchronous dispatch.
2. At +1556.84494ms, an 11.89392ms gap ends at AivKernel / ALLTOALL.
   On the launch thread, `aclrtStreamWaitEvent` spans from -2.5756ms
   to +11.7006ms (14.2762ms duration); `Dequeue@wait_event` spans the same
   interval. ALLTOALL launch then begins +11.7395ms. Meanwhile frontend
   dsa/moe scopes and GetWorkspaceSize calls continue on another thread.
   This is direct evidence of a launch-worker stall around wait_event, not
   simply Python being unable to prepare the next operator.

Other large gaps have kernels submitted well in advance, including a
25.89394ms gap with launch finished 43.838335ms before device execution.
A single explanation is insufficient.

## Interpretation and limits

The priority is the host asynchronous dispatch / event dependency chain,
not GEMM replacement. FULL capture could remove per-operator dispatch and
some host event handling, but no speedup is measured here. Correct device
stream dependencies may still be necessary inside a graph.

Uncovered means no selected compute or high-level communication interval;
it does not prove every engine (DMA etc.) was idle. Communication coverage
may itself include waiting. API duration can include blocking, descheduling
and profiler overhead; the trace cannot identify the cause of the long
wait_event API or a removable amount without further controlled measurement.
Native host/device mapping is accepted from the profiler, not independently
calibrated here. Different rank profile boundaries preclude naive exact
cross-rank subtraction. Never sum rank gap durations as wall-time savings.

Next discriminating work: recover the wait_event's matching producer and
queue order; assess whether the blocking is profiler-induced or persists
without profiling. Do not claim proven heterogeneous graph execution.
