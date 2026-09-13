# Worker submission versus scheduler lookahead

## Observed source contract, September 13

Pinned vLLM already has two concurrent batches for async PP1. EngineCore
`step_with_batch_queue` submits work before retiring the oldest future.
`multiproc_executor.WorkerProc.worker_busy_loop`, however, invokes each RPC
method to completion before dequeuing another. `non_block=True` at the engine
does not itself enqueue a device graph. Async output processing already has
a separate thread; do not invent another output worker to solve this gap.
The prototype `N2Scheduler` supplies FIFO/deferred-KV-free guards, not a new
two-wave queue. Restoring its bundle would also restore unwanted large draft
padding unless explicitly separated.

The retained LiveInfer reference is `AscendDSV4WaveExecutor.submit` in
`/root/my-ascend-workspace/livemodule-main-integration/src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py`.
It constructs and retains per-invocation pinned inputs, then queues:

1. Ingress: wait for that bank's previous graph reader, copy inputs, shadow
   construction, record ready.
2. Compute: wait ready and that bank's previous output copy, replay the whole
   device wave, record graph done.
3. Egress: wait graph done, copy banked output, record copy done.

`submit` returns without completing those device waits. `receive_oldest`
separately waits copy done and retires the invocation. The graph encompasses
continuation and target/draft computation, not merely the target neural network.
Events are recorded into producer streams before dependent waits are submitted;
this is not a protocol that waits on a never-recorded placeholder event.

## Measured worker issue gap, not event blocking

Reuse run098 native/metadata official databases. The bounded helper
`prototypes/full-mixed/profile_tools/worker_submission.py WINDOW --first-wave 3
--last-wave 13` checks CPU range containment, identical request/query geometry,
and exactly one target replay per forward. CANN event waits are restricted to
the worker's thread. No clock alignment is required for these local durations.
Receipts are each window's `analysis/worker-submission.json`.

For the metadata profile's eleven 16-request/96-query waves, rank5 medians:

| CPU interval | ms |
| --- | ---: |
| Previous eager draft runnable | 50.236 |
| Draft return to sample RPC return | 0.891 |
| Sample return to next target-step entry | 0.263 |
| Target-step entry to target replay API start | 5.239 |
| Worker event waits across these intervals, combined | 0.009 |

Across all eight ranks the last two issue intervals are0.248–0.265ms and
4.833–5.351ms. Native target issue remains19.345–25.682ms under the same profile
phase definitions. These are **profiled wall times**, not pure CPU execution
or additive removable device idle. Draft dispatch overlaps target computation;
do not claim removing50ms of serving time. Command enqueue/dequeue timestamps
are not recorded: the0.263ms interval includes RPC/wrapper overhead, not a
network transit or queue-residence measurement. Linux preemption remains
unproven without scheduling evidence.

Together with `continuation.py`'s late replay/short last-arriver ReduceScatter,
this rules out substantial worker event blocking as the dominant explanation
for these rank5 steady intervals. It does not establish the cause of every
rank's host skew, or qualify a composed split-draft run.

## Transfer boundary / next decision

The producer already snapshots CPU budgets on independent H2D storage and uses
device waits for destination reuse. Numerical accepted counts, sampled IDs and
draft IDs remain single-copy device State ordered on the compute stream. Keep
that ordering. CPU pinned-source overwrite still requires DMA completion or a
new retained source; blindly replacing `.synchronize()` with a device wait is
unsafe and will not fix a measured9us wait problem.

The native runner still returns through target preparation, sampling and draft
Python paths before processing the next command. First qualify the already
staged split-draft composition (run102 TP2 dummy passed; run103 real TP8 denied
admission). Then attribute residual target preparation and sampling/DSpark
metadata dispatch before deciding whether a larger fixed device program is
worthwhile. Do not add another scheduler queue or a concurrent thread that
mutates the same runner as a substitute for isolating host authorization from
device State and graph-owned inputs.

Native `AsyncGPUModelRunnerOutput` queues its copy-stream wait only after draft
submission. Earlier target-output D2H might overlap draft, but an early token
receipt must NOT authorize KV reuse while draft still accesses it. That is a
separate output-ready versus wave-retired contract, not a free reordering.

Acceptance for a continuous worker route is per-rank next-wave replay submitted
before previous device work drains, with generation/KV/input/output lifetime
gates intact. Two metadata banks or a two-entry engine queue alone are not that
evidence. Main, packaged defaults and runtime behavior are unchanged by this
audit.

## First executable worker cut

`--worker-continuous` now composes the producer, captured target metadata and
kept split-draft with `prototypes/full-mixed/worker_submission.py`. This is an
opt-in TP/DSACP prototype, not a new scheduler or a second compute stream.
Same-stream order already enforces numerical target/sampler/draft dependencies;
do not insert redundant waits merely to draw an event edge.

The cross-step callback can now be handed to the worker instead of run inside
target forward. The worker first queues sampling, draft and native async output,
records the compute tail, then applies the prior CPU correction. Two pinned
count-receipt slots plus per-publication events keep that old receipt distinct
from the new counts. The old callback reads its captured count receipt, not the
runner's now-current one. Exact GPU progress and KV are NOT double-buffered.
Special sampling/prefill/turnover keep native earlier retirement; command
reentrancy, a changed request map or a failure poisons the invocation instead
of continuing with a possibly wrong generation.

Run104 closes real TP8 split-draft composition: all eight ranks pass48 exact
target/full-KV checks and12 producer/sampler checks; all96 retained draft-bank
receipts have an initial exact check and no fallback. Reserved57.264GiB includes
the oracle snapshots and is not a serving-memory result.

Run106 closes the first worker-cut TP2 dummy gate:57 completed wave submissions,
22 whole-wave-deferred corrections on each rank,30 exact target/full-KV checks,
12 producer checks and16 checked draft banks total. The two count buffers cost
64 pinned bytes per rank, no additional numerical State. Run105 failed before
model initialization because a remote-only dummy-config path was used locally;
106 used the local checkpoint config with dummy loading. Both leases released.
All54 CPU tests pass. Real TP8 worker qualification and matched performance are
still separate gates; the implementation must not yet be called gap-free.
