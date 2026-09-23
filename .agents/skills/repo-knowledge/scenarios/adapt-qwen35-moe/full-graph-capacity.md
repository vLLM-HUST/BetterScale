# FULL graphs and usable KV: investigation, 2026-09-23

Use before trying to exceed the repaired native-matched24.25GiB/chip KV
budget. This is source/history synthesis, not a new NPU experiment or a
measured BetterScale-over-native capacity win. No runtime was changed.

## Reuse the LiveInference result, not its whole architecture

LiveInference checkout05ac1541 carries the relevant scenarios:
`.agents/skills/repo-knowledge/scenarios/qualify-runtime-memory-closure/GUIDE.md`
and `qualify-dfc-workspace-reuse/GUIDE.md`. Current source confirms the root's
unpublished calibration -> State fit/rebind -> retire calibration -> reclaim ->
final capture transaction (`src/livemodule/core/live_module.py:610`). Final
capture does not repeat ordinary eager warmups. `runtime/graph_memory.py`
attributes segments by device/pool/stream and refuses to credit private-pool
inactive blocks toward eager State allocations.

Accepted bounded precedent: DSV4 TP8 real weights, context1024/active1/prefill64,
240 residents/owner,2017 requests/286 waves, ordinary two-bank overlap, zero
READY-to-execution retained AND peak allocator growth on all ranks. Driver
free variation reached1,720,320bytes and remains a separate allowance, not a
universal bound. Qualification sourcec57dd84b; details in memory-closure guide.
This does not qualify Qwen256K, other portfolios or optional resources.

DFC EP2 workspace evidence: six graphs took248MiB reserved without effective
sharing and78MiB with the same native pool AND capture stream. A pool handle
with a different stream per graph still took248MiB. Capture-order/input-output
lifetime and generation retirement matter. A later wide-envelope capture-only
pool plus expandable segments cost20MiB more than an equally configured explicit
workspace control, not less. Warmup-only `MemPool` leaked old private segments
across generations on the inspected torch-npu version; do not transplant it.

## What BetterScale already has, and what it does not

Frozen runtime under workspace `runs/qwen35-moe-mtp-256k/`:
`local-candidate-runtime1/vllm_ascend/compilation/acl_graph.py` already passes the
global graph pool. Installed torch-npu2.10.0.post2 `npu/graphs.py` uses a shared
class-level default capture stream when none is explicit. Native uses this
machinery too. Adding another pool hint is not a newly available saving.
Core `gpu_model_runner.capture_model` already captures larger bins first and
trims ordinary cache before/after the portfolio. Blind extra `empty_cache()`
calls do not reclaim graph-owned scratch or holes in live ordinary segments.

FULL coverage means target/draft model bodies, not the whole serving wave:
- `model_runner_v1.py:2022` computes target logits after `_model_forward`;
- `sample_tokens` / `_sample` retain sampler/rejection work outside that graph;
- capsule `device_apc.py` retains graph-external preparation/postprocess tensors;
- banked ingress, persistent KV/GDN State, retained output/history and native
  communication resources remain separate lifetimes.
Do not delete all eager headroom or alias either bank's live State/outputs.
The draft-logits fix leaves later-step model token padding unchanged.

`_warmup_and_capture` still runs NONE eager warmup before every descriptor,
while earlier FULL graphs remain live; `draft_fia.Runnable` also primes its
exact FULL envelope before capture. KV has already been allocated by then.
Thus startup can require eager scratch alongside private graph residency even
when steady replay does not need that combination. LiveInference's calibration
transaction is the directly relevant lesson; transplanting it requires proving
KV/metadata rebind and final capture demand, not growing captured KV in place.

## Measurement boundaries and highest-value next discriminator

Repaired `request-sampling-full1` passed24 retrievals at24.25GiB/chip KV.
Its graph-capture free-memory delta is0.84GiB versus native's historical0.48;
neither is absolute graph-pool residency. Rank0 startup peak minus final is
348.777MiB allocated /436MiB reserved; final driver free is1.139GiB. These are
observations, NOT recoverable bytes or a measured maximum KV capacity. There
is no matched native whole-lifecycle allocator snapshot yet.

First compare native and repaired FULL at identical KV/TP2/C16/query4096/BF16/
MTP2 settings with a bounded diagnostic observer, not another throughput window:
load; eager warm; each capture boundary; READY; first/changed request; C16/mixed;
near256K; drain. Preserve normal overlap; reset execution peaks at READY without
inserting per-wave synchronization. Attribute actual pool/stream segments and
allocated/reserved/driver counters, rather than subtracting graph log numbers.
Exercise penalty/rejection and bank turnover, not only greedy retrieval.

Then choose by the observed owner:
1. Startup-only eager/private-pool coexistence -> calibrate with small KV, fit
   final KV and recapture without redundant warmups, with stable-address rules.
2. Serving-time external head/sampler/APC peaks -> reusable bounded buffers or
   separately captured islands with a serial scratch contract; retain ownership
   of cross-call outputs. One giant graph is not required.
3. Inactive ordinary-pool holes -> matched expandable-segment configuration
   experiment, not freeing graph-inactive blocks or introducing a custom allocator.

Do not add the entire eager activation peak again when the same storage is
already accounted for by retained graph scratch. Conversely, do not replace
all peaks by a max unless lifetimes/pool reuse prove mutual exclusion. Expose
an explicit native/driver allowance, whole-block rounding and loading feasibility.
Graph closure offers a more predictable, measurable non-KV bound; graph capture
alone guarantees neither smaller physical storage nor allocation-free serving.

Upstream invariant reference (CUDA, not an Ascend behavior qualification):
https://docs.pytorch.org/docs/2.10/notes/cuda.html#graph-memory-management
Stable addresses retain private pools; shared pools require nonconcurrent replay
and protection of outputs that remain live across graphs. Ascend applicability
here is supported by the installed source and retained EP2 evidence above.

## Matched diagnostic completed later that day

The discriminator above is now implemented and exercised, not just proposed.
Artifacts: workspace `runs/qwen35-moe-mtp-256k/memory-closure-profile1/`.
`source-identity.json`, both launch scripts, controller, pinned LiveInference
source archive, raw gzip allocator histories, rank phase receipts and
`summary.json` retain the exact experiment. Same local1/3 sequential pair,
TP2/BF16/MTP2/query4096/C16/context262144/KV24.25GiB, TASK_QUEUE_ENABLE=0
in both arms. Both allocator snapshots contain expandable segments; the pinned
Ascend platform enables them by default. Do not rerun “enable expandable
segments” as a new optimization, or borrow DFC's default-allocator saving here.

Each arm completed24 retrievals plus16 penalty-enabled128-output forced-length
requests (and a basic chat), using real MTP acceptance. Both actually observed
16 live requests. Native retained4 incorrect concurrent retrievals; FULL0.
This preserves the already-known native semantic limitation, not a new causal
quality comparison. Both server exits0 and30-second selected-card release IDLE.
Resource-tracker shutdown warnings remain in logs. No benchmark or maximum-KV
claim follows from diagnostic completion.

Rank0 measurements below are **MiB**, not GiB. Rank1 allocator results agree;
final driver free differs by less than4MiB. Checkpoints follow drained phases.

| Observation | Native | Repaired FULL |
| --- | ---: | ---: |
| READY allocated | 59377.01 | 59455.34 |
| READY reserved | 59776 | 59952 |
| Private graph pool residency | 326 | 366 |
| Reserved after cold/warm near256K | 60222 | 60118 |
| Reserved after penalty C16 | 60244 | 60320 |
| C16 penalty allocated peak | 59824.25 | 59801.59 |
| C16 penalty peak above retained allocation | 447.21 | 346.22 |
| Final driver free | 1044.60 | 783.03 |

Thus FULL saves104MiB of allocator residency after the long-context phase but
costs76MiB after penalty C16. Its transient allocated increment in that last
phase is about101MiB smaller, but extra persistent storage/private-pool/native
residency offsets the saving. FULL final device free is~262MiB lower;~186MiB
of that difference is outside allocator reserve and is **not yet attributed**
to graph executables, communication or driver variation. Source-level model
FULL coverage alone did not produce a whole-serving memory closure.

Private graph blocks report inactive/zero active bytes, but remain retained by
live graphs. Both graph pools stay flat through serving. Growth is in the
ordinary pool: READY -> penalty-end +468MiB native /+368MiB FULL. This directly
refutes treating all graph-external allocation as gone. Both arms' observed
startup allocator-reserved maximum is60388MiB during KV initialization, not
capture; do not infer actual OOM capacity solely from this historical reserved
peak (cached blocks may be reclaimable, and driver use has another scope).

### A concrete remaining FULL-only allocation

The bounded allocation history identifies repeated128MiB `torch.empty` calls
at `package/betterscale/patches/qwen_fia/wave.py:57`, inside
`Planner.native -> Frame.prepare`, including draft and target publication.
This is ordinary-pool scratch **outside replay**, not captured numerical FIA
scratch. `host_metadata.cpp::plan_native_queries` asks ACLNN for a plan;
`static_plan.cpp` intercepts and suppresses the selected numerical launch while
retaining metadata. The Python wrapper still allocates the full128MiB admission
ceiling for that host-planning call on each wave.

Counts of these allocations are NOT cumulative capacity loss: cached addresses
can be reused. Nor is128MiB yet a proved removable footprint. Never replace it
with a one-byte allocation or alias live output based solely on suppression of
one kernel: auxiliary ACLNN memory accesses and the passed workspace extent
still require proof. A narrow next intervention is an explicit metadata-only /
workspace-size contract at this native boundary, then a matched memory/correctness
probe. Merely retaining a permanent128MiB buffer removes allocation calls but
need not lower reserved HBM or improve capacity.

The same history also locates target top-k/top-p/rejection allocations in
`vllm_ascend/sample/sampler.py:263` (observed25.48/34.95MiB buffers), plus native
Mamba/draft preparation and FULL input preparation. These are actual graph-external
consumers, not reasons to deduct the entire eager profile twice. Prioritize this
observed planner/head/sampling tail before transplanting the more expensive
LiveInference fit/rebind architecture or inventing a new allocator.

## Reuse the profiling tools

Colocated helpers:
- `memory_profile_worker.py`: a diagnostic Worker mixin; imports the pinned
  LiveInference `TorchDeviceMemoryObserver`, `MemoryPhaseProfile` and exact
  `graph_pool_snapshot`, not a replacement allocator. Select
  `MEMORY_PROFILE_ARM=native|full` and put the matching admitted native_worker
  or candidate package, this directory, and pinned LiveInference `src` on
  PYTHONPATH. Use `--worker-cls memory_profile_worker.Worker` in a task-owned
  launcher. Do not use the mutable source of another live task.
- Startup load/profile/KV/dummy warmup/capture phases synchronize and reset
  peaks independently. Preserve the maximum across phase receipts; the last
  reset peak is not the whole startup maximum. Captures are named by mode and
  token envelope. Native records5 PIECEWISE and5 FULL envelopes here, versus
  FULL's32 bank/shape descriptors; do not compare these as physical kernel counts.
- Serving samples counters without synchronizing every wave. The bounded
  history keeps5000 recent events, not an exhaustive serving allocation trace.
  Raw snapshots retain current segment/block ownership and Python call stacks.
- On the **loopback-only development server**, the controller calls
  `/collective_rpc` with `method=memory_checkpoint`, `args=[label]`, only after
  requests drain. This saves waterlines/pools/history and resets the next phase's
  peak. `http-ready` separates API readiness from worker readiness. Never insert
  these synchronized checkpoints into a throughput window.
- `summarize_memory.py RUN_ROOT` emits bounded structured summaries. It counts
  native segments once per device/pool across streams; the observer's individual
  per-stream pool receipts must not be naively summed. Inspect retained and peak
  allocated/reserved plus driver free together. Report driver residual as an
  observation, not automatic process/component attribution.
- CPU accounting regressions:
  `python3 -m unittest discover -s .agents/skills/repo-knowledge/scenarios/adapt-qwen35-moe -p 'test_memory_summary.py'`.

Follow the workspace selected-device lease/admission/foreign-owner/release skill
before every NPU run. Captured source, native libraries, protocol/controller and
all logs remain frozen in the external capsule. This instrumentation is not a
new Worker product entry, performance release, or universal zero-allocation seal.
