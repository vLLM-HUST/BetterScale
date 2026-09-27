# Native async State cache actions (2026-09-27)

Continuation of [the TP1 research prototype](cache-maintenance-prototype.md).
Fletcher authorized native async integration first, then TP2. This is an explicit
maintenance execution interface, not automatic eviction, host LRU or prefetch.

## Ownership and execution

`models/qwen35/cache_actions.py` is scheduler-owned: submit pins the resident and
its shared FA pages; each rank must acknowledge `(operation, seat, epoch)` before
publication or reuse. Idle means both no request owner and final writer fence
processed. Cancellation records intent, never authorizes early reuse. Failed,
foreign, duplicate or stale receipts fail closed. Drop is irreversible.

`cache_worker.py` uses the existing `TorchHostStateBackend`, direction-specific
NPU streams and event completion threads. It copies declared numerical State,
including all candidate GDN rows and target/draft FA pages. Native scheduler
blocks lower to kernel pages (2048 -> sixteen128-token pages here). Placement
`resident_epoch` stays destination-owned. The CPU `_live_previous_verify` flag
must migrate too, before publishing the load receipt. Producer and MTP APC
completion events precede copies; no device-wide barrier is added.

`cache_engine.py` adds a private utility and Unix receipt inbox to the pinned
native EngineCoreProc. Receipt arrival enqueues native WAKEUP; scheduler-thread
reap resolves a utility Future even without another model wave. Native
`step_with_batch_queue`, model initialization, FULL graph, MTP and numerical
operators stay in place. `worker.py` submits actions before normal execute_model.

Opt in with `additional_config.state_cache_host_bytes` (per rank); supported
action protocol scope is DP1, TP1/TP2, one engine lifetime. The surrounding
qualified native model admission still requires35B TP2. EngineCore utility `state_cache`
accepts `snapshot`, `store`, `load`, `cancel`, `drop`, `wait`. This is not a
public HTTP API. Budget admission counts completed and in-flight host payloads;
canceled stores remain charged through host-drop quorum.

Native queue ordering and preemption policy remain unchanged. A pending prefix
restore can head-of-line block waiting requests; this implementation does not
inherit the research scheduler's skip-waiting admission policy. Transport/device
failure requires teardown rather than speculative pin release or live recovery.

## Evidence and continuation

Capsule: `runs/qwen35-state-lanes/20260927-native-cache-async/` on workspace/hw3.
`cpu-final.log`: **44 focused tests passed**, including real EngineCore idle
input-queue wakeup (CPU fake numerical backend), delayed rank quorum, cancellation,
capacity refusal, logical-page lowering and CPU verify-flag relocation.
`byte-admission.json` checks actual35B TP2 State declarations without activation:
95,604,788 resident payload bytes; 23,068,672 bytes per native2048-token FA block;
one-block checkpoint118,673,460 bytes per rank. Epoch8bytes are excluded.

`prototypes/qwen35-cache-maintenance/native_probe.py` runs real native AsyncLLM,
35B TP2/MTP2/FULL, E16/R20,6GiB total State budget,512MiB host budget per rank. It fills
seats, stores A while B runs, overwrites A's source with C, restores elsewhere
with no compute wave, and compares resumed output IDs against cold execution.
CPU tests are not numerical NPU qualification or performance evidence.

Initial `admission1/` never reached weights/cache execution: physical0/1 passed
30s admission, then native memory checks found insufficient HBM. After owned
cleanup they still used12,584/26,003MiB without visible PIDs. This matches the
workspace's `probe-ascend-npu/hw3-restarting-tenant.md`; do not lower memory
admission to fit unknown foreign work or repeat launches without new steering.
Fletcher subsequently assigned cards4–7. `candidate2/` freezes the newer source
(including scheduler host-byte admission and EngineCore source pin), constrains
selection to that set and retains independent admission/run evidence.

`candidate2/admission1/` admitted physical4/5 and loaded weights, but its probe
imported Worker/model classes before native platform registration. Warmup failed
with `UnquantizedFusedMoEMethod ... no attribute 'is_monolithic'`: the model had
captured the pre-Ascend MoE factory. `candidate3/` corrects the **probe bootstrap**
by calling native `current_platform.pre_register_and_update()` before Worker
imports, as the CLI does. No operator or donor-source workaround was added.

## Current numerical gate: RED, not qualified

`candidate3/admission1/` on physical4/5 reached real requests through native
async/FULL/MTP2. Both-rank store finished before B ended; C reused the source;
restore used another seat/pages and completed without increasing model_steps.
The resumed request reported185 cached tokens. **Its32 output IDs differed from
cold execution**, so the probe exited1. This is not a passing native cache result.
At this point attribution is unresolved: transport versus existing warm/cold
continuation versus the deliberately forced post-EOS fixture.

`candidate4/` adds an unmoved twin request and probe-only `native_audit.py`.
The audit compares each selected device view with its host payload bytewise
before sending store/load completion. It intentionally adds synchronous reads;
never use its timing as production overlap evidence. The fixture compares twin
initial IDs, then unmoved hot / relocated hot / cold continuations and persists
all three before asserting. Native production-source files are unchanged.

This diagnostic did **not** reach model execution: it admitted6/7, then the
foreign-owner guard observed external PID1658926 on7 and stopped only the owned
process group. Subsequently4/5 were occupied by unrelated `gdn-install` workers
(cwd `/home/jingyuan/gdn-install-20260927/admission-service2`). No foreign process
was killed. Fletcher was asked for a coordinated6/7 window; no automatic model
retry is queued. Continue from this red gate, not from the earlier TP1 PASS.

For another authorized attempt, freeze a new capsule and preserve old logs;
reuse candidate3 compile caches if desired, keep device selection within4–7,
and retain selected leases, fresh30s admission and continuing owner guard.
`candidate4`'s numerical/audit driver is ready but unexecuted. Native source has
44 passing CPU tests; the next useful evidence is the three-way/byte witness,
not another unmodified replay of the failed cold comparison.
