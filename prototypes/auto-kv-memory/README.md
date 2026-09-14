# Native automatic KV budget / FULL graph accounting

First measure the existing path, without changing the allocator. `memory_worker.py`
subclasses the released Worker only to record synchronized startup memory snapshots:
pre-profile, post-profile, after real KV allocation/before capture, and after all
native capture plus BetterScale startup preparation. It does not replace profile,
cache layout or graph capture. Startup synchronization means this is not a latency
benchmark. Record local device/DP identity when aggregating; TP rank alone is not
unique across DP engines.

Use the qualified native launch command with `--worker-cls memory_worker.MemoryWorker`,
remove `--kv-cache-memory-bytes`, and explicitly use `--gpu-memory-utilization 0.9`
for the first control. Keep the same context, seats, capture catalog and K5. Use an
admitted, descendant-supervised all-card launcher and run an HTTP smoke request.
Do not call this a larger-context, cache-hit or full-capacity stress test.

Source audit at Ascend9bf964cb/vLLM752a3a50:

- Ascend Worker profiles eager dummy work before assigning the KV budget;
  `compile_or_warm_up_model` captures graphs after KV initialization.
- ACLGraphWrapper already uses the global graph pool. Native capture orders
  larger buckets before smaller ones for pool reuse.
- The post-capture suggested manual KV size adds activation and graph memory.
  Whether both reservations are required depends on residual eager execution,
  sampler, draft and fallback lifetimes. Do not subtract one merely because
  target prefill is FULL.
- Native CUDA has `profile_cudagraph_memory`, but its temporary-cache initialization
  and graph-wrapper registry/cleanup are CUDA-specific. Ascend overrides KV
  initialization without that profiling signature and uses ACLGraphWrapper.
  Calling the CUDA helper directly is not an established Ascend solution.

September14 run164 failed before memory profiling: the initial admitted hw3 window
was lost to substantial device occupancy during startup. Workers reported free
memory below requested utilization. No CAPACITY_OBSERVATION was produced; this
is neither an automatic-KV OOM nor a measured graph-pool result. The supervised
job exited; remaining occupancy after its exit is not owned by this experiment.
Evidence is retained in `runs/auto-kv-20260914/result164/` (local, untracked).

Completion requires automatic startup and request execution, explicit accounting
of graph/preparation/draft pools and non-graph peaks, safe runtime headroom, then
actual cache occupancy under longer/concurrent requests. Only then update public
launch defaults; do not relabel existing fixed-budget E2E numbers as auto-budget.

## Active goal after Fletcher's September14 steering

Do not promote0.96 as a replacement default. The target is physical usable memory
minus measured execution residency/peak and explicit safety headroom, not another
fixed percentage. Remove the artificial15K/16K admission ceiling after validating
longer metadata/graph shapes. Reconcile the LiveInfer8GiB≈500K observation with
HBM versus mapped storage and per-request fixed-state overhead.

Run165 completed native automatic0.9 DP8 startup and a short HTTP completion:
KV4.505–4.506GiB/rank, eager-profile activation0.648GiB, post-capture allocated
53.994–53.995GiB/reserved55.000–55.020GiB/device-free5.079–5.107GiB. Run166
completed the0.96 accounting control: KV~8.16GiB/rank, post-capture free~1.50GiB,
three HTTP cohorts and all32 retrieval questions (pinned OpenCompass scorer100%).
Neither establishes long-context capacity or percentage-free production sizing.
No TP0.96 run was submitted. Both completed jobs have exited.

`runs/auto-kv-20260914/cache_math.py` reconstructs the DSV4 cache specs and calls
actual pinned Ascend grouping/pool functions on CPU. Its initial8GiB layout gives
174812 equivalent tokens at16K,482394 at64K,986645 at512K and1066457 at1Mi.
These are **constructed-layout projections**; the16K result differs from live
run163's179972, so validate actual layer names/specs/DSpark grouping before treating
those as real-engine capacities. The calculation includes per-request C4
compressor peak131 pages and C128 compressor38 pages at wave1026; it is not a
fixed bytes/token model. Preserve the discrepancy, not a calibrated guessed count.

Next evidence: capture the live group/spec census and allocator pool totals;
separate transient graph capture from persistent replay and eager fallback
lifetimes before designing a minimal automatic budget hook. Do not directly
reuse the CUDA-only temporary graph estimator. Long-context validation must
include real attention/indexer bounds rather than just a larger cache table.

Run168 advances the protocol: disposable target capture succeeds; after retirement
only~328KiB allocated growth remains. The initial cleanup retained GraphParams
singletons, so final native KV/backend initialization rejected a second set with
`Graph parameters have already been set!`. V2 explicitly retires the drained
singletons as well as catalogs; its CPU test protects ordering and idempotence.
Run169 is the corresponding NPU gate, not yet an accepted result here.

**Correction to the initial capacity inference:** the public command has no
explicit block-size, so the real170-spec census uses block32 (C4 state2,C128
state8), not historical benchmark block128. Common APC alignment is4K, not16K.
`capacity_census.py` now consumes that actual census and reproduces run163's
179972 exactly. Its public compact result is `docs/evidence/kv-capacity-20260914.json`.
At8GiB, single-request common-pool peak demand is4.210GiB at512K and7.793GiB at1Mi;
this is a KV ledger, not successful long-context execution. The original
constructed-layout results above remain a rejected starting inference.

## Recapture lifecycle failure (September14, run169 and two-card isolation)

Run169 V2 completes final capture and its short decode cohort, but large prefill
fails with AllGather AIV SDMA error507011, input8404992bytes (=1026×4096×BF16).
Post-capture free memory is about1.93GiB; this is not an observed allocation OOM.
The temporary allocator must NOT be adopted merely because allocated growth is
only~328KiB after trial retirement.

`recapture_probe.py` isolates the lifecycle without model/KV/attention on local
physical3/4, masked to logical0/1, installed torch-npu2.10.0.post2, CANN9.0.1,
HCCL_OP_EXPANSION_MODE=AIV, buffer256MiB. It primes HCCL, captures1026/12/6-row
BF164096-wide AllGathers into a disposable pool, retires it, allocates256MiB
surrogate backing, recaptures, and replays small then large shapes.

- `runs/auto-kv-recapture-20260914/job/`: exit1, same8404992-byte AllGather SDMA
  address error during final replay. No vLLM/model dependencies.
- `runs/auto-kv-recapture-retain-20260914/`: RETAIN_TRIAL=1 keeps old graphs AND
  their input/output tensors alive. Both ranks pass all six exact checks and
  exit0. This establishes a lifetime-sensitive failure, not which individual
  resource is stale. Retaining everything is a diagnostic, not a budget solution.
- `RETAIN_TRIAL=tensors` separates tensor backing from graph lifetime; record its
  result before changing production cleanup. Launch uses the same selected-card
  admission/lease and300-second owned timeout. Raw artifacts remain untracked.

The fixture allocates communication inputs/outputs outside capture. A passing
control does not prove real model graph-pool internals safe. Narrow the dependency
before another expensive eight-rank model run or claiming an upstream root cause.

The next controls completed on the same admitted physical3/4 pair:
`runs/auto-kv-recapture-tensors-20260914/` (keep external input/output tensors,
release graph objects) still fails507011; `runs/auto-kv-recapture-graphs-20260914/`
(keep graph objects, release external inputs/outputs) exits0 with six exact
checks/rank. These separate external tensor lifetime from graph-held resources.
They do not yet identify an internal HCCL allocation or prove an upstream bug.
`RETAIN_TRIAL=anchor` tests whether a prior, communication-only retained catalog
can protect later disposable captures; it is not integrated into the Worker.

The prior anchor catalog does NOT protect a subsequently destroyed catalog:
`runs/auto-kv-recapture-anchor-20260914/` fails507011 too. Do not describe the
condition as only a first-capture problem. Keeping trial graph handles and using
that same pool for final capture (`RETAIN_TRIAL=shared`) passes all six exact
checks/rank, exit0 in `runs/auto-kv-recapture-shared-20260914/`.

V3 `PreflightWorker` consequently keeps retired graph handles (never replayed),
releases their old State/packets, and reuses the final shared pool instead of a
disposable separate activation arena. CPU cleanup test covers handle retention
and State/packet retirement. This is still an experimental hypothesis pending
full-model allocated/reserved-memory and request validation; the raw two-rank
fixture alone does not establish production safety or final graph peak.

V3 full-model gate is submitted on hw3 as
`/workspace/my-ascend-workspace/runs/tp-continuation-20260914/170-hw3-dp8-physical-preflight-v3/`.
Frozen source/launch closure is sibling`auto-kv-v3/`; launcher PID2579909 at
submission (revalidate live identity, not this historical number, before waiting
or cancellation). It owns the home lease and bounded idle-card admission.
Control remains run166; no percentage-default or long-context guard has changed.

Run170 completed exit0: all three HTTP cohorts and32/32 pinned OpenCompass
retrieval pass. V3 budgets~7.9GiB KV/rank with1GiB explicit safety; after final
capture/free~1.69GiB, allocated~57.33GiB/reserved58.30GiB. Shared trial graph
handles leave only~329KiB allocated growth, while preserving communication
resource lifetime. Final capture reports only~0.35GiB *incremental* growth because
it reuses the trial pool; do NOT replace the full preflight reserve with this
incremental number. Compact exact eight-rank ranges:`docs/evidence/auto-kv-run170.json`.
Full locally recovered receipts:`runs/auto-kv-v3-20260914/result170/`.

The next gate is a separately scoped longer-context fixture (first64K, then
capacity occupancy), with an explicit experimental guard extension rather than
silently lifting public qualification. TP automatic sizing, actual long-history
runtime peaks, and the LiveInfer HBM8GiB/500K reconciliation remain open. Neither
the package's released allocator nor its public max-length guard changed here.

Next long-history gate submitted as hw3`171-hw3-dp8-physical-long64`, frozen
sibling`auto-kv-long64/` (launcher PID2590963 at submission). Only this private
capsule extends DP max-model-len to65536; production guard remains16K. Workload:
single32768-token prompt, single61440-token prompt, then eight61440-token prompts,
128 generated tokens each, followed by the same32 retrieval questions. Synthetic
repeated-token prompts exercise cache/history execution, NOT long-context quality
or the maximum admission/SLO concurrency. It retains the run170 physical budget
and shared-graph-lifetime protocol, all other launch settings unchanged.
