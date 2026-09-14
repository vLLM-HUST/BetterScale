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
