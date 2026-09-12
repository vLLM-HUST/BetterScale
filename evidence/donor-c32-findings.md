# Donor C32 / cache diagnosis — completed bounded observation

## Conclusions

1. **The old severe C32 collapse did not reproduce.** Do not retain 173.93
   tok/s / 38.86s TTFT-p95 as a stable donor C32 baseline or the basis of a
   general LiveInfer win.
2. C32 has a poor latency/throughput exchange in the available observations,
   but “no throughput improvement” is too categorical: the unmodified repeat
   reaches 266.64 tok/s versus original C16's 255.15 (+4.5%), while TTFT-p95
   rises from 4.23 to 9.14s. Single trials do not establish confidence bounds.
3. Native instrumented C16/C32 records show **no preemptions** and identical
   computed prompt work. C32 reaches 96.9% KV usage but is not continuously
   full. Replicated KV is a structural cost, not a demonstrated cause of the
   old collapse or repeated recomputation in this run.
4. **Prefix reuse is the strongest directly grounded optimization candidate.**
   Real native counters confirm zero hits in this cohort, but an exact 20K
   repeat hits 16,384 tokens: caching is not universally broken.
5. The captured C32 window exposes near-8K-global prefill/mixed execution,
   outside target's FULL decode coverage. FULL mixed/prefill graph is a
   plausible experiment, not a quantified speedup already established here.

## Results (same 96 calls / 22,321 committed outputs)

| Cell | Output tok/s | TTFT p95, s | Chunk gap p99, s |
|---|---:|---:|---:|
| Original C16 | 255.15 | 4.226 | 0.540 |
| Original C32 | 173.93 | 38.860 | 5.892 |
| C16 + native stat logger | 258.58 | 4.230 | 0.630 |
| C32 + native stat logger | 234.75 | 9.793 | 0.684 |
| C32 original entry repeated, no logger | 266.64 | 9.143 | 0.673 |

The original-entry repeat matches old entry bytes, plan, configuration and
capture-alignment adapter. Do not infer that the entire 234.75→266.64 gap is
logger overhead: asynchronous scheduling and generated histories may vary;
there is no interleaved repeated causal control. Likewise, the cause of the
old 173.93 outlier remains unestablished. The profiled cell is excluded here.

## What C32 actually schedules

Native scheduler snapshots, time-weighted by holding each until the next:

| Quantity | C16 | C32 |
|---|---:|---:|
| Average running requests | 12.37 | 17.64 |
| Average waiting requests | 0.34 | 1.70 |
| Time with >16 running | 0% | 57.7% |
| Time with KV usage >90% | 0.6% | 6.1% |
| Maximum KV usage | 90.27% | 96.87% |
| Observed preemptions | 0 | 0 |
| Draft acceptance | 71.23% | 66.55% |
| Computed prompt tokens | 664,196 | 664,196 |

C32 really increases active work, not only queue length. Yet global prefill
budget remains 8K and the profiled mixed/prefill wave already approaches it.
More admitted requests need not enlarge an already-full prefill batch. More
waiting, weaker speculative acceptance and end-of-cohort drain coexist with
that saturation. These are observed contributing conditions, **not a complete
causal allocation of every extra millisecond**. “Running” also includes
prefilling requests; it is not an all-decode batch count.

## Cache evidence and the smallest candidate

All 64 reusable turns have >=4K history; only one has >=8K and none has >=16K.
Total potentially reusable history is 355,583 tokens (logical opportunity,
not a measured counterfactual speedup). Native prefix hits and prompt-cache
source counters are both zero. The configured compressed cache's effective
block sizes are multiplied by compression ratio, and the hybrid coordinator
uses their LCM; a 128-row C128 block corresponds to 16,384 original tokens.

The exact-repeat controls give:

- 4,096-token prompt: zero hits on both calls.
- 20,480-token prompt: first zero, repeat **16,384 cached tokens**; completion
  1.864→0.941s for the fixed 16-output control.

A 4K negative control by itself is not a proof of 16K alignment: longest prefix
lookup may exclude the last prompt token. The source alignment and the full
cohort history distribution are essential context. Cache eviction/checkpoint
constraints also matter; not every theoretically reusable token is recoverable.

Before redesigning persistent KV placement, a candidate is the already
supported `block_size=32` configuration (with cache correctness checked at
8K/12K/20K repeats and then the unchanged cohort). It may reduce effective
C128 granularity to 4K, but **has not been run or adopted in this task**. This
is a configuration experiment recommendation, not a promised cache hit rate.

## Timeline and FULL graph boundary

Compressed eight-rank TraceLoom export (~18 MiB):
`/workspace/my-ascend-workspace/runs/donor-c32-diagnosis/20260912-c32-profile/analysis/donor-c32-tp8-end-aligned.json.gz`

The nominal 15s-after-arrival, 3s window is controlled by native worker RPC;
actual per-rank boundaries are in `engine/profile-window.json`. All ranks
match 1,725 collective family/group/ordinal markers. End-affine alignment is
for display, not calibrated cross-rank causality.

Rank0 observations: 3.670s compute-task span, 3.601s interval union including
communication, 1.602s non-HCCL task union. Host API counts include 804
`aten::item` calls (~312ms inclusive) and six `npu_fx_compiler inference`
scopes. SparseAttn inputs include 1008/1024 local query rows. Large totals
include AllGather, ReduceScatter, AllToAll and ScatterNdUpdateV2.

These durations overlap; CPU scopes nest; communication may wait. Do not sum
them as removable cost or assume all `item` calls synchronize the device.
Almost continuous task coverage does not establish useful hardware saturation,
especially with communication waits. A FULL mixed/prefill experiment should
measure actual end-to-end improvement, not just reduced Python call count.
Target decode is already FULL; draft is eager. No broader FULL integration
was implemented by this diagnosis.

## Artifacts / disposition

- `observations.json`: compact native counters, controls and repeat results.
- `PROTOCOL.md`: fixed protocol, capsule identities and hypothesis changes.
- `entry.py`, `observer.py`: native stat-logger / profiling-only worker extension.
- `summarize.py`, `analyze_timeline.py`: reduction and aligned export.
- Full raw capsules: `/workspace/my-ascend-workspace/runs/donor-c32-diagnosis/`.

All four cells completed their exact original budgets, no foreign-owner
observation, empty owned-process cleanup receipts. No installed donor source,
product default or public issue was changed. Performance figures are bounded
single-run observations; broader scheduler/capacity conclusions need repeated
paired measurements.

## Snapshot provenance

Copied from `CubeLander/my-ascend-workspace` commit
`9cb0f9b`, `prototypes/remote-kv-pull/donor-diagnosis/FINDINGS.md`.
The source scripts and protocol referenced above remain in that original
repository. Compact observations are copied here as `donor-c32-observations.json`.
Historical measurements describe the installed donor environment, not a newly
built validation of this repository's release-pinned submodules.
