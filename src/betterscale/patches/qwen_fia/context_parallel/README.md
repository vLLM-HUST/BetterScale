# Quota-partitioned decode/verification attention

This is the BetterScale-main landing point for the split-capable,
partial-outlined research implementation, not its full-only diagnostic controls.
Source donor: `ascend-op-prof-single-decode` at `ea92f61`; numerical pipeline from
installed CANN9.0.1 FIA. The qualified Qwen35 entry now enables this path by default; other geometries remain native.

## First serving contract

- Qwen3.5-35B-A3B BF16, TP2, MTP2, C16, paged128, causal TND, native262144
  context. Rank-local full attention is Q8/KV1/head256, not the research
  Q28/KV4/head128 fixture. Up to16 real requests plus the existing zero-KV
  padding row. Preserve the already qualified request-bounded draft sampling.
- Apply to pure target decode/verification. Retain existing native attention
  for single-request capture and all prefill/mixed and draft execution initially.
  The existing verification capacity family (3,6,12,24,40,48) is disjoint from
  prefill/mixed capacities; never use an unkeyed traced Python shape branch.
- Preserve original-order contiguous quota scheduling, startup cost and split
  hysteresis. Whole contexts write final O; only split contexts write partial
  O/logsumexp and participate in reduction. No task-sort/modulo24 scheduler.
- Input Q/K/V, query endpoints, actual KV lengths, mask and page table are
  read-only. Output and scratch are invocation-local graph-pool allocations;
  plan metadata follows existing two-bank publication/consumption fences.
- In the qualified MTP service, actual KV lengths are device-authoritative.
  CPU lengths are optimistic planning envelopes, NOT the exact consumed lengths.
  Porting static microbenchmark descriptors without resolving this distinction
  is incorrect. No new host synchronization to read accepted counts/actual KV.
- Padding is not an extra live request. Prove valid-output coverage, tail
  behavior, partial scratch bounds and no reads outside actual valid KV.

## Qualification and comparison

1. Host partition/metadata coverage and workspace bounds for Q8/KV1/D256;
   nonuniform small query groups, zero-KV padding, all-full and skewed splits.
2. Fresh CANN build; NPU comparison to native attention and independent reference;
   changed inputs/lengths, two-bank repeated graph replay, guarded scratch/output.
   Include device-authored KV changes across512-token tile boundaries.
3. Use the Qwen35 Worker and pinned runtime from current main (c238443 base).
   Earlier frozen capsules are numerical references, not substitutes for current
   main qualification. Preserve unrelated work and immutable evidence capsules.
4. Real-acceptance functional C16/long-context continuation gate, then matched
   old-attention/new-attention C16 E2E, with other optimizations identical.
   Use equal KV budget (qualified24.25GiB/chip), query4096 and request capacity16.
   Match main’s frozen SWE-prefix-reuse C16/900s protocol, seed20260924,
   exact-token continuation, natural MTP2 and no forced acceptance. Both arms
   retain native aligned State; this does not qualify the resident-State route.
   No public benchmark submission or SWE task-success claim is implied.
5. Report throughput, TTFT/TPOT/tails, request/token errors and actual attention
   routing, including short-shape regressions. One pair cannot estimate run-to-run variability; do not select a favorable
   loaded instance or infer repeatability from this result.

All NPU work queues on hw3 as jingyuan with auto-selected per-card leases,
30s fresh admission and continuing foreign-owner supervision. No local NPU,
legacy-lock deletion, fixed-pair queueing or displacement of other tenants.

## Packaged default and source build

From the pinned CANN9.0.1 environment (CPU-only compilation):

```sh
python src/betterscale/patches/qwen_fia/context_parallel/build.py /path/to/new-build
```

The builder materializes installed vendor source into the new directory; it never
patches CANN in place. The kernel artifact is `libbs_fia_cp.so`; build completion
alone is not a device qualification. Do not replace a library under a running
process or reuse a build directory holding an active artifact.

For an explicitly selected task-owned source build, these overrides remain available:

```sh
export BETTERSCALE_CONTEXT_PARALLEL=1
export BETTERSCALE_CP_LIBRARY=/path/to/new-build/libbs_fia_cp.so
```

Qwen35 admission supplies the bundled library and enables this by default.
`BETTERSCALE_CONTEXT_PARALLEL=0` explicitly selects native attention for diagnostics.
The main-tree wave adapter selects the existing
verification capacity keys6/12/24/40/48, only for Q8/KV1 and17 request rows.
The flag is fixed before capture; changing it does not rewrite a captured graph.
Native/draft/single-request/prefill frames retain their original2528-byte metadata;
only selected target frames use4096 bytes. This does not add another graph family.

## Device length lowering

Host lengths choose ownership and reducer slots, never the actual consumed KV.
Each endpoint carries its planned tile-count denominator. The producer scales
both endpoints by `actual_tiles / planned_tiles` with integer floor, preserving
adjacent, exhaustive512-token tile coverage as device feedback changes lengths.
Static split ownership survives an empty or full-span scaled slice: empty slices
publish the reduction identity `(O=0, logsumexp=-inf)`, not a direct final output.
Padding is a separate zero-KV tail, not another schedulable request.

This preserves the existing device-feedback copy after bank publication; no host
readback or acceptance-count synchronization is introduced. An underestimated
host envelope also preserves coverage, but may produce a less balanced schedule.

Bounded leaf evidence on hw3: six shapes including16×256K, nonuniform Q1..3,
two captured banks and16 replays/case, changed inputs, fixed plans with device-only
length changes across512-token boundaries and down to3 tokens,13-query zero-KV
padding, input immutability and scratch/output guards. Maximum absolute difference
from native BF16 FIA was0.0009765625; short edge requests also passed an independent
FP32 CPU oracle. This alone is not a model or E2E performance result.

Startup capture is a distinct boundary: native mixed dummy partitioning makes
capacity40 `[2]*15+[10]` and capacity24 `[1]*15+[9]`, although real verification
rows never exceed3. Only while the owned capture bank has an empty real-request
pool, the target frame copies/clamps query lengths like the existing GDN capture
adapter and represents the remaining capacity as zero-KV padding. Shared layer
metadata and the original device-length source are not changed. Real requests
still fail closed if any query row exceeds3.

## Completed bounded E2E observation (2026-09-27)

Current-main C16/900s on hw3, one AB pair at Fletcher's requested compute budget:
output369.81→391.30 tokens/s/chip (+5.81%), median TPOT20.52→19.36ms (−5.68%).
TPOT P95 was nearly flat, P99 worsened4.47%, and TTFT P95 worsened2.23%.
Both arms had zero request failures/preemptions; real split execution was
observed on both ranks after the functional/long-context gates passed.
The slowest TPOT rows in both arms were the initial short-prefix burst, not
long-context rows. Keep those samples in the headline; the cause is not isolated.

This is useful E2E throughput evidence, not proof of universal tail improvement,
repeatability or C16 extreme256K-skew performance (SWE reached about90–93K).
This observation initially kept the feature opt-in; the subsequent combined
qualification and Fletcher's September27 decision promote it to the Qwen35 default.
The repo-knowledge `adapt-qwen35-moe/context-parallel-attention.md` and its
colocated `context-parallel-ab.json` retain protocol, source identity and limits.

A subsequent single combined point enabled resident State via the packaged
`serve-qwen --runtime live` configuration (E16/R20/LiveStateScheduler) together
with the same attention flag/library. Both paths passed the hot-cursor/long-split
gate; C16/900s measured442.98 tokens/s/chip, decode P9062.91, TPOT median17.39ms.
This exceeds historical resident-only414.80 by6.79%, but TTFT P95 worsened
0.462→0.589s. The historical point was on a different host and no new control
was run; it is combination evidence, not a paired incremental-speedup claim.
See the same knowledge note and `resident-balanced-c16.json` for exact scope.

Distribution assembly includes the exact qualified binary identified by
`native.json`; `setup.py` rejects a missing or mismatched payload. The source
builder remains a development tool, not an install-time compiler. A fresh build
is not automatically qualified merely because compilation succeeded.
