# Mixed GDN solve/WY fusion: small-wave win, long-wave regression

September21 follow-up to [mixed-fusion.md](mixed-fusion.md). This is a bounded
operator exploration, NOT a default service change or product promotion. The
current experimental candidate keeps `MTP_GDN_WY_MODE=original`; normal product
admission is unchanged. No new model run, SWE/E2E claim, profile, or publication.

## Why this boundary

Latest retained512-capacity mixed profile (TraceLoom bf6fb49, rank0) has48calls
of each: solve16 sum2.849ms, merge64 sum5.012ms, WY sum6.574ms (~14.435ms total).
These are kernel durations in that recorded graph, not an HBM bandwidth diagnosis.
The source explicitly materializes FP32 Ad[T,24,16], then BF16 Ai[T,24,64]
between the three stages. Fully fused solve/WY removes both framework tensors
and retains the original BF16 inverse rounding before FP32 WY products.
Atcapacity512 the removed allocations total2.25MiB; read+write of full-capacity
payload would be4.5MiB/layer. Live prefill387rows instead has~3.40MiB logical
read+write payload. These are source-level byte counts, NOT measured DRAM traffic
(L2 and internal compiler scratch matter).

The compiler reports mixed AIC/AIV execution and nonzero workspace_size96256
for the inspected fused kernel (metadata field, not a per-layer traffic count).
Therefore do NOT say all intermediate computation now stays in registers/UB or
that physical HBM bytes fell by the theoretical payload count. The final native
lowering/PMU was not used to establish where every internal temporary resides.
A ttir/ttadapter/metadata witness is retained with the evidence.

## Prototypes and controls

- `mtp/solve_fusion.py`: BT64 grouped diagonal inverse plus pinned16→64 merge
  math; donor licenses retained, BF16 output boundary unchanged. Partial-row
  read addresses legalized before masking, following the paid Ascend load hazard.
- `mtp/solve_wy_fusion.py`: one kernel for inverse and WY, with1/3/6 heads per
  task. It also supports the negative partial-fusion control (read Ad, merge+WY).
- `mtp/chunk_layout.py`: opt-in modes original/parallel/fused/merge and
  `MTP_GDN_WY_HEADS=1|3|6`. Original remains default. The parallel arm changes
  both head scheduling and empty-task skipping, not head scheduling alone.
- `mtp/solve_fusion_probe.py`: isolated solve comparisons (1216,split64,fused64)
  and FP64 inverse oracle.
- `mtp/solve_wy_probe.py`: reusable complete-core two-bank CPU recurrence plus
  paired full-output/full-state/conv comparisons and unprofiled NPU-event timings.
  `MTP_GDN_PROBE_VARIANTS=original,fused3` reproduces the final envelope.
  Other variants: parallel,fused1,fused6,merge. Use a fresh capsule and admission.

All probes: hw3 physical5, pinned donors/model geometry from GUIDE, current safe
layout-fused core as control, actual TP-localqk8/v24/K=V128. No model weights or
TP communication in this experiment. No recurrent-state gather/scatter, protocol
change, acceptance D2H, or altered convolution representation.

## Final matched envelope (PASS)

Remote `mtp-fusion-envelope-20260921`, same base as GUIDE. For each point:
three changing input/count/warm/cold generations; active output, ENTIRE state
pool and conv pool exactly equal to current candidate. Separate independent CPU
recurrence covers six changing mixed partitions across two captured banks.
Each arm warmed then timed20replays/block in ABBA order; mean of two blocks.
All capacities refer to buffers, not actual request lengths.

| capacity | actual request lengths | current / fused3 core us | time reduction |
|---|---|---:|---:|
|16|3,9|380.94 /281.76|26.04%|
|64|3,17,2,1|432.59 /329.72|23.78%|
|128|3,65,17|455.71 /375.59|17.58%|
|256|3,193,33|534.02 /492.95|7.69%|
|512|3,33,257,97|714.12 /651.84|8.72%|
|1024|3,769,193|971.89 /1094.60|−12.63%|
|1536|3,1024,509|1246.94 /1438.47|−15.36%|
|2048|3,1536|1486.99 /1557.40|−4.73%|
|2048|1,1,1,1,1,1,1,2041 (all prefill)|1795.64 /2103.09|−17.12%|

Includes both convolutions, preprocessing, KKT, solve/WY, direct-pool H/O,
verification and restore. Excludes projections, output norm, FIA, communication,
HTTP and scheduler. This is not native-vLLM baseline or an E2E gain. Selected
partitions are not exhaustive qualification of every capacity/partition.
Earlier runs showed512 savings~12–13%; use this final matched envelope, not the
best earlier figure. No statistical/population confidence claim from two blocks.

## Negative controls that constrain the next move

1. `mtp-solve-fusion-20260921`: five shapes ×three changed matrices all exactly
   match donor BF16 output; FP64 inverse checks pass. At512, solve alone old1216
   ~189us, split64~154us, fused64~208us. At1536:249/303/496us. Thus smaller tasks
   are NOT universally better; large grouped diagonal work amortizes something
   valuable. Exact cause (task overhead/vector batching/compiler scheduling) is
   not isolated. Removing Ad alone is insufficient to ensure faster execution.
2. `mtp-solve-wy-20260921-v2`: whole solve+merge+WY fusion wins small waves,
   loses long waves. This motivates, but does not by itself qualify, a size-aware
   kernel policy. Default stays unchanged during this exploration.
3. `mtp-merge-wy-20260921`: keeping1216solve and fusing only merge+WY does NOT
   solve the tradeoff. Complete core original/partial-fused meansus:
   512:736/817;1536:1300/1337;2048:1549/1548;64:456/480. Reject as default.
4. `mtp-grouped-wy-20260921`: grouping3heads improves the small fused path
   relative to1/6, but still loses at1536. Final envelope above retests group3.
5. First `mtp-solve-wy-20260921` failed importing an obsolete, unused slot-oracle
   alias from the copied old capsule; no NPU arithmetic ran. Removed that harness
   dependency, not a numerical retry. All subsequent probes pass/release.

Inference: further fusion is real and profitable for small mixed waves, but
preserving batching/parallel scheduling is at least as important as deleting
explicit temporaries. Do not enable full fusion universally. Possible next
routes are a qualified size-aware policy or better Cube/vector tiling; neither
was silently selected or integrated in this exploratory round.

Separate structural opportunity: O already finishes each request-local chunk in
its epilogue. Routing that store directly to original token-major output could
remove the remaining restore roundtrip; verification must write only its disjoint
active rows, and padded rows must never clobber real rows. H/O fusion is harder:
H owns sequential state evolution, O parallelizes across chunks; combining them
changes concurrency and cannot be justified solely by the size of H/vnew tensors.
These are inspected source-level hypotheses, not measured improvements.

Local frozen receipts, launch/source snapshots, admissions/releases, compiler
witness and123-pass CPU log:
`/root/my-ascend-workspace/runs/qwen-mtp-solve-fusion-20260921/`.
The final capsule exits0; device5 is released. No service default, PyPI, website,
C8 eviction handling, or K3/K4 qualification was changed.
