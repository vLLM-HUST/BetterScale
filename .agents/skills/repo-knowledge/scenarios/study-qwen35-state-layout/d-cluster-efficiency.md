# D6 efficiency: balanced target-only decode, 2026-10-02

Enter before optimizing D-cluster step time or extrapolating the PD prototype's
handoff measurements into decode throughput. Source-only capacity arithmetic is
not an execution-width qualification.

## Frozen envelope and result

hw86, cards2..7, native AsyncLLM DP3/TP2/EP6, three fixed attention owners,
Qwen3.5-35B-A3B BF16, CANN9.1.0 / torch_npu2.10.0.post4, unchanged donor pins.
Same candidate-package-7-ep6-state and ep6-runtime as PD. Target-only actual
requests; MTP still loaded/captured/allocated but not executed. Non-strict HCCL,
native32-step finish cadence, FULL graphs, State8GiB/rank, maxlen8192,
E16/R20. No P process and no Store traffic. This is not final MTP or100K service.

Each cohort submits all three owners concurrently, prompt1024 / output128,
greedy/ignore_eos, unique salts. B1 and B8 per owner pass all requested lengths;
no task-quality evaluation is implied. Steady unprofiled dispatch periods use
adjacent equal-batch one-token scheduler rows with decode index40..110.
Client one-token inter-arrival timing corroborates the host periods.

- First run B1: client median14.99ms, rank medians14.97–15.02ms.
- Second run B1: client median13.59ms, rank medians13.47–13.60ms.
  Keep this run-to-run variation; do not present one sample as a stable constant.
- First B8 (24 cluster requests): client median18.453ms,
  rank medians18.421–18.470ms.
- Second B8: client median18.282ms, P90 18.569ms, P99 20.713ms;
  rank medians18.266–18.292ms. Rough steady output rate inferred from24/period is
  1.30–1.31K target tokens/s; this is not E2E service throughput or an MTP result.

## msprof evidence, not profiler-free timings

All six workers capture16 steady B8 forwards using torch_npu's native CANN
profiler, Level1 CPU+NPU, no stacks/shapes/memory instrumentation. Capturing starts
after23 equal-batch decode calls. Synchronize only at capture boundaries, never
in the normal timing path. Native msprof exports happen after model exit.

Each rank's native DB contains16 aclmdlRIExecuteAsync calls,16 slot-mapping
anchors,160 owned attention kernels and1280 GroupedMatmul tasks. Inspect12
interior slot-mapping-to-slot-mapping cycles, discarding capture edges.
Median device cycle20.999–21.033ms; mean21.648ms. This visible profiler overhead
must NOT replace the unprofiled18.3–18.5ms performance baseline.

Per-cycle interval union, averaged over six ranks:

| Recorded class | ms |
| --- | ---: |
| Matmul, including grouped expert GEMMs |7.241|
| MoE routing |1.941|
| GDN recurrent + conv |1.403|
| Owned FA |0.320|
| Other compute / indexing / elementwise |4.250|
| HCCL kernel intervals (including waiting) |4.443|
| No recorded compute/HCCL kernel coverage |2.050|

No compute/HCCL kernel interval overlap occurs in this sampled interior.
This is task coverage, NOT achieved FLOP utilization. Communication spans can
contain rank waiting; no link-bandwidth measurement was made. Uncovered spans
can include host work, launch/runtime gaps and instrumentation; they are not
proven reclaimable scheduler overhead.

Rank asymmetry matters: compute union14.91–15.33ms; HCCL2.56–5.86ms; uncovered
0.86–3.88ms. TP partner ranks1/3/5 have longer HCCL spans and smaller uncovered
spans. Across the whole16-step capture, TP2 allReduce duration is substantially
larger on those odd ranks. This is consistent with arrival/wait imbalance,
not proof of a slow transport link. Do not sum six ranks' durations as wall time.

The native DB reports rankId=-1. Worker EP rank labels are checked against
TASK deviceId2..7 and per-rank capture PIDs; do not invent corrected native ranks.

**Interpretation:** short-context FA is not this sample's main cost.
Investigate the critical-path combination of GEMMs, many small operators,
routing and collective arrival timing before attention-only optimization.
Do not change kernels, State layout, MTP or DP scheduling from this observation.

## Retained failure: B16 is not qualified

The first campaign attempted16 requests per owner (48 total) after B1/B8.
It failed in context_parallel/adapter.py -> plan.schedule with
`ValueError: Expected1..16 real requests; padding is not live work`.
This guard checks metadata lengths; the exact offending metadata was not
captured. Do not conclude it proves more than16 actual requests were admitted,
and do not weaken the guard. Its failure predates the profiler arm.
The second campaign omits B16 and completes B1/B8/B8-profile with exit0.
Future wider-batch work must resolve this preparation/capacity boundary first.

## Receipts, reproduction and teardown

Artifacts under /workspace/betterscale-pd-runtime:
- hw86-d6-efficiency and matching -source/.log: partial first run and B16 failure.
- hw86-d6-efficiency-profile and matching -source/.log/.exit: complete second run.
- timing-summary.json, timeline-summary.json and d6-msprof-timeline.json.gz in
  the second run. Combined timeline preserves shared timestamp alignment.
- Six original PROF trees and exported native DBs, not only summary tables.

The first shell wrapper malformed its exit-status printf; .exit was reconstructed
as1 from the retained exception/wrapper evidence. The corrected second wrapper
writes its observed exit0 with echo. Do not treat the first wrapper as reusable.

Probe entry: prototypes/pd-kv-layout/run_d_cluster_probe.sh OUTPUT (new directory).
This remains explicitly hw86/task-runtime scoped, not a portable service command.
Use analyze_d_cluster.py OUTPUT for timing; after exporting each PROF directory
with msprof --export=on --type=db --output=PROF_PATH, run
analyze_d_timeline.py OUTPUT. No installed donor/runtime changes.

Both runs released all NPUs; /dev/shm was empty after completion. The successful
run nevertheless required executor SIGTERM of six owned workers after grace,
and Python warned about three shared-memory objects. Preserve these shutdown
warnings; no persistent allocation leak was observed after exit.
