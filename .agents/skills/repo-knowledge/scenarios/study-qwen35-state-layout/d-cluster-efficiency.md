# D6 efficiency: balanced target-only decode, 2026-10-02

Enter before optimizing D-cluster step time or extrapolating the PD prototype's
handoff measurements into decode throughput. Source-only capacity arithmetic is
not an execution-width qualification.

Current status: the [B16 repair](#full-configured-b16-target-width-restored-2026-10-02)
now supports the configured16 live requests per owner. The earlier failure below
is retained historical evidence, not a current blocker.

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

## Full configured B16 target width restored (2026-10-02)

This section supersedes the earlier B16 failure as the current admission
boundary: **E16/R20 is now exercised at16 live requests per TP2 owner,48 total**.
It does not qualify arbitrary wider E, MTP execution or100K histories.

### Observed cause, not a relaxed limit

The pinned task-local Ascend attention builder extends metadata with a KV=1
dummy row when real requests fill the16-row sequence-length buffer but the graph
has extra query tokens. The owned CP adapter recognized padding only by KV=0.
At the real failure geometry every one of six ranks records:
- input_batch live count16, num_actual_tokens16, graph token capacity24;
- query endpoints1..16,24 and17 host KV rows, the last KV length1;
- device-authoritative sequence lengths still have shape[16];
- the old planner on this exact metadata raises the original1..16 guard.

The fix at the owned **target** publication boundary uses runner live count and
the exact unpadded token endpoint. A shallow copy converts only the proven
padding suffix to one zero-KV interval; native metadata and device length tensor
are unchanged. Genuine live KV=1 histories stay live. Non-matching counts,
frontiers, query ordering, graph capacity or non-padding suffix lengths fail
closed. The16-live-row planner limit and kernel binaries remain unchanged.
Capture keeps its separate disposable-metadata path; draft/MTP is not rerouted.

Source: context_parallel/adapter.py target_metadata and wave.py target boundary.
CPU regression reproduces the exact17-row failure, verifies encoding16 live
rows, input immutability/device-tensor identity, real KV1 rows, suffix compaction
and malformed-frontier rejection. The broader affected subset passes24 tests.
An older workspace-contract fixture lacked the already-required
context_parallel=False; only that fixture was repaired, not the runtime guard.

### Hardware evidence and limits

Isolated candidate-package-8-fia-padding is copied from candidate-package-7,
replacing only wave.py and context_parallel/adapter.py with the reviewed source.
Native donor/runtime, compiled libraries, arithmetic and State layout are not
changed. PD D selection now defaults to this candidate; an explicit
BETTERSCALE_PD_D_PACKAGE absolute path can select a retained diagnostic candidate.

Artifacts under /workspace/betterscale-pd-runtime:
- hw86-d6-padding with matching -source/.log/.exit/-run.sh: exit0, B8 and B16
  complete, each request produces128 tokens, all six ranks' metadata receipts.
  B8 client median18.355ms; B16 client median22.871ms, P90 23.175ms,
  P99 24.660ms. B16 host-rank medians22.846–22.883ms.
  Thus about2.10K target tokens/s inferred from48/median step, NOT E2E throughput.
  Same1K prompt, target-only, non-strict,8GiB State envelope as above.
- hw86-fia-positive-padding with matching -source/.log/.exit/-run.sh:
  one-card actual-kernel independent FP32 CPU causal-attention oracle,16 Q1
  requests plus8 padding queries. Raw host dummy KV1 goes through the new
  translation before native planning. Two graph banks /16 replays, dynamic
  device lengths, exact zero output padding, guard regions and immutable-input
  checks all pass. Existing rtol=.02/atol=.003 unchanged; observed aggregate
  max absolute error0.00926876 (including tiny-context cases), not a new tolerance.
  No native FIA reference was used or repaired.

Full-model token equality is not claimed: under non-strict HCCL, full128-token
B8 sequences differed from the prior run (0/24 exact); B16 had31 distinct
sequences for the identical prompt. The batch/run comparison is not a matched
numerical oracle. Correctness evidence for this repair is unchanged live
metadata/device identity plus the independent actual-kernel oracle above;
retain the previously accepted rounding/segmentation boundary, not an invented
token-equality SLA or a new generic tolerance.

The full-model supervisor exits0. Executors still use SIGTERM after grace and
Python warns about3 shared-memory objects. All NPUs and /dev/shm are clear after
completion. No persistent leak observed. No extra performance capture needed:
the existing B8 msprof timeline is not silently relabeled as a B16 profile.

The D-only probe accepts --batches and --profile-batch (0 disables profiling);
default timing now includes1/8/16 per owner, default capture remainsB8.
