# Decode-cluster efficiency: real requests and synthetic sizing

For the latest authorized100K/80%-State dummy observation (DP4TP2EP8, MTP2),
read [resident long-KV sizing](#resident-long-kv-dummy-sizing--october3-no-pd-pressure-run).
The D6 sections below remain historical target-only evidence.

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


## Resident long-KV dummy sizing — October3, no PD pressure run

Fletcher explicitly redirected the blocked host-memory campaign to **synthetic
shape/cadence sizing**, not a real-request warm-start benchmark. Do not restart
P4, allocate large host pools, or build a seed-service pipeline merely to repeat
this observation. Track the separate deployment dependency in
https://github.com/vLLM-HUST/BetterScale/issues/10.

hw86, eight910B2, DP4TP2EP8, native MTP2, D-only FULL graphs,
E48/R56, requested44GiB State/rank, context envelope262144. Same pinned
CANN9.1/torch_npu post4/donor runtime and native libraries as online PD.
No P process, rank-private host arena, State transport or live request scheduler
participates in the timed loop.

The fixture charges the actual declared State tensors:47,223,508,256B/rank,
including5,353,868,576B fixed resident/MTP State and1,815 logical FA blocks.
At100,000 tokens/request and11,264 dense bytes/token/rank, each request needs
49 logical2048-token blocks. floor((0.8*total-fixed)/request_dense) gives
**28 requests/TP2 group,112 total**. Unique non-null FA block rows occupy
31,650,217,984B/rank; fixed+active dense is78.3595% of total after rounding.
This includes all56 reserved seats; it is not80% of64GiB physical HBM, nor80%
live occupancy of the dense-only pool (that is75.59%).

The probe initializes synthetic zero KV and separate recurrent rows, uses
synthetic token IDs, publishes actual100K device lengths and runs the native
target+merged-MTP dummy path. It does **not** validate numerics or sample actual
accepted outputs. Native dummy execution is not the full serving step: no live
admission, request postprocessing, State transfer, or end-to-end return path.
Acceptance below is an external assumption, not measured by this fixture.
Synthetic MoE routing need not match real prompts. Do not relabel the result a
qualified maximum throughput or a production<50ms result.

8 warmup cycles then32 measured cycles, NPU events on the execution stream,
one synchronization before and after the window, no per-step synchronization:
- Eight rank means70.0044–70.0088ms; medians69.8209–69.8320ms.
- P95 nearest-rank70.0604–70.0753ms; first measured cycle75.59–75.74ms.
- Host dispatch medians69.77–69.82ms corroborate the event spacing.
- Historical v33/v34 positive output chunks averaged~2.826 accepted tokens.
  Assuming2.826 per request/cycle,112*2.826/0.07000885 = **4521.0 tokens/s**
  aggregate, **565.1 per D chip**. This is analytical, not measured goodput.

### Ingress/P capacity arithmetic and assumptions

A100K full checkpoint per TP rank is approximately95,608,332B resident frame
plus49*23,071,424B FA frames =1,226,108,108B. Pair total2.452GB.
D device restore and P-to-D network transfer are different byte counts:
an immutable peer DRAM hit can transfer only increments across the network
while still restoring the whole history into a newly assigned D device seat.

For one explicit warm-turn scenario, reuse the v33 SWE **continuation-only**
means (1865 turns):386.536 output tokens/turn and1338.880 new input tokens/turn.
Project that ratio onto100K resident histories; this is not a claim that the
original v33 workload itself averaged100K.
At4521 tokens/s:
-11.696 turns/s; P requires15,659.9 new-prefill tokens/s aggregate
  (~3915 per TP2 P instance if spread across four equally).
- Full D device restore each turn:28.682GB/s aggregate,3.585GB/s/rank.
- With hot peer immutable pages, P->D incremental payload~2.59GB/s,
  ~3.13GB/s allowing one extra full tail page/turn;
  D->P~2.34–2.88GB/s under the same accounting.
- Cold100K every turn instead requires~1.170M prefill tokens/s and full
  checkpoint network ingress, not the warm15.7K/3.1GB/s figures.
These are byte-volume models, not measured transfers, and exclude transport
headers, allocator headroom and burst margin. Output length, new input length,
device residency and peer-cache hit rates must accompany any sizing quote.

General formula: output Q =4*B*accepted_per_cycle/step_seconds;
turn rate=Q/mean_output; P work=turn_rate*mean_new_prefill.
Per-direction network increments add one resident frame per rank per turn,
not one per decode step. Full device restore scales with whole context.

### Fixture and artifacts

prototypes/pd-kv-layout/dummy_decode_probe.py and dummy_decode_entry.py are
disposable observation entries, not new serving worker admission.
Use a frozen stage_online_candidate.py --mtp D capsule with the existing
run_pool_node.sh environment and:
BETTERSCALE_PD_NODE_ENTRY=dummy_decode_probe.py,
BETTERSCALE_PD_RANK_PRIVATE=0, BETTERSCALE_PD_CONTEXT=262144,
BETTERSCALE_PD_MTP=1, BETTERSCALE_PD_DECODE_ONLY=1,
BETTERSCALE_PD_D_CONCURRENCY=48, BETTERSCALE_PD_D_RESIDENT_SEATS=56,
BETTERSCALE_PD_D_STATE_GIB=44, BETTERSCALE_PD_D_PACKAGE=<frozen capsule>;
arguments --output <fresh directory> --context100000 --fraction0.8
--warmup8 --steps32 (with normal spaces between flags and values).

Keep startup/capture untouched. The measurement-only AST clone of the pinned
native _dummy_run corrects two live-query/padded-request assumptions:
do not numpy.repeat the live partition, and use actual request count until
native FIA adds its padding row. Exact-source guards reject drift. GDN starts
at0 with an endpoint-filled padding suffix; req_ids AND req_id_to_index define
live rows. Existing FIA/GDN guards remain enabled. Two CPU regression tests
cover live partition preservation and unknown donor rejection.
The preceding v1–v4 setup failures are retained, not measurements: unused
cache-control configuration, padded request broadcast, missing GDN leading0,
then missing native request-index map. No installed donor or production
serving source was changed to get the dummy through those guards.

Successful raw evidence: /workspace/betterscale-pd-runtime/dummy-kv80-v5/
(result.json, sizing.json), matching -source, -launch.json and.log.
Frozen native capsule: dummy-kv80-v1-package (48df72e staged source);
later vN source directories vary only the disposable probe helpers.
All eight devices report no running process after normal model teardown.
Executors still needed SIGTERM after grace and Python reported4 shared-memory
cleanup warnings; no persistent NPU allocation observed.
