# Five-attention prefill timeline (2026-09-18)

The existing E3 SWE runner schedules each source independently. Any pending
prefill in that source takes priority over its decode work. With8 resident
requests and1024 input lanes it assigns each request at most128 tokens/wave.
Unused seats do not donate their128-row allowance to a longer request. This is
an experimental fixed-width scheduler, not a mature mixed scheduler.

`analyze_prefill_schedule.py` reconstructs effective prefill rows from monotone
encoded cursors and checks their sum against the receipt's actual prefill count.
For the completed `20260917T202430Z` E3 run:

-583 prefill waves across five sources,236,306 useful rows out of596,992 bucket
  rows:39.6% fill.391 waves advance only one request.
- The slowest source,attention2, has162 prefill waves;124 advance one request.
  Its174.9s cumulative prefill cost is distinct from42.1s decode-wave cost.
- These are tensor-row accounting figures, not a claim that every padded row
  costs exactly the same FLOPs as a valid row.

## Fresh profile

hw0 capsule `qwen38-model-20260918T063408Z`: same TP1x5+E3, real48 target+MTP K1,
State46GiB, e affine-head QSA. After normal warmup, all five attention roles
record their first two SWE prefill waves;1024 valid inputs/wave/source. The
receipt is labelled PROFILE and cannot be used as a completed throughput test.
Expert servers execute normally but are not profiled in this capture.

The two full waves show that padding is not the whole story. Attention0's
largest named task sum is `neural_collect`:98 calls,631.94ms total (~316ms/wave).
Attention2 records698.97ms. Collect includes remaining expert wait and result copy; weighted reduction
is a separate `MoeTokenUnpermute` task. **It is not expert GEMM time**. Other attention0 two-wave sums:
FIA103.45ms, QSA gather98.59ms, host PLE mailbox pull106.83ms. Do not sum
potentially overlapping task durations as an end-to-end decomposition, and do
not substitute instrumented timings for the unprofiled benchmark.

## View and reproduce

Local compressed files under
`/workspace/betterscale-confluence/runs/qwen38-prefill-profile-20260918/analysis/`:

- `qwen38-prefill-attention5-annotated.json.gz`: five-rank device view plus10
  recorded host wave bands carrying valid/bucket input counts (about4.9MiB).
- `attention2-prefill.json.gz`: individual attention2 TraceLoom detailed view.
- `operator-cost.json` and `profile-receipt.json`: query results and provenance.

Torch-NPU raw capture is analysed offline after releasing cards. TraceLoom
37323af preprocesses each provider DB, then its native distributed exporter
constructs event lanes. The view restores **same-host provider timestamps**;
we do not zero each rank's first event or invent collective endpoint matches
for point-to-point expert IPC. No additional independent clock calibration is
claimed. Host wave bands use the recorded PYTORCH_API timestamps in the same
provider domain. Each rank has2 markers and98 collect completions, matching
49 target/MTP layer calls per wave. The combined view has82,640 device slices.

Enable `QWEN38_PROFILE_STEPS=2` and absolute `QWEN38_PROFILE_DIR` before
`run_topology_case.sh tp1-e3 trace 46 <fresh-case-dir> affineheads`.
`device-service/profile_export.py <root> --roles attention0 attention1 attention2 attention3 attention4 --label qwen38-prefill-attention5` handles
preprocessing/merging; `annotate_prefill_profile.py <analysis-dir>` adds wave
bands. The remote exact launch/analyse scripts, raw profiles and SQLite files
are in `/workspace/betterscale-hw0/runs/qwen38-prefill-profile-20260918/`.
A relocated TraceLoom binary also needs its classification, structural-symbol
and reconciliation TSVs; pin the three TRACELOOM_*_RULES variables. Select the
provider's `ascend_pytorch_profiler*.db`, not its auxiliary `analysis.db`.

## Client/collect inspection

Source inspection of `client.py`, `wire.py`, `build.py`, and
`../qwen-next/client_kernel.cpp` establishes the current protocol:

- Submit, native shared expert, promotion, collect and retire are launched on
  the same current stream. Remote server execution can overlap shared expert;
  local submit copying cannot. Submit publishes READY only after all rows,
  scales, route IDs and the descriptor have been copied.
- Submit uses one AIV block and a row-by-row read/write loop. Target rows also
  copy their FP32 scale through a padded cache line. The IO helper fences each
  transfer and executes PIPE_ALL after each write. This is a correctness-first
  protocol, not a bandwidth-tuned prefill mover.
- Collect uses16 blocks. Each waits for every owner generation before reading
  any result, then copies route outputs individually into a materialized
  `[rows,10,2560]` BF16 buffer. At1024 rows this is50MiB per layer. Weighted
  reduction happens afterwards in native unpermute, not inside collect.
- `build.py` explicitly removes the old H2048 `neural_collect_reduce`: its
  fixed UB geometry/BF16 assumptions are not safe for this H2560 W8A8 port.
  Reusing token-ready collection requires adaptation, not flipping a flag.

In attention2's first captured layer, local submit runs647.82us and ends at
520999.48us; the first shared MatMul starts520999.49us. Collect subsequently
runs4028.82us, followed by a separate46.88us unpermute. Across98 layer calls,
submit totals62.105ms and collect698.971ms (two instrumented waves). Select
one representation of each task: the detailed TraceLoom export contains both
published-tree and device lanes; summing both doubles the measurements.

The current capture cannot separate server scheduling/compute wait from pull
cost. Next bounded diagnostic: timestamp owner completion versus collect
first/last transfer, with a results-already-ready copy control. Side-stream
submit should be tested separately, preserving input-ready, submit-ready,
shared-completion promotion and generation/retirement dependencies, including
FULL graph capture. Overlap does not remove the delayed READY publication or
prove a net speedup under shared memory-resource contention.

## Borrow the donor's shared-expert overlap, not just its stream count

Inspected the clean donor checkout at
`/workspace/strengthen-dsv4/upstream/vllm-ascend` (`9bf964c`).
`ops/fused_moe/fused_moe.py::_forward_shared_experts` switches to a dedicated
shared stream and inserts phase-specific event waits. `moe_comm_method.py`
records dispatch/combine boundaries; `moe_mlp.py` records the second-GMM
boundary. Shared quantization can overlap the router, shared gate/up overlaps
routing communication, activation overlaps routed down GEMM, and shared down
overlaps combine. The main stream joins shared only before consuming the result.
This inspected wrapper uses ordinary streams/events, not an explicit AIV quota;
that observation does not characterize every underlying collective's resources.

Retained native evidence, not a new hardware run:
`/workspace/strengthen-dsv4/runs/hw3-split-046/analysis/rank0.db`, TASK rows,
anchor `1789279550578849335`ns (first DequantSwigluQuant). Relative microseconds:

| Stream | Task | Start | Duration |
|---|---|---:|---:|
|10|QuantBatchMatmulV3|-1385.29|94.74|
|11|hcom_allGather_|-1356.05|160.64|
|12|GroupedMatmul|-16.54|250.83|
|10|DequantSwigluQuant|0|39.12|
|12|MoeTokenUnpermute|241.11|55.60|
|10|QuantBatchMatmulV3|249.27|64.66|

This confirms local task-envelope overlap matching the source's phase design;
it does not independently measure collective byte movement or saved latency.
Use native same-rank timestamps, not an inter-rank alignment fit, for this check.

For the separated design, start by forking shared computation from client
publication at the earliest immutable hidden-input boundary. Client input
quantization/packing may then overlap shared work; source READY still requires
all payload and route writes. Shared completion promotion must wait for BOTH
that generation's publication and shared completion: otherwise promotion can
read the previous generation. Collect/pull and shared can execute independently,
joining before output addition; retirement and bank reuse must preserve all
readers. Moving collect's busy-poll kernel earlier may contend with shared AIV
work, so initially isolate submit/shared overlap rather than changing collect
and publication simultaneously. Capture the fork and join into FULL graphs and
verify changed-input replay, not only an eager timing win. Unlike donor, routed
GEMM is remote here: do not copy its GMM2 wait onto the local shared path.

## First fork/join implementation and gate (September18)

`QWEN38_SHARED_OVERLAP=1` enables a warmed shared stream and bank-owned events.
The fork precedes input quantization/packing. Main still submits and then joins
shared before promotion/collect; the latter algorithms are unchanged. Default
remains serial. `probe_shared_overlap.py` is selectable through
`run_wire.sh ... --client-probe probe_shared_overlap.py`.

hw0 four-card gate: one source plus E3, real target layer0 routed weights and
**dummy BF16 shared gate/up/down weights** (not the complete model shared
expert). FULL serial/overlap outputs are exactly equal for1/32/1024rows,
priority0/1, three changed-input and rotated-routing replays/case. Servers drain
normally. This does not qualify MTP BF16 layer48 or the full48-layer service.
See `shared-overlap-result.json` for the complete receipt and10 alternating
serial/overlap event samples per case. At1024rows median serial/overlap is
4.657/4.676ms(priority0),4.648/4.679ms(priority1): **no net speedup**.

TraceLoom-preprocessed two-replay trace confirms the fork actually executes:
serial submit613.56us, then shared from613.58 to699.16us. Concurrent shared runs
from4902.44 to5006.80us while submit runs4936.12 to5562.46us. Shared's first
MatMul grows31.74 to49.48us in this one diagnostic pair (not a robust interference
estimate). Serial collect3804.48us versus concurrent3913.96us includes waiting;
these are instrumented samples, not the alternating benchmark medians.

Interpretation: moving shared under local submit removes its visible serial
placement, but routed server work already overlapped shared after publication.
With this short shared branch, the remote path still determines completion;
earlier collect can simply wait longer. Reducing time to READY and improving
collect remain distinct opportunities. Do not enable by default or claim a
serving win just because two streams visibly overlap. The hardware roles and
lease were released; profile analysis was CPU-only afterwards.
