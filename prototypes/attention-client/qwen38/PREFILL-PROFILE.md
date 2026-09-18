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
Attention2 records698.97ms. Collect includes remaining expert wait, result copy
and reduction; **it is not expert GEMM time**. Other attention0 two-wave sums:
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
