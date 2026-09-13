# Native DP target FULL prefill/mixed — opt-in experiment

Pins remain vLLM752a3a5 / Ascend9bf964c. The new `--dp-full` worker is separate
from the DSACP extension, native baseline worker, and installed packages.
DP8TP1/EP8 retains native scheduling, CPU DP mode synchronization, shared-expert
and attention multistream overlap, and the eager K5 drafter. No N+2 or speculative
query-graph optimization is included.

## Program and metadata contract

DSA normally slices `[decode | prefill]` in Python; capturing that changing split
is not replay-safe. The native decode metadata builder already has persistent
ragged-query SAS/QLI, RoPE, slot and start-position buffers. `dp_full.py` reuses
these for the entire target token bucket, with device query offsets describing
actual work. Inactive request seats repeat the last real offset; native actual
request handling clears unused state mappings before binding captured scalar
request capacity. Token allocation bounds follow the capture bucket.

Small buckets retain the native decode program. Larger buckets expose the same
stable metadata as a prefill descriptor, retaining native prefill prolog math.
The native prefill path uses BF16 norm then quantization; decode uses fused
norm+quant. A first all-decode implementation therefore changed prefill arithmetic
as well as graph coverage. It is retained in run capsules, not the current code.
Mixed waves still put their short decode queries through the large-bucket program;
this is NOT a claim of bit-identical native split arithmetic.

K5 buckets are6/12/132/264/516/1026 (bounded by the selected budget). The paired
DP8 study uses1026 per rank on **both** sides rather than letting K5 silently drop
an unaligned1024 capture bucket. Native DP padding can cause other ranks to use
the largest rank's bucket: saved dispatch overhead must be measured against this
extra padded attention work, not assumed to produce universal speedups.

## Acceptance (September13)

- CPU33 tests pass, including exact normalization/padding hooks, restoration on
  exceptions, unchanged native oracle entry, and all32 quality-input DP shards.
- Local run058: four-layer SWA/SWA/C4/C128 dummy, TP1, K5, FULL6–516;
  40 same-state checks against the **unified eager program** pass with exact
  output and every unique KV backing byte. Includes508 valid input rows,
  changing seats, mixed scheduling and chunk continuation. Not DP8 qualification.
- Runs055/056: all-decode program vs original native split fails exact KV bytes;
  run056 valid output max difference7.63e-6. Run061 shows merely padding native
  prefill to the same size does not eliminate it. Source identifies different
  prefill/decode norm+quant paths; don't dismiss this as stale pointers or claim
  bitwise equivalence to the original donor.
- Run062 with current large-bucket prefill math: first pure-prefill comparison
  is output/KV-byte exact against original native; mixed70-query step has output
  difference3.81e-6 and fails strict KV bytes. This boundary remains recorded.
- DP2 run060 and DP8 run063 initially failed on idle/dummy ranks. Source
  exposed an oracle error: native `_dummy_run` builds DSA's copied slot map,
  then clears the *source* slot map. Rebuilding metadata between replay and
  eager changed which writes were enabled. The same-program oracle now consumes
  the exact prepared metadata on both sides without rebuilding. Original-native
  comparisons still deliberately rebuild their distinct program. These failed
  runs are not correctness or performance acceptance.
- The reusable signed-zero exception is restricted to addressed BF16 SWA rows;
  every other byte remains strict. Signed-zero passes are explicitly not called
  byte-identical. Never reinterpret a whole aliased pool as BF16.
- hw3 native real run057 was interrupted after startup failed on device5's
  suddenly reduced available memory. Follow-up059 admission caught53GB HBM /
  69% compute on that card despite an empty process listing. No timing claim;
  owned workers released, foreign/hidden activity untouched.

- Run064 (corrected oracle, current prefill/decode bucket programs): all eight
  ranks finish the dummy workload;40 exact same-program checks/rank, valid
  output difference0, all KV pools byte-identical, no signed-zero exceptions.
  These checks cover up to134 valid rows;1026 is captured/replayed later in the
  run but the initial40-check cap misses its state checks. The oracle now also
  checks the first four calls of each larger bucket, even after the small-wave
  cap, so native32-wave drain cannot consume all verification coverage.

- Run067 closes large-bucket coverage:42 checks on every rank, all output
  differences0, all KV backing bytes equal, no signed-zero exceptions. Rank0
  reaches1018 valid rows in the1026 bucket while peer ranks exercise padded,
  mixed and dummy work. This is graph vs the same eager bucket program, not a
  bitwise comparison with the original native split.

### Same-budget real-weight performance

Runs065 (FULL) and066 (native control), sequential on hw3:8x910B2 HCCS,
DSV4 Flash W8A8, DP8TP1/EP8, K5, two active seats/rank,16384 context limit,
8GiB KV/rank,1026 target budget/rank, prefix cache off. Each cohort is repeated
 twice without profiling; profile cohorts are separate. EOS is ignored for
fixed output lengths. These are synthetic token cohorts, not an agent trace.

| Cohort | Native completion seconds | FULL completion seconds | Mean reduction |
|---|---|---|---|
|8 requests,4096 input +16 output each|3.642 /3.919|2.812 /2.687|27.3%|
|1x8192 +7x256 input,16 output each|5.189 /5.194|4.601 /4.589|11.5%|
|16 requests,128 input +128 output each|4.622 /4.591|3.906 /3.678|17.7%*|

The first four balanced-prefill waves are independently verified on every rank:
1018 real query rows per wave, one request/rank, followed by a24-row tail.
Median-of-rank-medians target event durations fall from390.88 /427.45ms to
296.31 /294.07ms. Candidate padding is1026 vs native1018. These device event
spans include collective/queue waits, not just arithmetic kernels.

*Do not call the decode-cohort result a steady decode speedup.* For the first
 ten fully occupied96-query/16-request decode calls, target median durations
 remain about44ms on both sides; target-cycle medians remain58–60ms. Initial
prefill, scheduling, acceptance and drain contribute to cohort time. Two
sequential repeats establish a bounded observation, not universal throughput or
SLO improvement.

Both real runs score32/32 on the same retained OpenCompass LongBench retrieval
inputs (roughly9.9–15K input tokens). This is not the entire OpenCompass suite.
Torch allocator peak is58.016GiB FULL vs58.013GiB native; maximum reserved is
58.670 vs58.635GiB. This excludes non-Torch physical allocations. No tested
capacity claim is inferred from the group-aware cache-equivalent token receipt.

Compact receipts and exact metrics: [`dp-full-result.json`](dp-full-result.json).
The implementation remains **opt-in via the DP probe/native client entry**;
it does not alter the separately packaged TP8/DSACP `strengthen-dsv4 serve`
profile or installed donor files.

### Timelines

Eight-rank native profiles cover the first eight target forwards of the skew
cohort. Canonical local capsules are
`/workspace/strengthen-dsv4-dp-full/runs/hw3-dp8-065/engine/profileskew` (FULL)
and the corresponding `hw3-dp8-066` path (native). Each contains the original
rank DBs, `analysis/sources.json`, clock markers/holdout receipts and
`analysis/target-draft-tp8-end-aligned.json.gz` (legacy filename; this is DP8).
Only completed compressed native exports are shareable. Timing in these
profiled windows is not the table's throughput evidence.

The FULL window has63 unique eager collective identities; their display-only
candidate affine fit passes the50us holdout gate (P95 1.60–6.84us). The short
window has too few small-control-only markers; use the unrestricted eager
identity set and retain its actual residual gate, never weaken the gate or
invent timestamp-nearest pairs.

## Reuse

Use the existing lease/admission launcher and a fresh capsule. `--donor-dp 8
--tp 1 --spec --budget 1026 --kv-gib 8 --real` is the native control;
add `--dp-full` for the candidate. Both use two active seats per DP engine.
Use dummy weights and smaller KV first; `--dp-shadow` plus
`DP_FULL_REFERENCE=unified` checks graph/eager under the same bucket program.
`native` instead rebuilds the original split metadata; `padded` is a diagnostic
that pads native compute bounds. Never conflate these oracles.

`--quality-requests PATH --real` reuses all32 retained OpenCompass LongBench
retrieval inputs, two concurrent requests per native DP client, preserving full
input/output and scoring with `score_quality.py`. No quality pass is inferred
from dummy generation. `--profile-after` with `DONOR_DP_PROFILE_STEPS=8` records
only the first eight target-forward intervals per profiling cohort, excluding
most native DP dummy drain; unprofiled timings are collected separately.
