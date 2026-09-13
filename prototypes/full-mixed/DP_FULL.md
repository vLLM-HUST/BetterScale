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

## Evidence so far (September13)

- CPU32 tests pass, including exact normalization/padding hooks, restoration on
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
- DP2 run060 (earlier all-decode version):10/11 exact checks per rank before
  rank1 finds signed-zero differences in SWA; valid output remains exact.
  The oracle now reuses the established exception only for addressed BF16 SWA
  rows; every other byte remains strict. It never reinterprets a whole aliased
  pool as BF16. Signed-zero passes are explicitly not called byte-identical.
- hw3 native real run057 was interrupted after startup failed on device5's
  suddenly reduced available memory. Follow-up059 admission caught53GB HBM /
  69% compute on that card despite an empty process listing. No timing claim;
  owned workers released, foreign/hidden activity untouched.

DP8 replay/state, full-checkpoint quality, same-budget unprofiled performance,
and the new bounded eight-rank timeline are **not yet accepted**.

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
