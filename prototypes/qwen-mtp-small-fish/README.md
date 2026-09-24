# Draft-only greedy and bounded GDN copies

Experimental Qwen3.5-35B-A3B BF16 TP2 / serial MTP2, not a released wheel.
Stage from the qualified **request-bounded sampling** capsule and its matching
Ascend runtime; never patch an installed runtime or live capsule:

```sh
python prototypes/qwen-mtp-small-fish/stage.py SEED RUNTIME NEW_OUTPUT
```

The output has source pins and a minimal diff. Native libraries are unchanged;
resolve their existing pinned paths as in the seed deployment. Put output,
output/package and output/runtime-source before the original runtime on
PYTHONPATH. Both opt-ins default to zero:

- `BETTERSCALE_MTP_GREEDY=1`: sets a per-draft-proposer and separate draft logits
  processor flag, reusing donor `compute_draft_token_ids` / `greedy_sample`.
  Global reduce-sample configuration and target logits/sampling stay untouched.
  It exchanges local maxima and global token indices, not the full vocabulary.
- `BETTERSCALE_GDN_SMALL_COPIES=1`: keeps b/a gate splits row-strided only under
  the owned GDN consumer, and packs immutable mixed QKV once for both conv roles.
  The decode QKV pack, z copy, output zeros and state ownership remain unchanged.

Do not infer other models, parallel layouts, quantization, padded vocabularies,
MTP widths or concurrent model runners from this admission. The second-draft
model envelope is deliberately unchanged. Request-row bounding predates this
increment and is not credited as a new saving.

## Bounded acceptance (September24)

Workspace evidence: `runs/betterscale-mtp-small-fish/20260924T160000Z-qualification`.

- CPU policy/staging tests plus prior sampling tests passed.
- Exact native greedy, real TP2 HCCL and retained complete graph/input/output
  families pass ties, infinities and changing remote winners at1/3/8/16 rows.
  `greedy_probe.check_initialized(directory)` is a separate fixture, not an
  allocation hook to run inside serving startup.
- Strided GDN tests pass independent CPU recurrence, full conv history and
  output checks across mixed cases/two graph banks. NZ-weight gate projections
  produce ND outputs; strided/packed preprocessing is exact in the tested sizes.
- Clean control and candidate pass24 exact chat retrievals:8K/32K/128K/262080
  cold/warm plus16 concurrent requests. Both exit0 and observe real MTP acceptance.
  This is functional coverage, not general model-quality certification.
- TraceLoom `ab8b5131191c6d5aeee2dd8566c34411f49ceab0` analyzes full two-rank
  decode/mixed profiles. Two independent analyses agree on every selected
  event/launch/member row. Each target's60 gate Slice tasks disappear; mixed
  QKV packing goes60→30, decode stays30; z/position slices stay unchanged.
  Draft vocab gathers disappear while FC gathers and model/sample rows stay.
  At16 sampling rows, both draft rounds' local sampling payload is
  7,946,240→320 bytes. This is tensor payload, not physical wire traffic.
- Rank-local decode draft envelope medians were2992.73→2695.56us (rank0),
  2988.49→2708.68us (rank1), in six-step diagnostic windows with outliers.
  Do not turn these into an end-to-end speedup or add overlapping member times.
  Mixed4096 has one missing exact INT64 collective/member join; its incomplete
  payload subtotal is not a claimed saving. Retain provider evidence.

Earlier serving with diagnostic collective graphs allocated/freed during model
load failed strict32K retrieval. Removing that instrumentation yielded the
clean serving pass; this does not establish the low-level vendor root cause.
Retain the failure and do not quietly inject fixture allocations into production.

SWE Prefix Reuse performance is a separate gate, using the same prepared file,
exact returned-token continuation, natural MTP2 and900-second windows. Never
publish profiler timings as leaderboard throughput.
