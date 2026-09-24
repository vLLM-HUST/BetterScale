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

### Completed35B SWE smoke

The five sealed900s windows (`swe35-summary.json` under the evidence root) have
2717 requests and zero failures. C1/2/4/8/16 output tokens/s/chip are
58.9217/102.7344/160.8350/242.4622/368.9278; P90 request decode tokens/s are
133.3429/125.8538/102.3561/74.0577/55.3789. The native public SWE client is
`vLLM-HUST/swe-prefix-reuse` at29136f1, with exact generated-token continuation
and fresh per-session salts, not synthetic acceptance.

These use explicit24.25GiB KV/chip. Historical FULL used20.25GiB, so the C16
throughput increase from349.6761 is **not an isolated small-fish speedup**.
The earlier request-bounded sampling fix enabled that larger KV budget. Keep
source/configuration curves distinct; same-config best-of chooses a whole run,
never independently maximal throughput and decode speed.

### Dense27B extension observation, not generic admission

The evidence root's `stage27.py` stages a separate dense Qwen3.8-27B capsule
from the September22 qualified dense seed:64layers/hidden5120,48GDN+16FA,
GDNv24/packed5120, FAq12/KV2, and dense MC2 preserved. It expands the query/host
allocation envelope to4096 and16 request seats, rebuilds/re-pins only the host
adapter, then applies this increment with an explicit dense-model guard.
Do not run the MoE-only runtime admission unchanged or transplant MoE literals.
The actual local checkpoint has no attested immutable upstream revision.

`qualified27-candidate.json` binds independent two-bank GDN/state/conv checks,
120 target+72 draft FIA waves, and24/24 real cold/warm/long/concurrent retrievals
through262080 input tokens. Exact-member candidate profiles show target gate
slices0, QKV packs48, z slices48 and no draft vocabulary gathers; complete decode
captures contain320 local greedy-stat tensor bytes across both rounds. Dense z
input is64 heads, not72. Mixed incomplete collective joins remain nonclaims.
This is not a dense before/after performance ablation or general quality proof.

The matched native deployment's8/8 serial cold/warm retrievals passed, but5/16
concurrent outputs ended at `cobalt-seven-` rather than `cobalt-seven-42`.
`native27-functional-limit.json` preserves response IDs2/5/8/11/13 and the
clean exit. This is failed native correctness, not MOD regression or established
harmless rounding. The related35B limitation does not prove a shared root cause.
Keep a native throughput-only policy exception explicit; never rewrite that
receipt as PASS or admit it as a correctness-qualified deployment.
