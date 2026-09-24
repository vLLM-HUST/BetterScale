# Mine retained TP2/MTP2 timelines without counting old fixes twice

2026-09-24 offline audit, not a new device run or product change. Use this when
investigating draft communication, padding, or GDN/RoPE glue in the retained
Qwen27 layout/metadata-optimized service trace.

Entry: workspace artifact
`/root/my-ascend-workspace/runs/betterscale-timeline-audit/20260924T154901Z-mtp-small-fish/`.
`FINDINGS.md` is the detailed source/trace synthesis; `analyze.py` reproduces the
exact-member/shape/collective ledger; `source_checks.py` executes original source
fragments against CPU stubs, without Torch/NPU imports. Raw inputs stay in
`runs/qwen-mtp-gdn-fusion-20260921/traceloom-mc2-denoised/` (four rank-local AugDBs
and Perfetto files; analyzer eb3cdbf). Capture is the September21 Qwen27 TP2 MTP2
real-weight service, **not** the later solve/WY operator probe or Qwen35 MoE.
Frozen code is the September22 leaderboard's `qualified-candidate`; its source
identity binds the original `mtp-gdn-fusion-service-20260921-v2` capsule.

## Observed draft work and source explanation

- Actual mixed `[3,257,97]` has512 model rows in **both** draft positions, and
 170 LM-head/argmax rows in both. Rank0 draft13.732ms includes two vocab gathers
 totaling4.374ms, two FC hidden gathers0.305ms, two LM heads3.353ms, argmax0.250ms.
 Rank1 independently agrees on collective counts and similar durations.
- Each vocab gather contributes170*124160 BF16 elements/rank:40.2588MiB local
 tensor payload (not measured link traffic). It is a real AIVEC task with exact
 COMMUNICATION_OP timing, not an MC2 lifecycle-display duplicate.
- `llm_base_proposer.dummy_run` computes `max(tokens//(K+1),1)` and caps only
 during profiling, not ordinary capture.64-token mixed similarly captures21
 sampling rows for4 real requests. Replay retains those captured shapes.
- The request-bounded sampling fix is **already qualified for the later MoE
 capsule**: see `adapt-qwen35-moe/GUIDE.md` and
 `prototypes/qwen35-moe-serving/draft_sampling.py`. It must precede ACL runnable
 construction; returned-ID slicing is too late. Do not advertise this audit as
 a new fix or claim all current serving entries retain the historical defect.
- Even with bounded sampling, `_run_merged_draft` keeps second-position model
 `input_batch_size=num_input_tokens`. The physical512-row second model and FC
 gather remain; FIA's metadata-tail compaction does not shrink GEMMs. A smaller
 request-capacity second graph requires coordinated FIA/positions/slots/bank
 changes. First-position prompt processing cannot be shrunk in the same way.
- Donor already has `enable_reduce_sample` plus distributed greedy selection:
 local max+global ID then two small gathers. Reuse before inventing collectives.
 The flag also changes target/rejection/random-sampling paths; CPU greedy parity
 is not whole-service acceptance. In steady C8decode the TWO vocab gathers total
 only~0.27ms, not the mixed4.37ms; two LM heads still cost~1.97ms.

## Smaller source-informed candidates, not measured savings

- Mixed's same read-only QKV view is implicitly packed before each of its two
 different convolution calls:96 Slice tasks,0.819ms; second-in-pair48 total0.288ms.
 One shared contiguous input before both calls is a narrower test than merging
 convolution algorithms. Preserve disjoint state/role handling.
- Decode's208 Slice tasks are96 b/a packs,48 QKV packs,48 z-layout packs and16
 position slices. Owned preprocess already supports separate a/b row strides;
 native convolution still expects packed QKV. Do not remove all contiguous calls.
- The remaining16 target Index kernels are FA cos/sin cache gathers, not the
 three already-removed AICPU slot gathers. Wave-local RoPE reuse is plausible
 after cache/config/dtype/positions identity and bank lifetime checks.
- Forty-eight output zeros cost~0.48/0.38ms(rank0/1) per steady target. Upstream
 [PR28182](https://github.com/vllm-project/vllm/pull/28182) records the graph-padding
 warning. Owned recurrence has empty/invalid-slot early returns too: establish
 output coverage, including startup/no-metadata and padded tails, before empty.
- Mixed restore remains3.63/3.40ms; rank0's following48 output TensorMoves total
0.470ms. Caller-owned output can remove a copy before the larger O/verify direct-
 store ABI change. Keep address-safe loads and mutually exclusive live stores.

## Query traps paid by this audit

All36,838 exact members join back to TASK with identical start/end. The96
AllGather observations require **device + connectionId + startNs + endNs**:
connectionId alone repeats across graph replays and produces ambiguous matches.
Rank1 provider compute shapes are N/A: infer rows explicitly from collective
count + pinned source width/ordered operator roles, never copy rank0 shape fields
and present them as measurements. Keep operator sums distinct from critical-path
savings and rank clocks independent. No new cross-rank waiting attribution,
unprofiled speedup, runtime modification or NPU qualification resulted here.


## Bounded graph-greedy probe observation (2026-09-24)

Before standalone TP2 graph-greedy tests, preserve **the complete graph family,
static inputs and returned outputs** until every shape/replay is finished, as
serving does. Do not replace/destroy them per shape and interpret the resulting
failure as evidence against the greedy algorithm.

Paid observation with the pinned0.25.1 donor, torch-npu2.10.0.post2,910B2,AIV:
a temporary per-shape fixture passed eager but sometimes returned old remote
values after input updates; an ordinary full-vocabulary all-gather/argmax control
also failed. Native loaded35B Worker initialization did not eliminate the fixture
failure. CPU packing, explicit input readback, CPU barriers and graph-produced
collective inputs did not settle it. Keeping all graph/input/output tuples until
test end was the sole change in `greedy-retained1`: both ranks passed1/3/8/16 rows,
BF16 equal-max ties, all-minus-infinity rows, infinity ties, and changed winners
on both ranks. A separate retained-family matrix covered3/8/16 rows, greedy/full,
none/input/output/both intermediate retention,12 changing-input replays each;
all24 cases/rank passed even **without** intermediate retention. Do not add
production keepalive buffers based on the earlier confounded probe.

This is a bounded observation and a useful fixture-lifecycle rule, **not** a
proven low-level HCCL root cause or completed serving/performance acceptance.
The complete probe family and failed evidence live at workspace
`runs/betterscale-mtp-small-fish/20260924T160000Z-qualification/`:
`greedy-fixture-investigation.md`, `greedy-retained1/`, `greedy-lifetime1/`.
The retained fixture still has to pass inside the real Worker before HTTP gates.
