# Decode/draft graph and replay investigation

September 12, 2026. **The bounded DSpark draft FULL graph works, and improves
matched steady decode cycles by about 20–21%. A stable whole-cohort throughput
improvement is not established.** Changes are opt-in worker extensions; pinned
upstreams and installed runtimes are unchanged.

## What the donor already does

Target decode uses FULL graph. DSpark explicitly sets `use_cuda_graph=False`
in its constructor, independently of `enforce_eager`. Native async DSpark
scheduling already exists. Our change does not introduce a new scheduler or
transplant LiveInfer's entire N+2 budget/receipt protocol.

Native profiling confirmed target replay with eager draft between calls. Dense
kernel tracks alone are not proof of graph replay. Three avoidable host fences
were identified: the target wrapper's stream synchronization and two QLI tiling
maxima read with GPU `.max().item()`.

## Implemented, reversible changes

- `ordered_replay.py`: admit only DSV4 FULL replay on the same stream as input
  production. Retain graph-internal event dependencies. The ordinary wrapper
  fence protects task-parameter updates in other backends; the DSV4 runner
  excludes `use_sparse/use_compress` from that update path. Other paths retain
  the original behavior. Stream identity is checked, not assumed.
- `qli_cpu.py`: reuse the existing local CPU query-offset and sequence mirrors
  for the identical tiling maxima. Optional verification asserts exact GPU/CPU
  equality. Missing mirrors and ineligible paths retain the original builder.
- `draft_graph.py`: private persistent metadata banks capture the complete
  `_run_merged_draft` body: context KV work, neural forward, logits and K5 Markov
  heads. Metadata preparation before that body remains outside capture.
  At most four exact-shape entries cover 1–4 requests, each with six context
  rows. Unsupported shapes/scalars fall back to eager. RoPE banks are private,
  not aliases of the target's global cache. Input addresses/layouts are guarded.

**Capture is not a committed invocation.** The initial call explicitly replays
once after capture before exposing its output/state. Same-state checks cover
that first invocation separately from subsequent replays.

## Correctness evidence and its boundaries

[Receipts](decode-shadow-results.json) preserve all-rank observations.

- Run024: real full weights, TP8, K5, strict HCCL, 3 GiB KV/rank for shadow
  headroom. All eight ranks passed 21 target comparisons against the unchanged
  **native graph**, with exact outputs/MTP and byte-identical KV pools.
  Draft counts 1, 3 and 4 each passed first-call graph/eager checks, followed by
  respectively 4, 3 and 8 replay checks/rank. Draft token IDs and KV bytes match.
- Run025: dummy TP8 checks cover count 2, including the initial call and eight
  later replays/rank. Real-weight count-2 numerical qualification was subsequently added in
  [cross-step run029](CROSS_STEP.md): initial capture plus one replay on all ranks.
- Run026: ordinary HCCL, real weights, normal auto KV sizing, all six measured
  cohorts and the separate profile completed; graph entries 1–4 all executed.
  Completion is not an independent numerical oracle.
- Earlier real target FULL-mixed graph/eager acceptance remains documented in
  [README](README.md). It is not interchangeable with the native-graph reference.

Two failures are retained rather than hidden. Run021 exposed missing KV writes
on the first capture invocation; adding the initial replay fixed it. Runs017–020
only qualified later replays, so their performance pilot is superseded here.
Run023 native target graph/eager whole-pool comparison found 917 differing bytes
in the first 16 MiB chunk after output checks passed. Padding/null-page writes
are a hypothesis, not a diagnosis. Run024 uses the unchanged native **graph** as
reference for the incremental replay policy; it is not a relaxed graph/eager pass.

## Corrected same-engine performance

Run026 uses the original `FULL_DECODE_ONLY` target path, full real
DeepSeek-V4-Flash-0731-w8a8 on 8×910B2, ordinary HCCL, four 64-token inputs and
128 output tokens/request. Six alternating phases use the same loaded engine:
original / ordered+CPUQLI / plus draft graph, repeated twice. Short warmup precedes
each phase. Timing is unprofiled; the profile is a separate final cohort.

Select adjacent waves with **four actual requests, six target queries each**.
Every rank has 21 selected intervals in every phase. Rank-0 medians in ms:

| Policy | Cycle, pass 1 / 2 | Draft device span, pass 1 / 2 | Cohort seconds, pass 1 / 2 |
| --- | ---: | ---: | ---: |
| Original | 65.64 / 64.32 | 7.91 / 6.79 | 4.320 / 3.856 |
| Ordered + CPU QLI | 64.89 / 64.32 | 8.09 / 7.27 | 3.292 / 4.447 |
| Plus draft FULL | **51.74 / 51.50** | **4.15 / 4.12** | 3.750 / 3.797 |

Cycle reduction is 21.2% / 19.9%, about 1.27× / 1.25× matched-wave rate.
All-rank median cycle ranges are 65.14–65.70 / 63.60–64.73 ms originally and
51.63–51.78 / 51.50–51.57 ms with draft FULL.
See [all-rank CSV](real-decode-corrected.csv) and [protocol summary](real-decode-corrected.json).

**Do not turn this into a serving throughput claim.** Actual positive wave counts
are 49/34/50/43/52/50 despite fixed output lengths. Speculative acceptance and
batch evolution differ; runtime captures can also occur as requests finish.
There are only two passes, fixed order, and no long-context or high-concurrency
service qualification. Fence removal alone has no stable large cycle benefit.

## Remaining intervals, not imaginary free performance

With draft FULL, rank-0 target-to-draft intervals are about 1.44–1.45 ms and
draft-end-to-next-target intervals 6.10–6.45 ms. The latter remains material.
They contain metadata, copies and sampling—not necessarily idle hardware.
Device event spans include queued work and waits. Draft host-call spans shrink
from about 35 ms to 10 ms, but host and device overlap: never add those savings
as if they were serialized latency. Likewise target span changes can reflect
arrival/collective waits rather than a different neural operator.

The native eight-rank profile and TraceLoom candidate clock fit live in:
`/workspace/strengthen-dsv4/runs/hw3-real-policy-026/analysis/`.
The short final profile records eight native target replays and eight native
draft replays on every rank, identified inside their respective host scopes.
Each has eleven forward calls overall; prefill/ineligible calls remain outside
these replay counts.
Compressed timeline: `target-draft-tp8-end-aligned.json.gz`. Clock fits are
**display-only candidate alignment**, not a physical simultaneity proof. The fitted ranks have holdout P95 residuals of 1.10–1.83 microseconds. Raw
sources, matching markers and holdout receipts are retained. [Profile helpers](profile_tools/README.md)
reproduce the import, fit and native export without ingesting raw JSON into chat.

## Reproduction

Use the existing launcher/runtime/model/lease contract in README. Native control:
`FULL_MIXED_PATCH=0`, `--mode FULL_DECODE_ONLY --tp 8 --spec --budget 288 --real`.

- `--policy-study --rounds 6 --output-tokens 128 --profile-after`: run026 protocol.
- `--decode-study --requests N`: bounded count 1–4; default 4.
- `--ordered-replay --cpu-qli --draft-graph`: enable the combined candidate.
- `--verify-qli`: exact CPU/GPU maxima assertion, excluded from timing runs.
- `FULL_MIXED_SHADOW=1 FULL_MIXED_ORACLE=native_graph DRAFT_GRAPH_SHADOW=1`,
  strict HCCL and `--kv-gib 3`: run024 incremental-state qualification.
- `--replay-study --rounds 4`: fence-only alternating control.

`diagnostics.py` records per-rank scheduler waves and event spans;
`summarize_policy.py` selects equal-work intervals. CPU contracts cover stable
metadata banks, first-call execution, bounded bank count and stream admission.

The extension is a useful working prototype, **not production-default FULL draft
support**. Broader request counts, K values, metadata shapes, long-running reuse
and service-level throughput require their own qualification.

The next dependency cut, conservative CPU bounds and post-submit receipts, is
documented in [CROSS_STEP.md](CROSS_STEP.md).
