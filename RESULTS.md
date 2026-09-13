# DSV4 graph work: bounded results and closeout

2026-09-13. This round stops at the split-context/query prototype and its quality
gate. No further scheduler or KV-distribution work is included. All changes are
opt-in; no installed donor runtime or upstream release pin was modified.

## Common environment

- hw3,8×Ascend910B2/HCCS, CANN9.0.1, torch_npu2.10.0.post2.
- Full real DeepSeek-V4-Flash-0731-w8a8 checkpoint, Ascend W8A8, BF16 activations.
- vLLM v0.25.1 (`752a3a504485790a2e8491cacbb35c137339ad34`),
  vLLM-Ascend v0.25.1rc1 (`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`).
  Source identity has bounded file checks, not a blanket rebuilt-runtime claim.
- TP8+EP, DSACP enabled, DCP size1, FlashComm1/AIV HCCL and shared-expert overlap.
- DSpark K5, at most4 active requests, greedy, prefix caching off.
- Timing: normal HCCL, no shadow, no profiler. Profiles are separate diagnostic
  windows. Shadow checks instead use strict HCCL and identical state snapshots.

## What improves, against which baseline?

### Steady four-request K5: two separately isolated increments

Both studies use target `FULL_DECODE_ONLY`, capture sizes24/288, global
budget288, max_model_len2048, automatic KV sizing at memory utilization0.85.
Each request has64 prompt tokens and128 output tokens. Select consecutive waves
with exactly4 real requests ×6 target queries; each phase has21 matched intervals
per rank. Below are medians across the8 per-rank cycle medians.

| Change | Exact comparator | Comparator -> candidate | Cycle reduction |
| --- | --- | --- | --- |
| Private-bank draft FULL, ordered replay, CPU QLI | Original target FULL decode + native eager DSpark |65.60 ->51.74ms;64.50 ->51.54ms |20–21% |
| Conservative CPU bounds + post-submit receipt handling | Above optimized graph path, cross-step cut OFF |52.05 ->46.60ms;52.46 ->46.26ms;51.66 ->46.11ms |10–12% further |

These are same-engine alternating controls in runs026 and031 respectively.
The latter prewarms all4 draft banks before timing. Ordered replay/CPU QLI alone
have no demonstrated large stable improvement; do not assign them the entire
draft-graph gain.

Engineering progression is approximately65 ->52 ->46ms per matched K5 wave,
roughly29–30% shorter /1.4× wave rate from the beginning to the end. **That total
is a cross-study summary, not a new direct end-to-end A/B measurement, and the
percentage reductions must not simply be added.** See
[decode evidence](prototypes/full-mixed/DECODE.md) and
[cross-step evidence](prototypes/full-mixed/CROSS_STEP.md).

### Prefill/mixed: separate native context ingestion from small query graphs

Runs041(control) and046(split) both use target FULL, capture sizes24/4128,
global budget4128, max_model_len8256,3GiB KV/rank. The control already includes
our stable-K5 optimizations above; it is NOT an untouched donor. Seven input
cohorts are run twice,64 output tokens/request. Warm observations below use
matching scheduled row counts, with medians across ranks, not selected best ranks.

| Actual scheduled rows | Prior optimized control | Split candidate | Change |
| --- | ---: | ---: | ---: |
| Small mixed:6+17 |63.96ms|46.37ms|27.5% shorter |
|7-token first wave|63.05ms|49.55ms|21.4% shorter |
|64-token first wave|202.92ms|204.03ms|0.5% longer |
|1025-token first wave|219.96ms|221.15ms|0.5% longer |
|4112-token first wave|259.65ms|262.16ms|1.0% longer |

This is a single paired shape observation, not a replicated universal gain.
Previously quoted67.77 ->49.39ms and65.01 ->46.09ms are the rank0 values for
these same short waves; the table deliberately uses all-rank medians instead.
Steady four-seat K5 remains essentially unchanged:46.20/46.98ms(control) vs
46.29/46.03ms(split). Large prefill is still dominated by target computation.

The kept design updates draft context KV at its true row count, then replays
only the small query/proposal body. Ordinary decode keeps its fused small
context+query fast path. No global4128 context padding or new N2 scheduler is
needed. Native same-stream ordering preserves context-write ->query-read order.
[Implementation and evidence](prototypes/full-mixed/SPLIT_DRAFT.md).

## Quality and state

- Retained OpenCompass LongBench English retrieval: **32/32,100 points in BOTH
  prior optimized control048 and split candidate049**. Original9,921–14,997-token
  inputs,32-token output limit, EOS honored; TP8/K5/four seats/budget4128.
- Quality capacity:12GiB KV/rank, max_model_len15104. Same questions and unmodified
  pinned evaluator60a28a727d3b7807eb3554928f3530d04c948452. This is not the full
  OpenCompass suite or general model-quality certification.
- Split real-weight query graphs pass first-capture and replay exact candidate-ID
  and whole-KV checks on all8 ranks. Native target shadow also passes in the
  bounded state-check envelope.27 CPU tests pass.
- Quality run049 actually executes101 context ingestions/rank,399,854 rows,
  maximum4112,zero context padding and no query-bank fallback.
- [Quality reproducer and results](prototypes/full-mixed/QUALITY.md).

## What is not claimed or adopted

- No stable whole-request or production service-throughput improvement has been
  established. Output trajectories and speculative wave counts differ. The
  short warm-cohort9.185 ->8.642s result is NOT attributed as a causal speedup.
- The all-mode N2 +padded-context experiment is retained as evidence but is not
  the adopted route; its incremental benefit did not justify that combination.
  Dynamic join/leave/mixed boundary waits remain future work, not solved claims.
- KV layout is still donor-style replicated HBM state across TP ranks. No
  request-owned KV, mapped C4 or memory-capacity virtualization was integrated.
  At model length15104,3GiB/rank gives native hybrid-aware capacity18,017 tokens;
  12GiB/rank gives72,090 tokens(4.77 full-length requests). These are logical
  capacity estimates for the SAME requests, not quantities to multiply by8.
  The current harness admits only4 active requests; this is not maximum-concurrency
  benchmarking.12GiB quality candidate peak allocated is53.42GiB/rank.
- At the original closeout, the deliverable was an opt-in prototype/patch
  collection. The kept path is now maintained in `src/strengthen_dsv4/` with
  an explicit worker/HTTP entry; see the [runbook](docs/RUNBOOK.zh-CN.md).
  This promotion does not broaden the performance or quality claims above.
  Runtime pins and unrelated work remain unchanged.

## Artifacts and handoff

Code and compact results are in this private repository. Large raw evidence
remains in `runs/` locally and the corresponding hw3 capsules. Latest compressed
8-rank TraceLoom timeline:

`/workspace/strengthen-dsv4/runs/hw3-split-046/analysis/target-draft-tp8-end-aligned.json.gz`

Its clock model passes the display gate(P95 residual0.72–2.31us), but is a
candidate affine alignment, not physical calibration. Never send the raw JSON
through the desktop conversation. All owned NPU tasks and leases are released.
