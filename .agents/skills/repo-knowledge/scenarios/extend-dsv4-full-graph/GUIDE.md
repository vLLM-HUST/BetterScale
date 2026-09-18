# Extend DSV4 FULL graph and replay scheduling

Use this scenario for **donor DSV4 graph coverage, draft separation and replay
scheduling**. For persistent external expert servers or Qwen38, go directly to
[separated experts](../integrate-separated-experts/GUIDE.md); their historical
gates no longer live behind this donor entry.

## Choose the question before reading evidence

Read this guide, then only the selected note/section and its owning source.
Prototype paths below are repository-relative. Past gates are bounded evidence,
not a list of enabled public features.

| Current question | First useful entry |
|---|---|
| What is shipped and how do I launch it? | `docs/RUNBOOK.zh-CN.md`, `patches/README.md`; [maintained entry](donor-gates.md#maintained-serving-entry) |
| Is prefill/mixed really FULL? | [source investigation](investigation.md), `prototypes/full-mixed/README.md`; [capacity trap](donor-gates.md#fixed-capacity-lesson-from-the-first-passing-probe) |
| Does an eager/graph State check mean anything? | [aliased-pool oracle](donor-gates.md#tp8-oracle-heterogeneous-pool-aliases-and-determinism) |
| Target/draft graph split or shape padding? | `prototypes/full-mixed/SPLIT_DRAFT.md`; [original-vs-padded boundary](donor-gates.md#prefer-separate-context-ingestion-to-oversized-draft-banks) |
| Continuous replay, N+2, receipt placement? | [LiveInfer continuation](liveinfer-continuation.md), then the relevant [authorization gate](donor-gates.md#cross-step-authorization-and-receipt-placement) |
| DP startup warmup? | `src/betterscale/patches/async_decode/README.md`, `docs/evidence/release-0.3.1.json` |
| Shared/communication overlap or KV prefetch? | `prototypes/kv-prefetch-overlap/README.md`; DMA differs: `prototypes/graph-dma/README.md` |
| Native DP+EP topology/cache ownership? | [native DP scenario](../study-native-dp/GUIDE.md) |

## Invariants that matter across these routes

- Use pinned donor source; do not mutate installed runtimes during inspection.
- Distinguish eager, piecewise and FULL by API/replay evidence, not density of
  device tasks. A white interval is not automatically removable idle time.
- Compare matched work and actual wave counts. Generation/acceptance differences
  can change cohort time without changing per-step efficiency.
- Keep exact-state, language-quality and throughput gates separate. Do not
  interpret aliased mixed-dtype State pools through every BF16 view.
- Warm the finite shape catalog before timing; runtime capture is not free.
- Preserve upstream shared-expert overlap and public Worker defaults unless
  the chosen experiment explicitly changes them.

[Donor gate details](donor-gates.md) retains the original receipts, failed arms
and scope limitations. It is a reference, not required startup reading.
