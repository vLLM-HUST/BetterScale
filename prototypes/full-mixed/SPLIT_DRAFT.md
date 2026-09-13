# Keep context ingestion out of the large draft graph

September13,2026. Opt-in DSpark/K5, TP8 DSACP, four seats. This is a smaller
alternative to the all-mode N+2 prototype, not an expansion of its scheduler.

## Boundary

Native `AscendDSparkProposer.set_inputs_first_pass` receives all newly executed
target hidden rows. Its context count comes from the final target query offset,
not from the number of candidate queries. Native `_run_merged_draft` first calls
`build_model_inputs_first_pass` to project/store this context KV, then computes
the small query body and Markov-biased K5 proposals. Prefill therefore does have
work to do for the draft, but need not enlarge the query graph.

The former `FullDraftGraphSet` pads every non-6B context to the global budget.
That is an experimental implementation choice, not an upstream requirement or
an N+2 invariant. Both target paths in runs041/042 were already FULL; the one
67.77 ->46.62ms prefill cycle is NOT evidence of a faster target or proof of a
new scheduling benefit.

`split_draft.py` preserves the original fused small-context decode bank for
ordinary K5 verification. Otherwise it calls the unchanged native context hook
once at its actual length, then runs a bounded query-only graph. The merged
upstream routine is reused with only its context hook temporarily suppressed;
no second copy of upstream Markov, sampling, query attention or KV logic exists.
The hook and metadata restore even on failure. Same-stream order makes context
writes available before query reads, with no new per-wave host fence.

Query banks depend on request count and native query execution mode, not the
context length. There is no context padding or fabricated context count. The
existing private metadata banks and first-replay/whole-KV oracle are reused.
For query-only banks the oracle begins AFTER the unchanged native context write;
it establishes query graph/eager equality, not a new whole-runnable shadow.

## Scope and checks

Run with the existing fail-closed lease/admission launcher:

```
--tp 8 --spec --budget 4128 --split-draft --cross-step-bounds \
  --ordered-replay --cpu-qli --observe-cohorts
```

Use `FULL_MIXED_SHADOW=1 DRAFT_GRAPH_SHADOW=1 HCCL_DETERMINISTIC=strict` for
state checks, not performance. There is no `--n2` or custom scheduler here.
The baseline remains `--draft-graph` in place of `--split-draft`.

Run044(dummy,budget288): all8 ranks ingest20 contexts totaling3065 actual rows,
maximum272. Four query banks(1/2/3/4 requests) pass initial-capture and15 subsequent
query checks/rank with exact candidate IDs and all KV bytes, no fallback. Target
shadow also completes. Standard decode retains the previously qualified bank.
CPU tests25 pass, including hook restoration and prevention of duplicate context
writes on normal/error exits. Hardware logs remain in the run044 capsule.

Quality acceptance is a separately measured OpenCompass gate, not an assumption
from completion or tolerant hidden-state comparisons. Fletcher chose not to
continue the cross-run greedy-token-divergence investigation; run043 was stopped
and released without results. Report matched step efficiency separately from
end-to-end throughput, which remains affected by actual wave/work counts.

Run045(real,budget4128): all8 ranks ingest16 contexts totaling6905 actual rows,
maximum4112, with zero padding. The four query banks pass first-capture and12
subsequent exact-ID/whole-KV checks/rank, no fallback. Target44 checks/rank have
zero differences. Compact identical-rank receipts are in `split-shadow.json`.
This confirms the bounded graph/state contract, not OpenCompass quality or speed.

## First unshadowed step comparison

Run046 uses the same real TP8/K5/budget4128/four-seat/3GiB-KV envelope and ordinary
HCCL as041/042, no shadows. Seven cohorts twice,64 outputs/request, then a separate
native profile. This is a single paired observation of matching shapes, not
identical generated token trajectories or a stable throughput result.

| Warm cohort / actual scheduled rows | Prior stable-K5041 cycle | Split046 cycle |
| --- | ---: | ---: |
| cohort8 wave1 /6+17 |65.01ms|46.09ms|
| cohort9 wave0 /7 |67.77ms|49.39ms|
| cohort7 wave0 /64 |202.96ms|204.28ms|
| cohort12 wave0 /1025 |220.08ms|221.38ms|
| cohort13 wave0 /4112 |259.71ms|262.38ms|

Table uses rank0 event intervals. `split-performance.json` also preserves all-rank
medians. Steady four-seat K5 cycle medians(across rank medians) are46.20/46.98ms
for041 and46.29/46.03ms for046. No meaningful steady decode regression is evident.
For the7-row wave, draft duration is13.84ms(041),5.89ms(padded042),3.03ms(split046).
For64 rows it is3.21/6.01/3.46ms respectively. Splitting helps exposed short-wave
launch cost and removes oversized context padding; large target work still
hides most launch cost, so those overall cycles do not improve.

Warm cohort completion totals are9.185s/9.678s/8.642s for041/042/046. These are
recorded for honesty, NOT attributed as causal throughput speedups because
actual output trajectories and wave counts differ. OpenCompass remains unrun.
There is no claim of quality acceptance yet.

Run046 native timeline lives at
`runs/hw3-split-046/analysis/target-draft-tp8-end-aligned.json.gz` after the gated
TraceLoom export. Named CPU regions distinguish `draft_context_ingest` and
`draft_query_graph`; GPU task timing must still come from native links rather
than assuming CPU scope duration equals device execution time.

Subsequent quality gate: runs048/049 both score32/32 on the retained OpenCompass
LongBench retrieval inputs. See `QUALITY.md` for the exact scope, source inputs,
capacity correction and score receipts; the earlier "unrun" status above belongs
to the performance-run handoff, not the current state.
