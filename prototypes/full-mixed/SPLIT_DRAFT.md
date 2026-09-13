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
