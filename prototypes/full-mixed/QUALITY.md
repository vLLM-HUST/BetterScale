# Retained OpenCompass quality gate

Run the existing32 LongBench English passage-retrieval inputs through native
TP8 DSV4 W8A8/K5, four seats, budget4128, normal HCCL. No synthetic prompts,
new scheduler, full-draft padding, strict collective or shadow instrumentation.
This is a bounded long-context semantic gate, NOT the entire OpenCompass suite.

Inputs are preserved in the workspace run
`runs/dsv4-main-acceptance/20260904T155850Z-4e3323ef/opencompass-c32.requests.json`.
They contain full already-rendered token sequences,9,921–14,997 tokens,32-token
output budget and EOS1. Do not decode/re-render or shorten them. Gold is retained
in `runs/query-gang/20260911-opencompass-g4096-v1/reference.json`, all32 entries;
the18-question request subset in that directory is NOT used. The deployed hw3
model tokenizer was copied for scoring and directly matched the local tokenizer.

The scorer reads `LongBenchRetrievalEvaluator.score` from pinned OpenCompass
Git object60a28a727d3b7807eb3554928f3530d04c948452, following the existing workspace
native-evaluation method. It does not replace the metric with custom matching or
claim to launch the full OpenCompass CLI. Every response, stop reason, prediction,
gold and individual score is retained. Evaluate all32; do not select only passing
questions or excuse malformed output as a formatting mismatch.

Use `--quality-requests PATH --real --spec --tp 8 --kv-gib 12 --budget 4128`
with either the prior `--draft-graph` or candidate `--split-draft`, both with
`--cross-step-bounds --ordered-replay --cpu-qli`. The input determines a15104
model-length bound; both arms use the same contract. Four questions are submitted
per cohort and all8 cohorts execute through native LLM.generate.

Run047 mistakenly reused the short-probe3GiB KV budget. Native hybrid capacity
was only18,017 tokens /1.19 max-length requests, insufficient for four long
requests. It was stopped with owned-resource cleanup and is NOT scored. Both
quality arms instead use12GiB. `quality_capacity` uses native hybrid-aware
`get_kv_cache_capacity` and rejects a less-than-four-request envelope before
submitting work, rather than estimating capacity from a legacy block size.

## Result (048 control /049 split)

Both arms finish all32 original questions and score100.0,32/32 fully correct,
using the unchanged pinned evaluator. `quality-acceptance.json` preserves the
compact result; full predictions and item scores live in each run's
`quality-score.json`. The control is our previously optimized FULL-target plus
stable-K5 draft path, not an untouched release.

Native12GiB capacity is72,090 tokens,4.77 concurrent max-length requests. The
candidate's fresh worker RPC validates that capacity on all8 ranks. In049 each
rank performs101 actual-length context ingestions totaling399,854 rows(max4112),
zero context padding and no query-bank fallback. This is runtime coverage, not
shadow checking(the quality run deliberately disables shadows).

27 CPU tests pass, including fail-closed capacity admission and complete
32-case result mapping. The scoring path reproduces the historical32/32 fixture
before it scores current outputs. Both hardware arms exit0 and release all8 NPUs.
This gate is accepted for the current split path; it is not general model-quality
certification or automatic qualification of subsequent scheduler changes.
