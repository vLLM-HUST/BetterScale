# Decode/draft graph and replay investigation

Pinned donors, opt-in worker extension, September12. This is experimental,
not a replacement of the installed donor runtime.

## What exists before our changes

Target decode has FULL graph. DSpark explicitly sets `use_cuda_graph=False`
in its constructor, independently of the user's `enforce_eager` choice.
The native scheduler already supports async DSpark scheduling. Do not describe
this work as inventing async scheduling for a wholly synchronous donor.

The wrapper's current-stream host fence protects task-parameter updates in
other backends. The DSV4 runner excludes `use_sparse/use_compress` from those
updates. A narrowly admitted DSV4 target can instead replay on the same stream
as its input/metadata production. `ordered_replay.py` does that, retaining
internal graph event dependencies. It is **not** by itself a complete
LiveInfer-style N+2 budget/receipt protocol.

QLI's builder already has local CPU query-offset/sequence mirrors, but obtains
two tiling maxima with GPU `.max().item()`. `qli_cpu.py` uses the same CPU
mirrors already consumed by SAS metadata; optional verification asserts exact
GPU/CPU maxima before removing these synchronizations in performance runs.

## Evidence so far

- `tp8-decode-013-profile`: four-layer dummy, K5, four64-token inputs and
 64-token outputs. All eight native profiles collected, exported offline because
 the workers are daemonic. Rank1 records66 native graph replay calls, matching
 target forwards, with eager draft calls in between. No claim that dense tasks
 are graph replay. Raw artifacts remain on hw3 in the same capsule.
- `tp8-ordered-014-shadow`:66 same-state target graph/eager checks on each rank,
 output/MTP differences0 and every KV pool byte-identical.
- `tp8-qli-016-shadow`:65 same-state checks/rank, same exact outcome, with
 CPU/GPU QLI maxima verified. Both use strict HCCL and dummy weights.
 See `decode-shadow-results.json`. A shadow snapshot fence is not evidence of
 concurrent metadata-buffer reuse safety.
- `tp8-real-015-replay-study`: one real-model engine, alternating original fence
 and ordered target replay. Completion times differ with20–32 scheduled waves;
 they are NOT evidence of a20–40% replay speedup. Rank0 median target-to-target
 event intervals were68.25/67.41/68.49/68.30ms. Target forward device spans are
 roughly47–49ms, draft around10ms, and the gap from draft end to next target
 around8.3–8.8ms. Event spans include queued work/waits, not pure kernel sums.
 Removing the target fence alone is, at most, a small improvement in this pilot.
- `tp8-draft-017-capture`: private stable metadata banks allowed the entire
 `_run_merged_draft` body to capture and replay32 times/rank, with8 shape
 fallbacks/rank. This includes context KV work, draft neural forward, logits
 and the K5 Markov chain—not draft metadata preparation. Run018 then passed8 same-state draft graph/eager checks on every rank: exact
 draft token IDs and byte-identical KV pools. This is dummy-model validation;
 real-weight and performance gates remain separate.

## Experimental draft contract

`draft_graph.py` captures ONE observed exact shape for four-request verification,
with at most24 target context rows. Different shapes/scalar metadata fall back.
Its private bank copies fresh metadata into persistent tensors before replay;
RoPE must NOT borrow/overwrite the target's global cache. Device values can
change; scalar structure and CPU tensor values are checked in the key.
The initial implementation favors a clear lifetime contract over minimizing
metadata copies. It does not cover arbitrary request counts, prefill, K values,
or target/draft shape dispatch, and is not advertised as production FULL support.

Use `DRAFT_GRAPH_SHADOW=1` to compare eight captured draft replays with eager
from restored identical KV bytes and require exactly identical draft tokens.
Keep shadow and profiler costs out of unprofiled performance measurements.

## Reproduction entry points

Same launcher/runtime/lease contract as README. Extra flags:

- `--decode-study`: warm up, observe four64-token prompts producing64 tokens.
- `--ordered-replay`: DSV4 target stream-ordered replay experiment.
- `--cpu-qli --verify-qli`: exact CPU/GPU tiling-maxima comparison; omit verification
 only in a separately qualified performance run.
- `--draft-graph`: bounded private-bank runtime capture.
- `--replay-study --rounds 4`: original/ordered/original/ordered in one engine.
- `--policy-study --rounds 6`: original / ordered+CPUQLI / plus draft graph,
 repeated twice, warming each phase. `--output-tokens` controls measured output.
- `--profile-after`: separate short native profile after unprofiled phases.

`diagnostics.py` stores per-rank wave token counts and device-event spans.
`summarize_decode_trace.py` summarizes native host scopes and compute/communication
coverage. Host scope time, device interval and true idle time are distinct;
never add overlapping spans or count all post-target time as an idle bubble.
