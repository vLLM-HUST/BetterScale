# Resume whole-wave asynchronous decode, September 13

Fletcher's active goal is to recover roughly8ms from the exposed decode
cross-step window while preserving the LiveInfer protocol and numerical gates.
This is a target, not a claimed result. Both input and egress must retain
independent ping-pong ownership. Do not replace two bank-bound invocations with
one shared writable packet just to reduce graph count. Numerical continuation
State itself remains single-copy and compute-stream ordered.

Source explanation:
`../../.agents/skills/repo-knowledge/scenarios/extend-dsv4-full-graph/liveinfer-continuation.md`.
Earlier failed adoption applies to the complete old candidate, not a theorem
that every useful mechanism in that branch is ineffective. This new goal
explicitly authorizes reconsideration with a narrower evidence question.

## What the retained evidence actually says

Run101, real DP8/EP8, K5,16 global requests/96 actual target queries, normal
HCCL, native eager draft. Two same-engine repeats; first10 consecutive fully
occupied steps, common ordinals across all8 ranks, no dummy drain:

| Policy | Cycle ms | Draft-to-next-target event interval ms |
|---|---:|---:|
| native |64.606 /66.500|13.977 /16.678|
| owned ingress + device producer |58.268 /57.961|9.116 /8.139|
| producer + captured metadata |50.611 /50.858|1.238 /1.235|

The last mechanism reduced the producer's exposed interval by7.878/6.904ms,
with cycle savings7.657/7.103ms. It is already strong evidence for the proposed
mechanism, but not new-code acceptance or guaranteed cohort throughput. Current
composed TP run112 has ~1.33ms in this interval, not another10ms to remove.
Retained run101 data root:
`/workspace/strengthen-dsv4-pingpong/runs/101-dp8-shared-pool-study/engine`.
Reanalysis JSON: main checkout's `runs/decode-continuation/retained-dp101-windows.json`.

## Invocation ownership map

- Host H2D source: owned pinned ingress, immutable until its DMA completes.
- Device input/metadata bank: no overwrite before its old graph reader finishes.
- Target hidden/auxiliary: consumed by sampler/draft on ordered device execution;
  distinguish these intermediates from the asynchronous client-output payload.
- Numerical cursor/accepted count/anchor/draft: exact single-copy device State.
- Whole-wave output: bank-owned if captured, with an output-copy-complete gate
  before its next write. Native uncaptured sampler currently creates per-call
  tensors retained by AsyncGPUModelRunnerOutput; retaining a tensor reference
  does NOT protect a captured buffer from the next graph overwriting it.
- Host result: retained per invocation until consumed; not an ingress scratch.

Any integration must document which allocation serves each role and which
recorded event orders its next reuse. Do not infer safety merely from graph
count, RPC queue depth, or keeping a Python tensor reference alive.

## New bounded execution

Local artifacts are under main `runs/decode-continuation/`.

- 113: preliminary single-target dummy attempt rejected by runner geometry gate
  before model initialization.114: failed importing ACL after task wrapper
  accidentally replaced CANN PYTHONPATH. Both invalid; owned resources released.
  Their code is parked as `parked-single-target.patch`, not the chosen route.
- 115: DP2 dummy, original dual target banks + explicit ingress + captured
  metadata, composed with DP FULL prefill. Runtime then failed because the
  optional FakeTensor native-projection audit wrapped the already-installed
  live producer. It mutated nested producer slots not covered by the audit's
  shallow runner restoration. This is an invalid diagnostic composition, not
  an output/KV mismatch. The launcher released the workers.
- 116: same dual-bank composition without that redundant projection audit.
  Retains exact native preparation comparisons and independent target/full-KV
  oracle. Completed successfully (details below). The runner now rejects the invalid flag combination before
  importing the model runtime. Preserve both the failed capsule and the gate.

Next gate: finish bounded dual-bank correctness, then real TP8/DP8 matched-cycle
measurements with explicit policy scope. Preserve graph-bank memory and output
copy lifetime evidence; profile separately from throughput. Do not enable the
unhelpful worker retirement extension by default. The package/main runtime is
unchanged so far.

116 completed: each DP2 rank passes48 target/full-KV comparisons (maximum
output difference0) and12 exact preparation checks. Both target banks replay55
times/rank. Ingress has4 shape/bank slots (83,824 pinned bytes/rank). Metadata
cache reaches5/8 entries across the two ranks; all are within the original
bounded geometry. This run used runtime packet refresh, not captured copies,
so its elapsed time is not the intended performance configuration.

117 failed full-machine admission (foreign jobs on local6/7).118 acquired a
later window and began real-weight loading, then a new foreign process appeared
on card6. The launcher stopped only its owned workers and released the lease;
there is no118 correctness or timing result. hw3 card6 remains occupied too.

Another bounded observation: run101 rank0 metadata cache grew4->8 entries
between repeats. Do not count first-shape capture as steady-state performance or
assume a2-request warmup covers independently padded DP shapes. The receipt now
exposes finite `(requests,padded_requests,padded_tokens,host_carrier)` keys without
publishing raw addresses, so a warmup can be checked instead of guessed. This
adds cohort-boundary evidence only, not per-wave host synchronization.


## Fresh DP8 closure and packaging gate

121 passes real-weight strict-HCCL same-state checks: all8 ranks each pass48
native-graph target/whole-KV checks with output maxdiff0, and12 exact native
preparation checks. Oracle memory peaks58.469GiB/rank; not a serving peak.

120 passes real-weight normal-HCCL study and32/32 retained OpenCompass retrieval
questions. First10 consecutive common fully occupied ordinals, repeated twice:
pair cycle59.893/59.085ms -> metadata50.377/49.998ms; draft-to-target interval
9.417/9.493 ->1.238/1.234ms. The interval is not entirely idle hardware.
No claim of stable cohort-throughput gain. Quality scoring uses the unchanged
OpenCompass60a28a7 evaluator, not a full-suite run.

The eight-rank profile is limited to the beginning of each cohort. Its mode
receipt continues to drain after profiling stops; continuation.py therefore
requires explicit --profile-prefix to inspect only complete leading pairs.
Small-control markers are fewer than20; unrestricted unique eager collective
identities pass the SAME50us display gate (candidate metadata1–3us P95).
No captured-instance guessing or residual-based point deletion.

A closed async_decode package is being qualified separately, with native DSA
FULL in target_full and unchanged TP8 composition. It omits study policies,
reference graphs, runtime tree-copy walks, FakeTensor discovery, file output,
and the unhelpful worker-retirement extension. Run122 exercises the actual
package through an external numerical-oracle subclass; it is not accepted yet.
Final sampler output remains native per-invocation allocation retained through
D2H, not a falsely claimed captured whole-wave two-slot egress implementation.
