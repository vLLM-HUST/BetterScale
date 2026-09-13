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
  oracle. Pending. The runner now rejects the invalid flag combination before
  importing the model runtime. Preserve both the failed capsule and the gate.

Next gate: finish bounded dual-bank correctness, then real TP8/DP8 matched-cycle
measurements with explicit policy scope. Preserve graph-bank memory and output
copy lifetime evidence; profile separately from throughput. Do not enable the
unhelpful worker retirement extension by default. The package/main runtime is
unchanged so far.
