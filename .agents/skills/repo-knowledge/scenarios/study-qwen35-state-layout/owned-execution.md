# Optional live execution: accepted construction order

Fletcher approved the plan on2026-09-25: make live execution an optional
BetterScale path, with one owner for State initialization, warmup, capture,
invocation and retirement. Default native execution remains unchanged. The
intended configuration is a mutually exclusive native/live selection, not
independent allocation/capture switches; exact public CLI spelling is not yet
implemented. Unsupported live configurations must reject before initialization,
never fall back after partial activation.

Construction order:

1. Own the existing small numerical graph probe through LiveModule lifecycle.
2. Complete a real-weight smallQwen35 TP1 execution vertical, including target,
   draft, verification and hot-resident continuation.
3. Connect the qualified vertical as the optional serving path, branching before
   native cache planning/runner initialization. Reuse request ingress and usable
   numerical/weight-loading leaves without retaining hidden BlockPool ownership.

The runtime provides physical capabilities; the model root owns its generation;
leaves declare State/metadata/graphs. Do not add a second lifecycle coordinator.
Automatic capacity fitting, CPU offload and TP2/MoE remain later gates.

## Observed first cut: owned GDN graph lifecycle

`prototypes/qwen35-state-lanes/gdn_graph.py` composes the complete small-model
State tree with two declared LiveGraphs and banked stable output MetaTensors.
It is intentionally a fixed0.8B GDN probe, not a full model or generic runner.
Capture declares resident0..3 writes and no token-page writes; StateTensor lowers
those resident IDs to all three recurrent candidates. The fifth hot seat remains
untouched. Explicit metadata construction resolves each bank's outputs; replay
does not use the full-forward shadow compatibility fallback.

`betterscale.live.arch.ascend.graph.ACLGraphBackend` is the small physical
specialization transferred from LiveInference05ac1541 with namespace-only
changes. It is selected explicitly, does not initialize devices on import, and
uses the already-owned common graph lifecycle.

Single-card0 evidence:
`/root/my-ascend-workspace/runs/qwen35-state-lanes/20260925-owned-graphs3/`.
Under selected-device lease, fresh30s admission and foreign-owner guard,24 waves
passed independent candidate/conv-history checks. Max absolute errors were
`7.62939453125e-06` for convolution and recurrent outputs and
`9.313225746154785e-09` for recurrent State. Conv history matched exactly.

Additional observed gates:
- Nonzero State initialization survives warmup and capture.
- Close rejects while a LiveInvocation has not retired.
- After completion and close, graphs are unprepared and State unbound.
- A new activation rejects an old captured execution; initialization is restored.
- Root owns physical graph reset; the probe does not create/reset NPUGraphs.
- Device0 returned to IDLE after the run.

42 focused CPU tests passed, including the relocated Ascend backend protocol,
existing State/rollback cases, public-package identity, the external-import guard,
and a CPU execution of the actual GDN graph entry with stand-in numerical kernels.
This last test checks metadata construction/capture restoration, not GDN math.

Two earlier bounded entries exposed protocol integration errors, not numerical
failures: owned-graphs1 had no explicit metadata construction program and correctly
rejected full-forward shadow; owned-graphs2 passed context positionally to a
keyword-only helper. The actual entry now has a CPU preflight to catch such
contract errors before NPU admission. No guards were disabled to obtain the pass.

This does not yet qualify full-model graph execution, concurrent bank reuse,
request scheduling, State eviction or HTTP serving.

## Bounded donor loading seam

The pinned0.8B snapshot is now downloaded at
`/workspace/models/Qwen3.5-0.8B`, revision
`2fc06364715b967f1860aea9cf38778875588b17`. A single-card0 admitted probe
(`runs/qwen35-state-lanes/20260925-loader1/`) used native `initialize_model`
with explicit text target and MTP classes, loaded248 target parameter names and13
draft parameter names, and shared the target embedding/lm_head with the draft.
Seven attention cache fields (six target plus one draft) had zero payload;
GDN cache fields are not published by these constructors. No Worker or runner
was constructed, no native cache planner was called, and no graph or forward was
executed. Device0 returned to IDLE. This proves a usable loader seam, not model
numerical equivalence or a finished model adapter.

[probe_donor_load.py](probe_donor_load.py) retains the bounded construction
recipe. It requires the pinned donor runtime, admitted NPU environment, exact
local model snapshot, and an explicit `CAPSULE` output directory. Its file-based
single-rank distributed rendezvous belongs to that output directory. Never
execute or import it outside admission. Vision tensors are deliberately skipped;
text prefixes are remapped explicitly. Do not turn this probe into a general
model loader without parameter-coverage and actual-forward acceptance.

A concrete next census item: native MTP forward consumes both input token
embeddings and target hidden states (`pre_fc_norm_embedding`,
`pre_fc_norm_hidden`, then `fc`). The current declaration-only continuation has
anchor token/progress but no hidden boundary tensor. Full continuation needs a
bounded per-resident hidden seed and a defined valid boundary; carrying only the
anchor token is insufficient. This is source-derived, not a passed continuation
transition. Do not reintroduce an entire target-hidden history to fill that gap.
