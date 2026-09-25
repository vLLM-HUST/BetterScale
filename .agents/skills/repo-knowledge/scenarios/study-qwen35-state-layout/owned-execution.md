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
