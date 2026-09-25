# Optional live execution: accepted construction order

Fletcher approved the plan on2026-09-25: make live execution an optional
BetterScale path, with one owner for State initialization, warmup, capture,
invocation and retirement. Default native execution remains unchanged. The
intended configuration is a mutually exclusive native/live selection, not
independent allocation/capture switches; the CLI spelling is now `--runtime native|live`. Live serving remains
reserved and rejects before native resource preparation. Unsupported live configurations must reject before initialization,
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

`src/betterscale/live/llm/qwen35/gdn_graph.py` composes the complete small-model
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

The follow-up declaration now includes `anchor_hidden[R, hidden_size]` BF16,
initialized invalid alongside anchor_token=-1; its shape is checked using the
real0.8B config. This adds2KiB per resident, not per context token. Writing the
correct accepted target boundary and preserving draft-prefix validity are still
model-integration work, not implied by allocating the tensor. The historical
owned-graphs3 result predates this declaration-only extension.

## Packaged root boundary

The root, State declarations and bounded graph/kernel closure now live under
`src/betterscale/live/llm/qwen35/`. Prototype scripts only exercise the package;
no runtime source is resolved from that directory. QwenStateRoot is the model
State composition, not a completed LiveLLM forward implementation. GDNGraphRoot
remains an explicitly seeded numerical fixture. Do not rename either into a
fully qualified serving root. Python construction/activation is the working
opt-in entry; CLI live serving is a fail-closed reservation until the real-weight
vertical and scheduler are connected. Native remains the default.

Packaging acceptance:47 focused CPU tests passed (including the nested14 State
contracts); a fresh sdist/wheel includes the full root closure and no prototype
or external livemodule package. The installed wheel passed15 focused tests,
including isolated root imports, graph lifecycle and native/live launcher
admission. Receipt: `runs/qwen35-state-lanes/20260925-packaged-root/`. Numerical
AST comparison is identical except import relocation; no new NPU result is
claimed. The transferred experimental kernel retains its donor unused-local
lint warning; it was not numerically edited for packaging.

## Real-weight execution frontier (2026-09-25 evening)

Fletcher authorized continuing through35B-A3B TP2 live execution end to end.
Work branch `codex/qwen35-live-e2e` starts from merged main6347a9e. The numerical
root is now being built in `execution.py`/`numerics.py`; `residents.py` owns only
host leases/page IDs, and `generation.py` advances the resident continuation.
No native Worker/runner/KV allocation is called. These additions are under
qualification, not yet the public serving route.

Observed small0.8B TP1 gates:
- `runs/qwen35-state-lanes/20260925-target4/`: complete weight-name coverage,
 7 successive real target tokens through all24 layers. Independent Transformers
 BF16 CPU cold-prefix reference (`target1/reference.pt`) agrees on all7 greedy
 choices; worst full-vocabulary logit RMS0.064645, worst hidden cosine0.999617.
 These are observed cross-backend differences, not a general error tolerance.
- `20260925-generation1/`: eager MTP2 emits the same12 tokens as target-only;
 real accepted7/10 proposals. Unrelated B uses empty seat1 and leaves seat0's
 complete GDN candidates/history exactly unchanged. C hits19 committed tokens
 on seat0 and its8 outputs equal independent cold target-only recomputation.
 D evicts the older seat1. Root close/reactivation resets the generation.
 All selected-card jobs exited and released; these are functional, not speed
 or full-graph results. Frozen package snapshots live inside each later capsule.
- Eight CPU resident/protocol tests protect empty-first hot hits, shorter-prefix
 misses, live-lease eviction rejection, page exhaustion without partial growth,
 EOS truncation and target-hidden correction of recursive draft rows.

Paid integration details:
- Native MTP first pass shifts input IDs but **keeps target positions**. Only
 recursive proposal steps increment position. After target verification, replace
 recursive draft prefix rows with corresponding target-hidden seeds; retain only
 one boundary vector per resident and the current short wave, not full history.
- Direct donor construction supplies original QwenGatedDeltaNetAttention,
 whose method is `split_ba`, not Ascend adapter `_split_ba_for_tp`.
- Device selection/custom-op loading alone does not initialize donor Triton
 device properties. Call `init_device_properties_triton()` explicitly under
 admission before fused gating; this must not require constructing a Worker.
- `construct_live` was a test-local helper, not an exported runtime API. The
 packaged example now correctly uses `with live_runtime(runtime): Root(...)`.

The full-model graph root (`graphs.py`) reuses the same numerical forward,
registers target1/3 and draft1/2, and bounds plain attention to an explicit
context envelope.17 focused CPU tests include the real four-graph metadata
entry with stand-in arithmetic and no forward-shadow fallback. Its model NPU
qualification is the next gate; do not label the eager receipt as capture proof.

`20260925-model-graphs2/` subsequently passed the real0.8B four-graph vertical:
actual root type/count were asserted, no forward-shadow graphs, same12 output
IDs as eager target, real7/10 proposal acceptance, exact untouched hot GDN State,
19-token warm hit and8-token warm/cold equality. Close/reactivation and release
passed. `model-graphs1` is explicitly **not graph evidence**: a harness class
selection error plus a preparation edit racing script load meant its scope label
was wrong. Its separate `qualification.json` invalidates that claim. Never edit
an admitted/queued capsule; stage a new one and assert actual runtime identity.

TP2 frontier is NOT yet accepted: `20260925-tp2-live3` loaded both ranks'35B
weights, executed eager and four owned graphs, and matched the initial12 IDs.
The warm/cold check then differed at output3 (zero-based): warm13 vs cold15.
Do not label this harmless numerical noise or publish the live route as qualified.
Independent native reference and teacher-forced eager1/graph1/eager3/graph3
comparisons are prepared to separate State reuse, batch shape and capture.
Earlier TP2 integration failures exposed two concrete donor seams:
- MoERunner.load_weights returns flattened RoutedExperts names, while
  named_parameters includes `.routed_experts.`. Coverage now resolves only
  aliases backed by the actual routed module and reported parameter; never
  exclude all expert weights from coverage.
- Native MTP context passes `model_instance=None`. The donor's has_layer_idx
  caches its first target observation; passing the draft model after target
  incorrectly reads draft.model.start_layer. Match native's explicit boundary,
  not a fabricated draft attribute.

Both target and draft now use the same native logits_processor.get_top_tokens
in greedy-only mode (TP value/index pair reduction). Full vocabulary output is
retained only as an explicit diagnostic mode. Loopback HTTP/CLI integration is
being prepared, but must remain on this work branch until the TP2 discrepancy
and public-entry acceptance are closed.
