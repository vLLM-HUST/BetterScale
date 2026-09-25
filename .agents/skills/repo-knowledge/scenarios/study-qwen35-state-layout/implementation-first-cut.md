# Independent State declaration/allocation cut

## Accepted implementation boundary — 2026-09-25

After accepting separate execution/resident/token-page capacities, Fletcher
explicitly asked for a worktree and implementation using LiveInference's
**declare first, allocate later** flow. This supersedes the earlier research-only
boundary for this isolated prototype, not the rejection of secondary development
on native heterogeneous storage. No offloader or new allocator was authorized.

Worktree: `/root/my-ascend-workspace/betterscale/.worktrees/qwen35-state-lanes`,
branch `codex/qwen35-state-lanes`, base
`0e2055825359678c89aae225de8fd13f5d1bbd77`. The parent checkout, installed runtime,
qualified serving capsules and their processes were not changed. Research notes
were copied into this worktree to preserve the accepted design alongside code;
other dirty parent work was not copied or removed.

Enter [the prototype README](../../../../../prototypes/qwen35-state-lanes/README.md)
for exact declarations, reproducible CPU checks and outstanding interfaces.
LiveInference was clean at `05ac15419c0e73650e687ceb9daffeb7874865f0`.
That first cut imported the external library. Fletcher subsequently rejected
that dependency boundary: the same common runtime closure is now transferred
into `betterscale.live`, not reimplemented or re-exported from `livemodule`.
See `src/betterscale/live/README.md` for the precise intake; the historical NPU
observation below remains tied to its original external-library source.

## Observed result

-14 CPU contracts passed using torch2.10.0+cpu and the real Torch/grouped State
backends: no allocation at declaration; two independent capacity domains;
full0.8B geometry; candidate clearing; elastic-budget accounting; binding/release;
rejection of native preallocation/unknown addressing ABI; rollback on failure.
The config is pinned to model revision
`2fc06364715b967f1860aea9cf38778875588b17`, with the full24-layer text sequence.

-One Ascend910B2 leaf run passed after selected-card0 lease,30-second under-lease
admission and continuous foreign-owner guarding. Other cards hosted foreign
work; this was not an isolated-host performance measurement. Evidence:
`/root/my-ascend-workspace/runs/qwen35-state-lanes/20260925-leaf2/`.
The launcher, admission receipts, full wave receipt and release observation live
there. No model weights or network serving were involved.

-The NPU allocated all50 numerical State tensors plus6 continuation tensors for
E4/R5/P32, page128, MTP2. One GDN leaf executed on two native NPUGraph banks,
24 waves with variable lengths, permuted resident order, padded rows, old accepted
candidate3 with currentT1, and an untouched fifth seat. Conv addresses the seat;
SSM addresses `seat*3+candidate`. Every candidate row and conv-history row was
checked, not only outputs or graph/eager agreement.

-Max absolute errors: conv output `3.0517578125e-05`, recurrent output
`1.9073486328125e-06`, recurrent State `9.313225746154785e-09`. Conv-history
storage matched exactly. The recurrent oracle intentionally consumes observed
BF16 conv outputs; separate conv-output comparison bounds that numerical seam.
Native graphs were synchronized and reset before root close.

## Source boundary discovered while implementing

The base branch's public `decode_kv.py` admits only non-speculative H8/HV24;
it cannot run the speculative candidate proof. The first admitted run failed
on that explicit guard before recurrence (`20260925-leaf1`); this was not a
State allocator failure. The qualified experimental capsule has the necessary
static candidate expansion, correct candidate-column stride, and previous-wave
selection handling, but its geometry guard is H8/HV16.

The prototype isolates that candidate kernel in `gdn_candidates.py` with only
H16/HV16 geometry admission changed for0.8B TP1. Original source Git blob:
`4c1518d498467947f8f8e2b42e71e299a0cf1db7`; original capsule:
`runs/qwen35-parallel-matrix-20260924/full-tp2-ep2-capacity32-source/package/`.
AST equality after that guard substitution was checked (formatting excluded).
License notices are retained. Do not silently replace the published kernel or
call this narrow copy a general geometry qualification. Consolidation belongs
with the eventual production consumer integration, not a second kernel family.

## Not yet implemented / not proved

This cut declares and realizes semantic storage; it does not yet supersede the
native serving scheduler. There is no resident directory, hot-seat reuse/eviction
transition, page allocator, model weight loading, prefill/decode integration,
real verifier-driven selection, draft-resume protocol, or LiveModule-owned graph
execution. The small numerical gate does not prove model logits, prefix reuse,
HTTP correctness, CPU offload, TP2/MoE or performance.

The consumer borrowing seam is guarded by an explicit addressing ABI; it is not
an automatic conversion of existing Mamba modules. Rebinding attached consumers
is rejected until cached pointer tables and captures have explicit retirement.
An ABI string alone never establishes correctness or graph ownership.

Continue at the composed execution/continuation boundary using the accepted
hot-resident policy. Do not interpret the continuation's allocated counters as
an implemented state machine or let old native blockpool authority survive
behind a new `kv_cache` reference.

## Owned namespace follow-up

The PR review requires BetterScale to own its runtime dependency closure. The22
common implementation modules (plus package initializers) are transferred with
namespace substitution and change notices, retaining upstream module boundaries.
No models, serving stack, architecture ports or native binaries were taken.
The root deliberately drops the old lazy Ascend-only compatibility export.
No upstream import, module alias, editable install or checkout search is used.

The prototype and its runner now import `betterscale.live`. The CPU regression
set includes transferred SIMD State/grouped-backend contracts and an isolated
process that forbids `livemodule` imports while executing all14 Qwen tests.
Namespace-only implementation equivalence is checked against the pinned source;
this is not a new hardware or full-model qualification.

Observed relocation gate:33 CPU tests passed (32 transferred State/backend
contracts plus the subprocess boundary test running14 Qwen cases). A fresh
sdist/wheel build contained all26 owned Python files and the package README;
native payloads came from the retained0.5.1 wheel and passed existing build-pin
checks. The wheel was installed with no dependencies into a separate target;
the forbidden-external-import gate also passed against that installed artifact.
Two existing public-package identity/Worker-import checks passed. Artifacts and
the22-module AST comparison receipt are under
`runs/qwen35-state-lanes/20260925-owned-runtime/`. No installed runtime was
modified, no PyPI release was made, and no namespace-only NPU rerun was needed.
