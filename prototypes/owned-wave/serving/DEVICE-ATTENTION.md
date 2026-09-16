# Remove per-layer host attention task updates

## Established seam

The swe-profile1 TraceLoom augmented DBs retain3072 FIA host calls and3072
update begin/end pairs per rank for64waves. Every FIA host interval falls inside
its ordered update pair; calls run on the host task-queue thread, not the begin/end
thread. This temporal pairing is not a runtime-handle join. Native source
attention_v1.py:update_graph_params explicitly brackets FIA.out with task updates
and records the ExternalEvent. All3072 device FIA tasks have graph model IDs.
Thus device computation is captured, but every layer still needs host preparation.
The frozen TraceLoom exact replay partition reports unsupported/no_exact_replays;
do not infer graph exclusion from visualization hierarchy alone. Query observations
are in swe-profile1/traceloom/fia-task-update-audit.json.

## Bounded device-length feasibility (2026-09-16)

Do not repeat the failed GE nesting attempt without a changed backend hypothesis.
`probe_device_lengths.py` runs installed torchair via torch_npu/dynamo, with
CompilerConfig.mode=max-autotune and tiling_schedule_optimize=True. No runtime
installation or donor edits. Existing910B2 device4, selected-device admission.

Native TorchNPU FIA v1/v2 expose SymInt[] lengths. TorchAir's separate `air` op
accepts device int64 Tensor lengths and runs tiling on device through GE.
Official source: https://github.com/Ascend/torchair/blob/master/docs/zh/ascend_ir/api/ops/npu_fused_infer_attention_score.md
The installed air-op implementation explicitly rejects eager/reduce-overhead mode.

Fixture: BF16 paged decode, four requests, Q16/KV2 heads, head128, page128,
32pages, table4x8; query cumulative offsets[1,2,3,4]. Actual lengths change
[17,129,257,513] → [31,128,300,700] → [8,33,64,256]. Both ge2/ge3 compare exactly
(max error0) against native FIA for all three sets, without task updates.
This is leaf decode evidence, not full prefill, whole-model or performance proof.

Preserved failures:
- device-lengths-ge1: incorrect4D KV layout rejected; GE expected BnNBsD, not
  [blocks,page,heads,dim]. Use native donor's3D [blocks,page,heads*dim] view.
- ge2: device-length tests pass; ACL nesting rejected first because the GE model
  was initialized on another stream: `Unsupport run graph with different stream.`
- ge3: initialized and captured on SAME non-default stream. Device-length tests
  pass; nesting rejected by runtime: `The stream cannot be used for model execution
  during the capture stage.` / `rtModelExecute ... reason=stream is captured`.
All supervisors exit1 due to these preserved failures; release records exist.
No nested ACL replay was qualified. A GE attention call per layer is not the
requested whole-wave host-free replay and must not be presented as completion.

## Historical decision (superseded below)

Fletcher was asked whether to preserve LiveModule/ACL and permit an alternative
attention kernel, or preserve original FIA and invest in whole-graph GE migration.
The recommendation is to keep the existing State/graph/retirement plane and first
qualify a compatible existing attention implementation, not rewrite matrix kernels.
Do not silently migrate the entire execution plane or claim the task complete.

An alternative read-only source intake exists at
runs/owned-wave/flash-attention-npu-intake, commit
cbd54b111a482fbeb005f2a66c420154a1f83421, from
https://github.com/MinghuasLab/flash-attention-npu (BSD3-Clause root license).
It offers scheduler_metadata generated on AICPU and fwd_kvcache that skips host
length synchronization when supplied. The snapshot is NOT built or NPU qualified.
CATLASS gitlink769cd40a8716b28650b6bebb08db4834eea4462f is not initialized.
Check page128, causal prefill, workspace lifetime and metadata producer capture
before adoption. Upstream setup.py auto-fetches CATLASS and compiles os.cpu_count()
workers; bound CPU/memory explicitly and never install into the donor environment.

## Native static-plan route supersedes the premature two-way choice

Fletcher redirected the inquiry to pre-wave planning. Do not require choosing a
replacement kernel or GE migration before testing the narrower native route.

The swe-profile1 device task name matches installed CANN9.0.1 artifact
FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da, tiling key
5000000000010200203. The installed config maps that artifact; its matching kernel
source selects SplitFuse::FAInfer<BF16,BF16,float,true,false,MASK_CAUSAL,TND>:
paged, non-split-KV. This is not extrapolated from the unrelated IFA all-vector path.

Installed source under opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/
fused_infer_attention_score establishes:
- kernel_common.hpp:GetQSBlockTile ignores kvSeqlen and returns128; GetKSBlockTile
  returns512. GetQNBlockTile depends on query length and GQA group size.
- flash_attention_regular.h binds actual Q/KV lengths to GM tensors and reads them
  inside runMainLoop for addressing, causal boundaries and KV loop bounds.
- For the selected non-FD path, firstBatchTaskNum/totalTaskNum describe Q/head task
  ownership. FD-only coreInfo/splitInfo/split reduction buffers are not consumed.
  Fixed per-bucket query geometry therefore supports testing fixed task counts
  while KV length changes. Do NOT change query population under a stale plan.
- Workspaces, layer-specific Q/K/V/output addresses and plan identity are separate:
  sharing geometry does not mean sharing live outputs or unfenced scratch memory.

Read-only upstream ops-transformer snapshot0051c52061ea33d8bbb05f759fb6b5bf0e37ab5c
at runs/owned-wave/fia-static-plan-audit/ops-transformer supplies the host-side
FillSplitCoreTilingData formula and workspace calculation. It is NEWER source,
not a claimed exact installed-host implementation. No installed CANN files changed.

Further useful lead: installed libopapi exports
aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize. Newer upstream public
array wrappers call that Tensor entry using FakeArray. An exported symbol alone
does NOT establish a supported device-pointer contract or matching ABI. Resolve
bootstrap tiling values vs runtime device length binding before calling it; don't
invent its prototype from the newer source. No raw binary launch or static-plan
NPU test has yet been performed. GE leaf success remains separate evidence.

Next discriminator: retain the native non-FD binary/plan, bind stable device
length buffers, capture without update waits, and vary KV length across page/tile
boundaries against the original FIA oracle. This is narrower than replacing
attention or migrating the whole graph to GE; the evidence does not yet prove
its binding path or performance.

### Call-boundary audit: plan versus binding (2026-09-16)

Important correction: the augmented DB records3072 GetWorkspaceSize calls and3072
execute calls, but only49 named InnerFusedInferAttentionScoreTiling intervals/rank
(2.379ms rank0 /2.719ms rank1). Do NOT describe this as3072 full tiling computations;
this is consistent with plan caching, although trace counts alone do not prove
which caching policy is used. The removable target is the per-layer host API/task
update protocol, not just the tiling arithmetic.

In newer upstream source, the ordinary execution route uses aclIntArray lengths,
then l0op::ConvertIntArrayToTensor calls executor->ConvertToTensor. Both length
inputs are declared ValueDepend(OPTIONAL), so changing a Tensor signature alone
does not remove host value-dependent planning. The actual capture-time allocation,
copy and address binding require runtime verification before reuse.

The discovered TensorGetWorkspaceSize/FakeArray call sites are specifically the
GetMaxWorkspaceSize route. FakeArray creates only a shape/dtype descriptor with
NULL data; it DOES NOT bind a caller-owned runtime device length buffer. Thus the
symbol is not evidence of an existing usable direct-device execution API.

The intended narrow boundary is: immutable per-bucket plan + layer-private stable
operand/output bindings + shared bank-local dynamic length/block-table inputs.
Capture must not retain a constant-length copy that overwrites the shared inputs
on replay. Keep no-update tests distinct from GE leaf feasibility and validate
cross-page/512-token/long-context transitions plus inactive rows before integration.

## Owned native launch now passes the leaf gate

Enter `../fia-plan/README.md` for the implemented pinned bootstrap/relocation
adapter and preserved failures. FIA launch's mixed-kernel FFTS prefix puts query
at byte8. Its five public placeholder relocations include inline Q/KV lengths;
removing ONLY those two relocations binds owned GM tensors without replay-time
constant-length copies. Original binary/config/opaque tiling and K/V descriptors
are retained, with separately owned workspace and stable operands.

`fia-static1` passes two-bank decode through1024; `fia-static2/run/q1` passes
Qwen0.6B head geometry through8192; `fia-static3` passes128/256 query prefill with
changing prefix/page mappings through4095. Every comparison to native FIA has
max error0. The failed later prefill stage of static2 is retained: argument size
varies with inline length count, not always3008. No new attention kernel, GE model
or per-layer task update participates in these replays. Whole-model integration
and performance are separate gates; see the adapter README for current receipts.

Whole-model closure is now recorded in `../fia-plan/PERFORMANCE.zh-CN.md`:
real Qwen3-0.6B TP1, two-order unprofiled controls plus old-owned control and a
separate official/TraceLoom profile. Static FIA has zero host FIA/task-update
calls for64 replays/1792 graph-member attention tasks. Retained elapsed means
2.9772s native ->1.7593s static owned; old owned2.6789s. Keep the differing
frontend boundaries, full-model token divergence, cold APC mismatch, startup
cost and lack of new30B/TP/DP/EP qualification explicit.


## 30B shipping gate exposes a missing FD plan

Do not generalize the small-model result or the capture-time non-FD label.
`../fia-plan/REGRESSION-30B.zh-CN.md` records the candidate-only complete trace
(95.827s retained, slower than reused baselines) and matched four-step TraceLoom
attention58.5->70.5us/layer. Native launch metadata at~4.5K selects FD510…/23blocks,
while the frozen short plan is non-FD500…/8 useful tasks. The installed source
and actual launch are the evidence; the newer host source alone is not an ABI
proof. FD core/split/reduction metadata must become wave-owned too. No repair or
new30B speedup is claimed; no model baseline was rerun.
