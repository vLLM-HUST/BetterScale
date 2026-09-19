# Joint-wave transplant: bounded native-body evidence

Enter here before re-investigating whether target + sampler + DSpark can share
one graph, or promoting the protocol into the serving Worker. Implementation and
reproducer: `prototypes/joint-wave/README.md`. This is an isolated oracle, not a
change to the released BetterScale serving path.

## Evidence identity and envelope

2026-09-16, repository base `b0e6ad9`, native vLLM
`752a3a504485790a2e8491cacbb35c137339ad34`, vLLM-Ascend
`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`. Each capsule below snapshots actual
prototype sources and records four installed/native source comparisons; the base
commit alone does not identify the uncommitted prototype.

Capsules: `/workspace/strengthen-dsv4/runs/joint-wave/run00N`, receipts under
`engine/{oracle,result}.json`, full log under `run/run.log`. Local 910B2 physical0,
TP1, dummy four-layer DSV4 SWA/C4/C128 + one-layer DSpark K5, greedy one seat,
prompt128/output32. Native numerical bodies have `enforce_eager=True`; the candidate
captures one outer graph, not nested native FULL dispatch.

Observed progression: run002 exact snapshot replay; run003 one capture and six
consecutive native-reference steps; run004 six autonomous replays with exact
receipt/final raw KV and delayed bank copy; run005 also one-token EOS/length stop
and three unchanged-KV terminal drains each. Native dummy acceptance was always
two tokens; do not claim exhaustive acceptance-length or real-weight coverage.

**Latest complete gate: run009 PASS, launcher exit0.** Six independent reference
replays and six autonomous replays use one capture, max two outstanding waves,
end152 assigned grant and one backpressure retirement. All receipts/final KV
match; EOS/length drains and bad-generation/insufficient-grant poison tests leave
KV unchanged. Fourteen CPU storage/transport/window tests pass. The capsule's
`source/` is the exact executed implementation, not the earlier run005 source.

## What the port actually owns

Graph: device metadata/positions/slots → native target/logits/rejection sampler →
native DSpark → device continuation and banked receipt. The numerical region is
checked with `DeviceOnly`. Cursor means **next target input position**, distinct
from LiveInference's emitted-anchor count. Host authorizes resources, does not
reconstruct anchor/draft from receipts. Bank ingress waits prior graph reads;
replay waits grant readiness and prior D2H; copy waits graph completion.

Invalid sequence/generation/grant poisons a resident. Terminal/inactive masking
must cover both target and draft slot mappings and lengths. Merely masking the
output token does not protect compressed/state KV. Whole backing-byte comparisons
include aliasing and state-cache bytes, not just the visible MLA tensor slice.

## Resource seam exposed by fail-closed checks

Do not derive token coverage as min(block_count * block_size) across raw groups:
C4/C128 rows count compressed slots (run006). Convert each homogeneous group's
assigned slots into original positions. Do not substitute allocation-pool capacity
for request-owned block assignments.

The tiny sliding/state groups are the limiting grants in this fixture: an 8-slot
group yielded end144 at positions128/130/132, then end152 at134/136/138. Run007
failed at134 because the adapter retained the original grant. Refreshing from
native allocated rows fixed the six reference replays (run008), but its host
correctly rejected two-wave lookahead past152. Two outstanding waves is a ceiling,
not a duty: draining another receipt can tighten the horizon without more KV.
For K5, authorize against receipt-derived position + outstanding*6 + next6 + draft5.
If no outstanding wave can relieve the shortage, allocation renewal is required.

The autonomous fixture starts with the original token/KV seed and the last native
reference step's already allocated tables/grant. This is a bounded stable-table
handoff, **not** evidence of online replenishment, safe table rebinding or fresh
request admission. Do not silently remove these distinctions when integrating.

## Still required for serving

New-generation admission and old-reader fencing; real allocator renewal and
prefill/mixed handoff; finite startup graph catalog; exact native count-publication
and output ownership; multi-seat/TP/DP qualification; real weights and acceptance
variation. No performance claim: diagnostic KV copies and synchronization dominate.
Native serving remains the independent reference and its KV is restored after each
oracle. No installed package or production Worker hook was changed.

## Qwen3-30B-A3B parallel compatibility (September16)

Before generalizing the DSV4 continuation to ordinary Qwen attention, read
[the Qwen oracle and matrix](../../../../../prototypes/joint-wave/qwen/README.md).
Real48-layer BF16 passes fixed-snapshot joint target/logits/sampler capture in
TP2/DP1/EP2, TP1/DP2/EP2 and TP2/DP2/EP4; every rank compares6 native snapshots
with2 graph replays each, all sample IDs and complete KV bytes exact. The combined
arm uses18/19-token English prompts and changing IDs, with exact TP peer agreement.
All use native PA decode and ALLGATHER MoE; **not MC2, AllToAll, FIA or skew**.
Capsules under `runs/joint-wave/qwen-{tp2,dp2,tp2dp2}-real1`; each has complete
all-rank receipts and exit0, plus the executed source snapshot.

Do NOT report these as the DSV4 single-capture/six-wave continuation gate. PA's
Tensor-typed context_lens carrier is CPU in this builder. Moving it to device in
the reduced TP2 probe fails native ATB `PagedAttentionOperation setup failed`
(qwen-tp2-smoke3), before joint capture, whereas unchanged-carrier snapshot mode
passes (smoke4 and real matrix). This establishes that direct carrier substitution
is not a working adapter in this runtime; it does not prove no device-capable
attention implementation exists. Native graph task updates or a device-aware ABI
are separate follow-on routes. Keep this negative observation beside the positive
TP/DP/EP composition evidence rather than calling Qwen fully compatible.
