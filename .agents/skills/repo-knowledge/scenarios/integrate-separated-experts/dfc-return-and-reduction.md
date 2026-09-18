# Borrow DFC return pipelining without regressing the client footprint

Source audit, 2026-09-18; no new hardware measurement. Use when optimizing
Qwen38 `neural_collect_fused`, not when choosing a GEMM backend.

## Exact source, and the comparison boundary

Upstream checkout `/workspace/strengthen-dsv4/upstream/vllm-ascend` at
`9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`. Read under
`csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/`:

- `dispatch_ffn_combine_bf16_kernel.hpp::CombineV2` and the following
  `DispatchAndCombine` completion/unpermute tail.
- `utils/block_epilogue_pertoken_v2.hpp::operator()` and `Finalize`.
- `unpermute/moe_token_unpermute.h::CalMultiOutToken`, `CalPartOutToken`,
  `CopyTokenIn`, `CalToken`; sibling tiling sets `buffer_num=4`.

The quantized `dispatch_ffn_combine/op_kernel/` sibling has both CombineV1/V2;
V2 uses16-row subtiles versus32 in this BF16 branch and includes dequantization
in its return epilogue. Both inspected paths return to peer memory, synchronize,
then run local unpermute. This audit is of pinned source, NOT proof of which
branch the historical separately built lab DFC binary selected. FAIR-DFC.md's
EP2 BF16/H2048/K8 timing is not directly comparable to Qwen38 E3/H2560/K10.

## What DFC actually does

1. Combine follows the down-GEMM tile assignment. AIV waits on the corresponding
   expert/Cube progress flags before reading output, rather than waiting for all
   experts to complete before starting every transfer.
2. Its epilogue alternates two UB slots. Each slot has its own completion event;
   the next MTE2 read waits for that slot's previous MTE3 use, not a PIPE_ALL.
   BF16 return copies local down output to UB, then writes the relevant tile
   slices DIRECTLY into each source rank's shared output window. Counts/prefixes
   determine contiguous source segments within each expert. Finalize drains both
   slots. It is a producer-driven PUSH, not remote fetch inside unpermute.
3. Cross-rank completion still precedes unpermute. This implementation is NOT
   fully asynchronous arrival-order token reduction.
4. Local unpermute batches route indices/probabilities for a group of tokens,
   then visits each token's top-k in index order. BF16 rows are cast to FP32,
   weighted and accumulated, then rounded once to BF16. The first contribution
   initializes the sum; subsequent contributions add. Input/output queues separate
   copy, vector and output lifetimes. Runtime allocates four input buffers, but
   the source loop still calls CopyTokenIn then CalToken per contribution: do not
   advertise four outstanding remote pulls or a four-route prefetch schedule.

## Our current difference

`prototypes/attention-client/qwen38/client_reduce.cpp` first joins all owners,
then token-owning AIVs pull each route directly from a SERVER-local output window
and reduce in FP32. Server SEND is a local staging copy, not the D2D return.
Thus the measured collect span includes residual server wait AND remote transport.
DFC's local unpermute duration alone would omit the remote return it already did.

Our fused collector already avoids a client `[tokens,K,H]` HBM staging allocation.
Blindly transplanting DFC's whole push/unpermute arrangement restores that memory
cost. At512 physical rows,K10,H2560,BF16 the payload is25MiB/layer, not measured
bus traffic; the final reduced output is2.5MiB.

The immediate implementation seam is more modest: each route calls IO.Read,
which waits MTE2_S even though payload will be consumed by Vector, then PIPE_ALL,
Cast/Muls/Add, then PIPE_ALL again. IO.Write also synchronously drains MTE3 and
PIPE_ALL. This is correct but serializes reusable-buffer lifetimes unnecessarily.
We can borrow DFC's slot ownership, not necessarily its communication direction.

## Smallest next experiment

Keep all-owner join, generation, retirement, route order, Muls then Add semantics,
BF16 output rounding and wire layout unchanged. First measure a **fused-specific**
all-results-ready collector under FULL replay. Existing probe `ready_collect_ms`
is legacy collection and cannot supply this control.

Candidate: two BF16 UB input slots, prefetch route k+1 while Vector reduces k;
use MTE2_V and V_MTE2 dependencies to protect input readiness/reuse. Keep FP32
scratch and accumulator disjoint, and a separate output lifetime until MTE3 is
done. Metadata scalar reads may retain MTE2_S. Do not just delete PIPE_ALL from
IO: its shared scratch also carries generation flags and other protocol writes.
AtH2560, two BF16 rows + two FP32 rows + one BF16 output use35KiB, leaving room
within64KiB for aligned metadata, but addresses/events require an explicit map.

After this isolated test, compare live-server leaf and whole-model shadow; the
end-to-end gain can be small if server waiting dominates. Include changing inputs,
nonuniform weights, repeated experts, empty owners, tail token counts and repeated
generations. Preserve fixed summation order: earlier online collect has an
unresolved whole-model shadow failure followed by a pass; do not label it fixed.

Only after measurement consider route-ready pulls before all-owner DONE. That
needs readiness plus final drain, and in-order reduction can cause head-of-line
waiting. Push is a separate memory/protocol tradeoff, not a prerequisite for the
first optimization. No speedup or production default change is claimed here.

## Implemented bounded successor

The two-slot fixed-order prototype is now in
`prototypes/attention-client/qwen38/client_reduce_pipeline.cpp`. Read its sibling
`COLLECT-PIPELINE.md` and `collect-pipeline-result.json` before rerunning the leaf.
At1024rows, warm ready collect falls1.029->0.921ms and full real-layer leaf
3.745->3.640ms;24 changed-input comparisons are exact. This is a modest measured
consumer improvement, not evidence that the remaining server wait is removable.
New ABI-marked builds default to the pipeline inside fused mode; `=0` keeps
the serial control. Explicit `QWEN38_PIPELINED_COLLECT=1` requires its ABI-marked
binary; older binaries retain the serial path when unset.
The same candidate also passes real48-layer+MTP A2TP1+E3 FULL continuation:
12 eager/captured shadows each129/129exact, four capped two-turn SWE traces,
and all roles exit0. This is a correctness gate, not a matched throughput gain.

The sibling COLLECT-PIPELINE.md now records the remaining server seams and
`collect-server-seams.json`: broad-hit1024rows has403–470us convert/export and
429–448us post-fetch/pre-pack coordinator intervals. They are not pure kernel
times. This diagnostic closure still publishes route-ready flags unused by
fixed-order collect; standard builds default those flags off. Preserve the ABI
when testing their removal, and do not confuse broad-hit GEMM time with the
earlier hot10 fixture.

## Server-side successor

`prototypes/attention-client/qwen38/EXPORT-PIPELINE.md` now owns the SEND follow-up.
Two input/scale and output slots pipeline the unchanged INT32/scale/BF16 math
inside64KiB UB. No-route-ready control versus candidate at1024rows gives
3.425->3.228ms complete leaf, with convert/export intervals roughly370–386us
falling to191–207us.24 full output tensors match bitwise across server builds.
The real A2TP1+E3 full-model gate also passes12 shadows each129/129exact.
These are bounded observations; the capped model run does NOT establish an
end-to-end gain. New ordinary builds now default to pipelined export;
`--no-pipelined-export` keeps the serial control. Route-ready builds default to
the old backend; explicitly combining both still fails pending that separate gate. Do not accidentally enable the
unqualified online notification branch when reusing the helper.

After default promotion, candidate broad1024-row post-fetch/pre-pack is still
429–437us and PACK225–236us. Prefer investigating serial coordinator Group and
row/scale copy organization before blaming the genuine wide-expert GEMM traffic.

## Route preparation: paid negative result

Before touching coordinator metadata, read
`prototypes/attention-client/qwen38/ROUTE-PREPARE.md` and its receipt. UB-tiling
scalar route IDs regresses the complete1024-row leaf3.275->3.406ms. Preclassifying
owner-local IDs shortens Group but leaves the complete leaf unchanged; counting
at admission removes a scan yet gives only1.5–1.7% at1024rows and no consistent
smaller-shape benefit. No default or production-builder change was adopted.
All three candidates match24 complete control outputs. The apparent430->170us
preparation improvement is NOT260us of end-to-end saving. A future attempt must
measure acceptance plus grouping/publication, not relocate a named interval.
