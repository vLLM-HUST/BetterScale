# Remaining preparation and return dependencies

Source and existing-receipt audit after46822f8. No fresh NPU run or new speedup.
Run preparation_audit.py on the hw0 remote2 capsule from FAIR-DFC.md.
It joins each slot through SEND completion before computing per-wave intervals.

| Median, us | Server0 | Server1 |
|---|---:|---:|
| Last FETCH end → PACK command |19.01|19.04|
| First FETCH start → UP command |66.38|66.33|
| UP end → DOWN command |0.26|0.35|
| DOWN end → SEND end |24.56|23.25|
| SEND end → next wave's first FETCH |67.82|110.50|

All24 waves are paired in this receipt; none of the23 next preparations overlaps
its preceding return. These are independently aggregated medians, not an additive
latency decomposition. The last gap includes publication, waiting for the OTHER
server, client collection/reduction/republication and server admission/descriptor
work. It is not measured pure communication, CPU time, or67/110us removable idle.
Faster server1 waits longer: do not optimize it by deliberately making its math
slower or attribute the entire wait to its own kernel.

## SEND is local staging, not the cross-device return

Follow addresses, not names:

1. device_joint.serve allocates each client's `local` export on the SERVER device.
2. persistent_service.serve passes `[c['local']]` as PersistentEngine.outputs.
3. ReturnStreaming copies down results to those server-local allocations in route
   order. The earlier prefix optimization overlaps this LOCAL copy with down.
4. Source ClientBank config holds imports of those allocations in transport.peers.
5. kernel.cpp::neural_collect first waits for BOTH server DONE generations, then
   copies owned route outputs across devices into client raw[0].
6. neural_retire advances the client counter after collection; native unpermute
   performs the weighted route reduction. Graph ordering keeps the following
   request from overwriting live inputs/outputs before the preceding body completes.

Thus existing early-return evidence does NOT show down/D2D overlap. Full server
DONE precedes any cross-device result fetch. A route-expanded buffer still exists
on BOTH sides, followed by a separate client reduction. For broad32/source,K8,H2048,
BF16: each server stages1MiB of route outputs for the two clients; each client
collects1MiB across the two servers and emits128KiB of reduced output. These are
logical payload sizes, not measured bus bytes or a bandwidth lower bound.

The smallest next candidate is **collect whichever owner has completed**, rather
than joining both before any collection. Each mover can track a two-bit completion
mask, poll each unfinished owner and copy only that owner's routes once. Keep
final kernel completion and retirement after both owner subsets; no partial
attention result, early slot reuse or new server protocol is permitted. Owner0
first unconditionally is not enough: it can simply pick the slower server.
This could hide fast-owner copying while the slow owner finishes, not hide all
cross-device return under down. It needs a same-work NPU control before adoption.
Fusing collection with top-k reduction is a separate arithmetic/ownership change.

## Internal mode builds obsolete segmented catalogs

persistent_vector.cpp::Group receives cfg[14] through a bool argument. Both
external two-GEMM segmentation(mode1) and continuous internal traversal(mode2)
therefore build two complete128-entry int64 cumulative catalogs, in addition to
the full catalog and two route maps. Five Transfer.Write calls serialize reuse of
one UB buffer, each waiting for MTE3 completion.

persistent_cube.cpp distinguishes those modes correctly: mode1 consumes ptr[9/10]
as actual GEMM catalogs, whereas mode2 uses ptr[6] for BOTH complete GEMMs and
reads ONLY ptr[9][127] as its scalar split boundary. No reader uses the rest of
those two segmented catalogs in mode2. A scoped simplification can publish just
the boundary for mode2 while retaining both catalogs for mode1. Keep generation
publication AFTER its DMA completion; do not drop the invalidation on the Cube
side. This removes redundant work but does not prove it saves the whole19us.

Do not overlap Group with FETCH by only moving the function call: movers read the
same source map buffer Group rewrites, and a post-fetch admission snapshot may
still add a source and change packed destinations. Descriptor separation or a
frozen-admission boundary is required for that different optimization.

The subsequent opt-in implementation and acceptance are in
[ROUTE-PULL.md](ROUTE-PULL.md), including client-owned fused reduction.
