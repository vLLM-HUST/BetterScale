# Native DSV4 decode ping-pong integration (work in progress)

Scope: native DP8TP1 / TP8DP1, EP8, K5, stable text decode. Keep the existing
scheduler, prefill route, release pins, State ownership and eager/kept draft
choice. No N+2 scheduler rewrite. No performance or deployment acceptance yet.

## Transfer the invariant, not the old engine

LiveInfer's source reference is `AscendDSV4WaveExecutor` in
`src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py` in the retained
`/root/my-ascend-workspace/livemodule-main-integration` checkout. It binds two
independent metadata/argument trees to `device_wave_0/1`, uses ingress/compute/
egress events and retains ONE numerical continuation State. The host holds a
two-wave ledger. Graph-read completion and output-copy completion are distinct
reuse gates.

Native DSV4 already has the device feedback operation:
`update_num_computed_tokens_for_batch_change` computes previous GPU progress +
actual accepted count for continuing speculative rows. **Do not double-buffer
that tensor** or replace it with optimistic CPU progress. CPU lengths used for
metadata tiling and GPU lengths used for addresses are separate contracts.

## Bounded implementation stages

1. `--pingpong`: capture two PRIVATE decode call packets and two native graph
   entries per bounded small descriptor. Preserve storage aliasing and slices
   across metadata fields, including heterogeneous views. Scalars use the
   original native capture exemplar; never capture bounds from the first short
   live request. Prefill remains the prior native/FULL path. Target/draft/State
   operations stay on the native device stream. This first stage copies native
   metadata into private banks (D2D), NOT overlapped H2D.
2. `--pingpong-sources`: rotate pinned CPU ingress sources, preserving their
   logical contents and NumPy aliases, and wait for the same bank's prior input
   preparation event. Device progress, sampled/draft feedback and D2H accepted
   count destinations remain single-copy. Both dummy and actual waves use the
   same scope. Native GPU staging destinations / device ordering stay unchanged.
3. `--pingpong-continuous`: reuse the already tested stable-verification receipt
   cut, extended explicitly to DSA and the admitted seat limit. Authorize using
   conservative CPU bounds, keep device progress exact, submit target before
   consuming the previous CPU bookkeeping receipt. Retain the current source
   DMA completion fence before mutating its CPU logical view. Extend the shadow
   oracle to rebuild exact CPU metadata for its eager reference.

These flags are experimental and compose in that order. None is enabled by
`strengthen-dsv4 serve`; do not promote a partial bridge as the complete
LiveInfer three-stream protocol. A separate H2D lane is justified only after
source/destination lifetime qualification and an observed remaining gap.

## Evidence / unresolved gates

- 43 CPU tests pass, including independent bank/storage-view identity, repeated
  tensor aliases, rejection before partial refresh, same-bank-only host waits,
  coherent logical CPU state on rotation, and fail-closed generation poisoning.
- Run068: local card6 was admitted, then a foreign process arrived during
  initialization. The launcher stopped only this run; no model evidence.
- Run069: both native decode graph banks captured, but the runtime stream guard
  incorrectly retained the temporary startup-capture stream. Explicit post-start
  RPC binds the serving stream instead. The failed run is not acceptance.
- Run070: source-slot setup rejected the existence of an accepted-count event.
  Native K5 allocates it even without a proven active D2H write. It is now kept
  untouched as a feedback event, with its original wait, rather than asserting
  absence or removing it. This setup failure is not a graph-state failure.
- Run071: graph/source slots pass40 same-program checks, exact output/KV;
  two decode banks each replay19 times,52 total actual/dummy source scopes.
- Run072: adding the stable receipt cut passes40 checks;30 independently rebuild
  exact CPU metadata,35 late callbacks and36 bound bypasses. Single card only.
- Run071 also exposed an ownership mistake in the first packet inventory:
  copying the immutable full RoPE tables cost2.0GiB across four small banks.
  `get_full_cos_and_sin_dsa` returns precomputed model constants, unlike mutable
  position-selected runtime RoPE. Full compression RoPE and Hadamard are now
  explicitly shared with address/layout guards; mutable runtime RoPE stays private.
- Run073 DP8: idle ranks inherited the previous actual wave's `skipped` oracle
  flag and rebuilt dummy metadata after its source slot map was invalidated.
  Outputs matched, but the independent write map no longer matched. Reset the
  per-wave admission/reference flags at entry to the common source scope (used
  by BOTH actual and dummy waves). A CPU gate protects that reset. Run074 checks
  this diagnosis rather than loosening the KV oracle.
- Run074 DP8: all eight ranks pass40 checks, exact output and whole-KV bytes;
  immutable-table sharing reduces private packet backing to1,127,872bytes/rank.
- Run075 TP8: the packet layout guard rejected a shorter leading metadata view.
  Native DSACP FULL retains its capture view over the same backing. The bridge
  now copies that entire backing and permits only leading metadata-view changes,
  retaining dtype/stride/offset/backing-extent guards (never argument reshaping).
- Run076 TP8: all eight ranks pass48 checks against the unchanged native graph;
  output difference0, whole-KV bytes exact,43 exact-CPU-metadata checks/rank.
  Private packet backing878,776bytes/rank. TP uses strict HCCL,4 dummy seats and
  native K5 alignment. This is incremental graph-vs-graph, NOT a new native
  graph/eager equivalence claim; earlier native graph/eager uncertainty remains.
- Still required: matching real-weight control/candidate steady windows and
  compressed diagnostic timelines,
  memory delta, real retained OpenCompass quality before default promotion.

Original capsules live on hw3 under
`/workspace/my-ascend-workspace/runs/strengthen-dsv4-full-mixed/pingpong-*/`.
Never include profile/shadow time in the performance comparison. DP and TP must
match global active requests and actual target query rows; retain acceptance
and output-throughput accounting separately from step time.

## Isolate the benefit before promotion

`--pingpong-study` runs two within-engine repetitions at16 global requests,
96 target query rows, K5, eager native draft. Four policies share the weights,
State allocation, prefill route and reserved graph pool:

- `native`: original native target replay/fences and receipt handling.
- `cut`: previously qualified ordered replay + late receipt, plus CPU QLI
  metadata maxima for TP. No banked packets/sources in the executing path.
- `sources`: cut plus two pinned CPU source slots.
- `pair`: sources plus two private graph packets and alternating replay.

Warm each policy separately. Compare the first10 fully occupied FULL cycles,
not total cohort time alone; keep every request's actual output/wave count.
`cut -> pair` is the incremental banking effect, NOT `native -> pair`.
The retained native reference graph exists only under study/shadow flags.

Run077 exposed a real-weight host cost hidden by the four-layer fixture: TP8
`pair` took336ms/cycle versus about71ms native. The refresh walker treated
metadata shared across layers as a tree, repeating validation/temporary-view
creation. Run078 was stopped by its verified owned launcher during startup;
there is no DP performance evidence from it. The walker now visits each
(source,destination) identity pair once; a CPU test protects both DAG reuse and
alias-split rejection. Runs079/080 retest this specific diagnosis. No slowdown
is dismissed as noise, and the banking bridge remains opt-in.
