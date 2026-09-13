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

- 41 CPU tests pass, including independent bank/storage-view identity, repeated
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
- Runs074/075: DP8 corrected same-program shadow and TP8 unchanged-native-graph
  shadow queued on hw3. TP uses strict HCCL,4 dummy seats and native K5 alignment.
  The latter oracle is incremental graph-vs-graph, NOT a new native graph/eager
  equivalence claim; source retains the earlier native-graph/eager uncertainty.
- Still required: exact-metadata bound shadow, all-rank DP/TP state checks,
  matching control/candidate steady windows and compressed diagnostic timelines,
  memory delta, real retained OpenCompass quality before default promotion.

Original capsules live on hw3 under
`/workspace/my-ascend-workspace/runs/strengthen-dsv4-full-mixed/pingpong-*/`.
Never include profile/shadow time in the performance comparison. DP and TP must
match global active requests and actual target query rows; retain acceptance
and output-throughput accounting separately from step time.
