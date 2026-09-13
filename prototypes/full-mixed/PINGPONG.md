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

### Stop per-layer host copies at the graph boundary

Runs079/080 remove the repeated DAG walk but still find about10ms of pair
submission overhead: DP pair71.3/72.8ms versus native62.0/61.7ms; TP pair78.4/
79.3ms versus native67.7/68.2ms (two within-engine repeats, first10 occupied
cycles). This rejects the runtime packet-refresh bridge as a performance patch.

`--pingpong-captured-copy` instead records the fixed native-input-backing copies
inside each target bank's graph. It relies on the SAME persistent capture input
addresses as the untouched native FULL graph, holds those source allocations,
and removes the metadata walk/per-copy host dispatch from replay altogether.
It still does not create an independent ingress stream. It is an explicit
alternative to runtime refresh, not a relaxed numerical acceptance threshold.

Run081 TP8 passes48 checks/rank and run082 DP8 passes40 checks/rank after this
change, with exact outputs/KV. Runs083/084 measure the real-weight effect.
The observation now attaches actual query/request counts and dummy status at
native mode selection; padded cross-DP buckets are never actual work counts.
Older early occupied windows use their retained real-schedule ordinals; requests
can arrive one tick apart, so take a common later window only when every rank
still has all16 global requests active. Do not assume rank-local window starts
are equal or include native drain dummy waves in the throughput denominator.

## Fletcher's correction: enter at shadow construction, not after preparation

The captured-copy bridge is still an endpoint adapter. It does NOT implement
LiveInfer's host shadow protocol, even if its copies are captured. Stop layering
more endpoint machinery onto it; retain it as a bounded negative/control result.

Read the retained LiveInfer checkout's:
- `src/livemodule/runtime/invocation.py:134`: `shadow_replay` runs the recorded
  explicit construction program and publishes its ingress before PREPARED.
- `src/livemodule/runtime/shadow.py:GraphIngressEffectTrace`: graph-owned stable
  host sources and device destinations, ordered H2D effect matching and staging.
- `src/livemodule/arch/ascend/request_parallel/dsv4/speculative.py:752`:
  `_construct_wave_routes` makes CPU owner geometry into ingress; actual lengths,
  addresses, validity, rotary selection and descriptors remain device work.
- `src/livemodule/arch/ascend/request_parallel/dsv4/wave_executor.py:186`:
  ingress waits the SAME bank's graph-done, compute waits ingress-ready and that
  bank's old copy-done, then egress runs separately and retains host source leases.

This is NOT the same as our numerical same-state `*_shadow.py` oracle. Do not
confuse the name with validation, and do not move actual speculative progress
to host merely because some metadata are host-constructed.

### Donor seam to change

Pinned MRV1 `_prepare_inputs` (814) interleaves CPU projection, H2D, and real
State-dependent GPU operations. Its larger `synchronize_input_prep` scope (1773)
also includes state bookkeeping, DP shape agreement and metadata building.
Wrapping the resulting call tree cannot extract that producer dependency.

For admitted stable K5, separate:

| Responsibility | Producer / lifetime |
| --- | --- |
| Request identity, scheduled width, block leases, previous-row mapping, CPU tiling bounds | Host construction; snapshot into bank-owned pinned ingress, not mutable runner bookkeeping arrays |
| Actual accepted counts, sampled/draft IDs, computed-token progress, KV | Single continuation State on device; preserve native arithmetic and ordering |
| Exact positions, lengths, slot mapping, SAS/QLI descriptors, selected RoPE | Device derivation ordered after previous feedback and before the target; not speculative host guesses |
| Output receipt and CPU accounting | Separate retained egress/source lifetime; do not recycle pages or mutate in-flight H2D sources |

The minimal credible integration is producer-level host snapshot + explicit
publication and device derivation, with per-bank reuse events. Preserve native
DP collective order/shape agreement and prefill/turnover fallback. DP and TP
share the protocol but NOT the same metadata builder or collective schedule.
A generic copy of every field, a FakeTensor walk through the whole model on each
step, and an extra end-of-preparation packet clone are not substitutes.

Open implementation hinge: identify the exact H2D input set and separate the
late CPU bookkeeping destinations from DMA source slots. In particular native
`num_computed_tokens_cpu_tensor.to(device)` is a host budget input to the GPU
correction kernel, not the numerical progress State. `num_computed_tokens` on
GPU must remain single-copy. The existing current-input-DMA host fence can only
be removed after this ownership separation, not just after adding a ready event.

### Closed endpoint-adapter measurements (083/084)

Both layouts use16 actual requests,96 target rows,K5,8GiB KV/rank and eager native
draft. Two within-engine repetitions; numbers are occupied cycle medians in ms:

| Layout | native | receipt cut | + host source slots | + captured-copy graph pair |
| --- | --- | --- | --- | --- |
| TP8DP1 | 69.31 / 69.16 | 67.35 / 68.60 | 67.57 / 69.19 | 69.23 / 69.27 |
| DP8TP1 | 62.25 / 60.57 | 62.54 / 61.63 | 62.81 / 62.63 | 62.55 / 62.94 |

Capture removes the adapter's large host-launch tax but does NOT establish an
incremental speedup. Do not promote this bridge or use cohort-time variation as
a banking win. Packaged defaults and main remain untouched. These measurements
are not a test of the producer-level LiveInfer shadow protocol.

### Producer host-projection audit

`--producer-shadow-audit` is one diagnostic stable `_prepare_inputs` call, not a
new serving path. It uses the same host-real/device-fake separation as LiveInfer
to inventory inputs and find explicit device action boundaries. Do NOT treat
FakeTensor's disabled fallback as protection against arbitrary direct launchers.

Run085 entered native block-table `compute_slot_mapping`, which directly calls a
Triton kernel outside Torch dispatch, and ended with507035 / invalid MPU address.
The owned process was reclaimed; release records show no residual NPU processes.
No successful projection or correctness claim comes from this run. Source shows
this direct launcher consumes fake positions. Run086 explicitly excludes that
pure device-write action from host projection and preserves its persistent
output contract; this is a changed boundary, not a retry of unguarded fake mode.
The audit checks progress/input/position/length/slot-map device buffers unchanged
and never modifies production defaults. Further native direct launchers, if any,
need explicit boundaries before they can enter host shadow.

Run086 completes the input-preparation host projection with the explicit
slot-mapping boundary:115 virtualized Torch operations, no observed readbacks,
and unchanged progress/input/position/length/slot-map device buffers. Its21-copy
H2D inventory is incomplete: it omitted ten `aten.to.dtype_layout` occurrences
(the detector now recognizes that overload). Do not call21 the complete ingress
count. This diagnostic is discovery, not the serving implementation.

Fletcher explicitly chose manual construction of the actual shadow protocol.
`decode_shadow.py` now implements a bounded producer instead of replaying that
FakeTensor audit at runtime:
- explicit CPU K5 geometry and mutable resource/budget snapshots;
- two pinned ingress/source and device-input banks, their own H2D stream;
- H2D-source reuse waits old upload, device bank reuse waits old consumption;
- native accepted-count arithmetic and direct slot-mapping kernel in a real
  captured device-preparation program, leaving progress single-copy;
- current native sampled/draft feedback handed to the preparation program on the
  compute stream, never guessed from CPU budgets;
- prefills, turnover, masked partial queries and hybrid feedback remain native.

This first producer boundary covers `_prepare_inputs`, NOT the entire downstream
DSA/DSACP metadata builder or a complete three-stream donor runtime. Existing
late-receipt/current-input-DMA guards remain until the remaining H2D owners are
separated. Run087 checks the new producer's device fields and complete sampler
metadata against original preparation from identical State. No new performance
claim precedes that gate.

Run087 caught an exact sampler **dtype** contract error (target logits indices
were int64, native uses in-place int32 arithmetic). No tolerance relaxation:
geometry now emits the native int32 result. Run088 passes12 original-preparation
checks, including initial capture, both banks, and stable2-request to1-request
shapes. Run089 additionally captures the device metadata program and passes38
original-native target output/KV checks (all output differences0, KV bytes exact)
and12 input/sampler checks. Both are single-card dummy gates, not throughput or
TP/DP qualification.

`decode_metadata.py` refuses Torch-visible host/device transfers in the captured
metadata stage; reviewed direct native Triton metadata operations execute with
REAL device tensors. Tiling maxima use the admitted model limit, while exact
lengths and positions stay dynamic on device. Cache identity includes actual and
padded geometry plus CPU source-carrier identity; target metadata is cached but
DSpark receives a fresh common envelope and refreshed per-group views. Native
prefill/dummy/turnover and original exact-bound oracle builders remain separate.

The next scoped fence removal follows the actual callback writes: native
`correct_spec_decode_token_counts` changes request state and CPU computed-token
budgets only. The explicit producer already snapshots that budget into owned
pinned ingress, so its admitted waves can retire the old receipt without waiting
for CURRENT native input DMA. The real receipt wait itself remains in the native
callback. Native/fallback waves retain the current-DMA guard. This change is
CPU-contract-tested and enters the next multirank gate; it is not yet a measured
continuous-replay gain.

TP2 run090 exposed a missing activation, not a new QLI algorithm gap: the native
DSACP builder still extracted NPU scalars because this new entry had not enabled
the already-qualified CPU-QLI hook. Capture rejected that readback (107027).
The metadata entry now explicitly activates CPU-QLI on TP layouts; DeviceOnly
also rejects `_local_scalar_dense` before a backend readback. CPU tests cover
local host/device work, H2D, D2H, cross-device copy and scalar extraction.
