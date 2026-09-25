# Proposed first State-ownership vertical: small Qwen35, one NPU

2026-09-25 research result. Fletcher selected a small same-family single-card
model instead of starting at TP2. This is the proposed declaration/activation/
acceptance cut, not implementation or authority to launch a device job. No
model weights were downloaded and no accelerator was initialized here.

## Select the smallest representative model, not a tiny fake architecture

Candidate: official Qwen/Qwen3.5-0.8B, TP1, BF16, text-only, greedy, MTP2.
The metadata-only observation is in [small-model-metadata.json](small-model-metadata.json),
including immutable revision and official config/weight-index URLs. The index
contains actual MTP parameter names, not merely an MTP config flag.

Geometry: target18GDN+6FA, one draft attention layer; hidden1024; FA q8/kv2,
head256; GDN key16/value16, head128, convolution kernel4. Thus GDN qk:value
ratio is1:1, unlike the35B TP2 local8:16 ratio. Inspect the optimized kernels'
admission and independent recurrence before asserting coverage. Do not widen
production admission merely to run this pilot. Text-only loading must correctly
handle the multimodal checkpoint namespace and exclude vision execution.

`/workspace/models` currently contains35B, not this small model; a bounded scan
of named local model/workspace roots found no small candidate. This is not an
exhaustive host-wide model inventory. The metadata reports1,746,882,752 weight
bytes including the full indexed artifact; this is NOT a runtime memory budget.

Native comparison geometry would imply a1024-token common block from
1MiB SSM /1024-byte single-K token; kernel page128 gives8kernel pages. This is a
source-derived check, not startup evidence. The new design need not inherit
that coupling: keep FA page128 and choose checkpoint interval1024 explicitly
for the first comparison. Treat any native startup disagreement as a failed
geometry assumption, not permission to patch around it silently.

Proposed fixed test envelope: two seats, MTP2, context4096, bounded prefill
chunks, two invocation banks, a small explicit checkpoint quota. Fixed capacity
comes first; automatic fitting and throughput optimization are outside this
vertical. Device availability/admission remains a separate future probe step.

## Resident-seat correction takes precedence

Request lifetime and resident-seat lifetime are different. The table below is
an earlier candidate census, not a mandate to allocate a separate checkpoint
pool: read [resident-seat-policy.md](resident-seat-policy.md) first. Running GDN
state may remain in its seat after request completion. Retained checkpoint lanes
are conditional on a demonstrated rollback/branch/eviction requirement.

## Declaration census — owners first, domains second

These are proposed semantic declarations, not frozen Python API names.
Preserve exact per-layer identities; never wrap whole donor arenas as one State.

| Contents | Semantic declaration owner | Capacity/addressing | Content lifetime |
|---|---|---|---|
| Target FA K, V | Each target attention leaf | Token-page domain;128-token native pages | Referenced prefix pages until last lease/eviction |
| Draft FA K, V | MTP attention leaf | Explicit draft token-page domain (may couple counts, never addresses) | Valid shifted-input prefix and current draft work |
| GDN convolution history | Each target GDN leaf | Fixed seat domain; one history with speculative extension | Mutable running request |
| GDN recurrent candidates | Each target GDN leaf | Fixed seat domain;K+1 separately addressable matrices per seat | Candidate selection until next permitted overwrite |
| GDN saved conv/recurrent checkpoint | Same GDN leaf or its registered checkpoint child | Shared checkpoint-slot domain, distinct physical lanes | Immutable published prefix until last reader/eviction |
| Accepted selection, anchor/progress, request generation | Qwen MTP continuation owner | Fixed seat domain | Request incarnation, not transport-bank lifetime |
| Next proposal IDs / necessary hidden continuation | Actual MTP/continuation consumer owner | Bounded seat domain where cross-wave persistence is required | Until next verification/commit |
| Wave sequence / failure latch | Root control child | Exact capacity1 | Root generation |
| Block-table ownership/lease bookkeeping | Explicit request/cache authority | Addresses the admitted domains | Qualified request/checkpoint leases |

Conv and recurrent candidates do not have the same multiplicity. For this
geometry a native-form conv history is5*6144*2=61,440bytes/seat/layer; one
recurrent matrix is16*128*128*4=1,048,576bytes, with three candidate matrices.
Do not allocate three whole conv histories just because recurrence has three
candidates. K-V versus V-K recurrent ABI must remain explicit even though the
matrix dimensions are both128.

A checkpoint has one recurrent matrix plus the meaningful convolution window.
For the first cut retain an explicit declared checkpoint representation; do not
invent a compression/transcode optimization. Whether to retain a five-row
backing with only three meaningful history rows or normalize to three is still
an implementation hinge for the actual restore consumer. Do not hash/compare
unused tail bytes as semantic state.

Checkpoint slot IDs can be shared across layers as a vector handle while each
layer owns distinct bytes. Cache lookup/leases/publication are not StateTensor
responsibilities. For the pilot choose a fixed bounded policy; don't implement
an external CPU pool or a general replacement scheduler.

Banked ingress, metadata, egress, temporary activations and task-local buffers
are not automatically StateTensors. Classify by cross-call semantic lifetime,
not by whether torch allocated memory. A graph transport bank is not a GDN
candidate bank and not a request generation.

## Existing Qwen implementation: reuse the contract, not its entire policy

LiveInference05ac1541 `serve/qwen35/mtp/root.py` already promotes target
attention/GDN, declares accepted_tokens and gdn_state_anchor, and composes a
draft child plus SlotContinuation. It demonstrates exact State binding and
separation of output count from accepted-input selection.

But it is not this new vertical ready-made:

- Target attention and GDN currently join PhysicalBlockArena's elastic domain;
  our seat/checkpoint split is not already implemented there.
- `llm/qwen35/mtp.py:Qwen35NativeMTP` creates linear key/value caches with
  register_buffer, not register_state. Its root keeps complete target hidden
  history and `_initialize_draft_state` rebuilds draft context from that history.
- `resources.py:SlotContinuation.target_hidden_history` scales with context and
  hidden width. The BetterScale capsule uses its shifted-input/delayed-hash
  protocol instead of this full-history sidecache. Do not silently import a
  different memory/restore policy by transplanting the root wholesale.
- The named Qwen GDN State program inspected is implemented for CUDA, not an
  already qualified Ascend port.

The entry decision is therefore bounded composition of existing lifecycle and
State primitives with the actual Ascend numerical path, not importing a new
universal engine or copying LiveInference into a second independently maintained
runtime. Exact package/pin delivery remains to be selected at implementation.

## Initialization handoff — one owner at each stage

| Stage | New authority / existing facility to reuse | Native authority that must stop |
|---|---|---|
| Resolve model/weights and physical execution port | Explicit pinned loader/port; construct numerical modules once | No implicit engine path that also starts a second State lifecycle |
| Declare all model/control State | Module tree + exact StateDomains | No KV-spec page equalization/shared_by as placement truth |
| Admit fixed pilot capacity | Existing backend planner; explicit finite budgets | No get_kv_cache_configs-derived common pool count used as the new capacity oracle |
| Realize and validate all lanes | StateBackend + root generation transaction | No runner raw allocation/reshape or whole-arena adoption |
| Bind numerical consumers | Generation hook loans exact lane tensors/typed metadata | No independently mutable module kv_cache ownership |
| Build dependent descriptors | Metadata/State program after binding | No cached Mamba pointer tables from an earlier generation |
| Initialize / warm / capture | One root lifecycle; banked execution resources | No parallel native capture/profiling cleanup owner |
| Publish executable generation | Complete initialization/capture succeeds | No partially active runner fallback |
| Failed activation or close | Retire all enrolled readers, unbind then release | Native cleanup must not clear/reallocate the owned State |

The frozen runner's allocate/reshape, initialize_kv_cache, cleanup and execution
paths should become fail-fast forbidden calls AFTER handoff, not just counters
that happen to stay zero. A consumer binding shim may temporarily expose a
kv_cache tuple, but must only borrow new lanes. Retaining native KVCacheManager
as the final semantic owner would not complete the State reform.

Do not combine native startup allocation with a second root allocation and call
that handoff. Also do not promise current root.close can retire foreign native
graphs: captures must belong to the selected root lifecycle or have an explicit
enrolled retirement owner. Graph destruction / stream drain precede lane release.

## Corrected first acceptance: retain hot seats (Fletcher, 2026-09-25)

This supersedes the original mandatory A-seat0 → B-seat0 → C-seat1 story.
Request completion releases execution occupancy, NOT resident State. Seat0
retains A's continuation state. Unrelated B uses empty seat1 first. A compatible
continuation C hits seat0 and continues there. Overwriting a hot seat for an
unrelated prefix is an explicit eviction, not routine request cleanup.

Use real small-model weights, TP1, two resident seats, MTP2 and context4096.
Execution-bank identity remains separate from resident-seat identity.

1. A runs on seat0, crosses the selected boundary and finishes. Drain readers;
   preserve its resident numerical state, prefix identity and valid cursors.
2. Unrelated B uses empty seat1. Assert seat0 contents/identity survive unchanged.
3. C extends A's compatible resident prefix. Admit it on seat0, consuming that
   retained state without mandatory checkpoint copy or prefix recomputation.
   Compare its continuation against independent cold recomputation; check B is
   untouched. A new request lease must not invalidate compatible resident State.
4. With both seats occupied by retained state and no empty seat, unrelated D
   requires an explicit victim decision. Only a safely quiescent victim can be
   evicted. Invalidate its old identity before new writes, install D's identity,
   and prove the other seat remains intact. A later request for the victim's
   former prefix must miss unless another explicitly retained representation
   actually exists; do not silently invent a checkpoint store.
5. Include incompatible lookahead and a prefix shorter than the retained GDN
   state. Do not use a later recurrent state to satisfy an earlier boundary.
   Such a hit needs a separately retained compatible checkpoint or must miss.
6. Closing with active invocations must reject; after drain close succeeds.
   Fresh root activation rejects old handles/descriptors. Inject one failed
   initialization and verify no partially published generation remains.

Cross-seat restoration becomes a separate optional test ONLY if a future,
explicitly selected eviction/offload/branch policy retains a checkpoint that
requires it. It is not the primary warm-hit path or a prerequisite imposed by
this first acceptance story. No external pool or separate checkpoint allocation
is assumed merely to make request turnover work.

Acceptance branches: natural real-weight acceptance is primary. It may not visit
all zero/partial/full branches in a tiny fixture. Cover missing branches in a
separately labeled controlled-proposal protocol fixture (alter proposal input,
not verifier result); verify against actual target tokens. Do not claim forced
acceptance as real-model acceptance quality or performance.

CPU structural checks should reject overlapping lanes, stale generation handles,
wrong candidate/conv multiplicity and incomplete activation. Existing recurrence
oracles check intermediate candidates, not just output/final state. Byte-only
roundtrip or successful capture alone cannot pass. Same-seat warm continuation
is intentional; prove retained-state use and numerical equivalence rather than
rejecting it for not exercising a copy.

This small dense model cannot qualify MoE routing, TP collectives or35B TP2
performance. Those remain later integration gates after the single-device State
protocol is proven. No900s benchmark or CPU connector belongs in this first gate.

## Remaining implementation hinges, now bounded

- The exact checkpoint conv representation and the ownership of draft prefix
  validity when restoring without full hidden-history recomputation.
- The minimal Ascend model-loader/binding path that stops before native State
  allocation while preserving existing numerical operators and weight sharing.
- Kernel/head-ratio admission and required graph shapes for this small geometry.

Resolve these from the actual consumers or a CPU shape/transition fixture before
writing an allocator replacement. They are not grounds to redesign the already
available State lifecycle, and no implementation is claimed in this note.
