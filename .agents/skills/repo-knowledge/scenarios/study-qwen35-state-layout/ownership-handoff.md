# Research: displace native State allocation and management

## Scope and source boundary

Fletcher authorized careful research on replacing native ownership, not a runtime
implementation or hardware campaign. Retain numerical consumers where possible;
do not extend the union-layout representation. This is a proposed cut, not an
accepted migration API or proof that its first stage can already execute.

Inspected BetterScale HEAD: `0e2055825359678c89aae225de8fd13f5d1bbd77`.
LiveInference reference remains `05ac15419c0e73650e687ceb9daffeb7874865f0`.
Native runtime and September24 Qwen35 FULL capsule roots are in GUIDE.md.
Read-only inspection; no tests or model runs were performed for this note.

## First distinguish two existing paths

**Qwen35 TP2/MTP2 FULL C32 capsule** still inherits native startup, allocation,
block management and runner execution. `parallel_full_worker.Worker` composes
`PartitionObserver` and BetterScale Worker. `service_adapter.py` replaces GDN
numerical/metadata leaves and installs `device_apc`; it does not replace the
State backend. Public package Qwen qualification is not identical to this
experimental capsule: do not infer MTP35 coverage from public defaults.

**Tracked `prototypes/owned-wave`** genuinely uses LiveInference execution.
Its serving example is ordinary Qwen3-30B attention, not Qwen35 GDN/MTP.
`serving/cache.py:PrefixLeases` explicitly requires one attention group.

Its `live_root.py:AdoptedStateBackend` returns pre-existing buffers. Both
`OwnedRoot` and `serving/root.py:ServingRoot` declare each entire byte arena as
one State with `ExactStateCapacity(1)`. `joint-wave/storage.py:backing_views`
collects whole underlying storage, not exact semantic State lanes. Thus:

- Real LiveModule activation, State snapshots, graphs and invocations exist.
- Native startup still allocates the arena; model views retain its storage.
- PrefixLeases creates its own native KVCacheManager, reusing native hashing,
  allocation, eviction and refcounts; it is not the original scheduler instance.
- Closing the adopted root relinquishes its lease, not all native references.
- Adopting a whole union arena would hide its internal overlap from semantic
  State validation. This must not become the Qwen35 migration design.

This existing path is useful execution-protocol experience, NOT evidence that
native State allocation and management have already been displaced.

## Existing authority chain in the Qwen35 capsule

```text
EngineCore: collect specs / profile available bytes
  -> get_kv_cache_configs: grouping, padded pages, shared_by, block capacity
  -> scheduler KV config / native KVCacheManager
  -> executor.initialize_from_config
       -> NPUWorker.initialize_from_config
         -> NPUModelRunner.initialize_kv_cache
           -> attention backends + input-batch/block-table geometry
           -> initialize_kv_cache_tensors
             -> _allocate_kv_cache_tensors
             -> _reshape_kv_cache_tensors
             -> bind_kv_cache (module fields + runner list)
           -> draft attention initialization / optional connector registration
       -> compile_or_warm_up_model

schedule: native block IDs, prefix lookup, allocation/release, Mamba align state
  -> input batch and block tables
  -> device_apc.prepare: seat identity, selection, state migration
  -> target / draft numerical writers
  -> device_apc.postprocess: accepted state + completion event
  -> later prefix publication / reuse under capsule's hash boundary protocol
```

Do not collapse these owners into “the allocator.” There are at least four
separate authorities to transfer:

| Authority | Present owner | Required exit condition |
|---|---|---|
| Physical layout, backing and capacity | Core cache planner + runner allocation/reshape | State declarations/backend produce the actual storage and admitted capacity; no native second allocation |
| Binding and generation lifetime | runner lists, module kv_cache, graph catalogs, lazy pointer tables | Consumers borrow exact bound lanes; old generation cannot remain reachable by executable graphs or pointer descriptors |
| Logical row ownership and retention | KVCacheManager/coordinator/BlockPool/Mamba manager | New explicit domains own admission, IDs, refcounts, checkpoints and eviction; no hidden common-pool exclusion |
| Content mutation and commit | numerical kernels + MTP/device_apc + scheduler hash publication | Every writer and publication fence has an explicit owner; allocation pinning is not a substitute |

## Earliest useful physical seam — necessary, not sufficient

`initialize_kv_cache_tensors` is a recognizable boundary: it returns typed
per-layer tensors and publishes them to numerical modules. In the capsule,
GDN `forward_core` consumes `*self.kv_cache`; that consumer need not care which
allocator supplied those tensors.

However replacing this method alone would NOT remove native management:

1. EngineCore has already selected capacity from the old padded/group layout.
2. Input-batch and backend metadata already encode native group/block geometry.
3. KVCacheManager still owns a shared block-ID namespace and Mamba retention.
4. Draft attention and APC still derive state addresses from those bindings.

A temporary independent-storage provider under the old ID manager could be a
bounded discriminator, but would retain native management and potentially
waste capacity. It is not the proposed final architecture, is not implemented,
and must not be reported as completing the requested displacement.

The research cut therefore spans **capacity agreement → State realization →
consumer binding**, plus an explicit contract for logical domains. It cannot
start after old capacity decisions and silently keep their byte arithmetic.

## Hidden borrowers and revocation hazards already located

- `vllm/v1/worker/mamba_utils.py:initialize_from_forward_context` creates
  state base-address descriptors once behind `is_initialized`. `device_apc`
  initializes it lazily. Rebinding module kv_cache alone leaves stale addresses.
- `device_apc` also retains seat maps, running columns, selection, pinned input
  carriers and `_mtp_apc_done`. These control resources must be classified and
  reinitialized/retired with their proper lifetime, not mistaken for checkpoints.
- Native `_cleanup_profiling_kv_cache` clears runner cache/config/groups and
  rewrites module kv_cache fields. It must not remain an authorized mutator of
  a published LiveInference-owned generation.
- Native/draft graph captures retain addresses beyond Python assignment.
  LiveModule.close only knows its own registered graphs/invocations. Wrapping
  borrowed State does not automatically retire foreign captures or drain events.
- Worker initialization includes connector and optional sleep-pool ownership;
  these remain explicit unsupported/held boundaries until audited, not defaults
  to inherit accidentally.
- BetterScale `patches/auto_kv` has trial allocation/cleanup, but its documented
  caller is DSV4. Do not import that lifecycle as evidence of the Qwen35 path.

## Reuse LiveInference's actual contract, not its names

Current `core/live_module.py` validates a complete State realization before
binding domains/lanes. Close rejects active invocations, releases graph users,
runs generation-release hooks, unbinds metadata/State/domains, then releases
backend realization. This is an existing lifetime owner to reuse.

But it does not automatically know native KVCacheManager, borrowed model
fields or Mamba pointer descriptors. A migration must enroll/retire those
borrowers or stop creating them. Do not keep two activation/cleanup owners.

Candidate semantic census for the Qwen35 research (not a frozen API):

- Per-layer FA K and V: token-page domain, including draft attention.
- Per-layer GDN convolution and recurrent state: live seat/candidate state.
- Retained GDN prefix checkpoints: separate from seat lifetime; compatible
  FA prefix coverage and MTP token identity are necessary for a reusable hit.
- Accepted-state selection/progress: model execution state, not hidden allocator
  metadata; distinguish persistent semantic state from banked ingress/workspace.

The exact seat/candidate/checkpoint copy policy remains unresolved. A seat-only
GDN redesign that discards checkpoints is not behavior-preserving APC migration.

## Proposed bounded next inquiry

Before implementation, trace one request through fresh prefill, MTP accept/
reject, checkpoint publication, seat departure, prefix-hit resume on another
seat, then generation close. Record for each transition the lane/domain IDs,
last writer, readers, copy direction, completion fence and release authority.
This will settle which parts of Mamba manager are semantic requirements versus
artifacts of union storage, and whether existing LiveInference Qwen State
programs can supply them unchanged.

Only then choose the smallest executable vertical. Its eventual acceptance
must demonstrate independent semantic lane allocation, correct block
multiplicity/prefixes, agreement on capacity, no calls to retired allocation/
cleanup entries, unchanged numerical continuation across checkpoint restore,
and safe failed activation / live-invocation close / generation replacement.
These are research-derived gates, not a newly authorized test campaign.
