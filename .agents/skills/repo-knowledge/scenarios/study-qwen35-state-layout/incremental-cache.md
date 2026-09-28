# Incremental State backup and sparse device residency

Enter before replacing whole-checkpoint transfers with incremental FA backup,
partial device eviction and missing-page restore. The assessment below is
historical; the implementation and qualification receipt at the end supersedes
its unimplemented status without widening the measured scope.

## Accepted direction (Fletcher, 2026-09-28)

No predictive prefetch. On demand, retain still-valid device FA pages and restore
only holes. Reusing a cold request's physical pages invalidates only overwritten
pages, not the rest of its prefix. Backups should reuse unchanged host FA pages.
GDN/continuation remains an exact-boundary snapshot, separate from FA residency.
Execution and DMA pins still exclude a page from reuse. An unbacked page may be
evicted; absence on both tiers must not become an invented cache hit.

## Source boundaries observed

Inspected offloading branch `b331c0e` (measured behavior `5d1dbfc`) and fetched
main `852c106`. Main has prefill-round-robin grants but does not yet contain the
phase-one cache actions/policy/worker. They are divergent work, not deleted
features: integrate both explicitly; do not replace main's scheduler wholesale.

- `resident_leases.py`: `discard_victim`, `claim`, `evict_hot` release a seat's
  entire block list and discard its prefix identity. Separate GDN-seat reuse
  from page residency. A seat with missing pages cannot be advertised as the
  current ready/warm offer until repaired.
- `seat_scheduler.py`: `_allocate` reclaims entire idle residents on native
  allocation failure. `_computed` accepts only a complete exact resident; native
  FA hash publication is deliberately disabled. Preserve that GDN boundary gate;
  do not simply re-enable FA-only prefix hits. `_publish_completed_residents`
  retains a partial final block, not only immutable full pages.
- `cache_actions.py`: `Checkpoint` owns only a block count and aggregate byte
  charge; every backup creates a new full checkpoint. `load` discards the victim
  and allocates the checkpoint's full count. Replace those assumptions with
  references to logical FA page versions, surviving physical placements and a
  transfer plan containing only misses. Charge unique host payloads, not each
  manifest's repeated references; reserve in-flight allocations until quorum.
- `cache_worker.py`: command block IDs lower from native scheduler blocks into
  State kernel pages; resident and page domains are already separate. Existing
  direction streams, producer/APC events and TP completion receipts are reusable.
- `live/runtime/host_state.py`: State-domain selection already supports sparse
  destination IDs, and copies coalesce consecutive physical runs. But each
  snapshot allocates packed lane payloads, and `_validate_payloads` requires
  exactly the original lane closure and block count. Arbitrary partial reads and
  sharing between checkpoints need explicit host-page addressing/ownership; a
  shorter destination list alone is not supported. Avoid one Future/event or
  pinned allocation per tiny page: submit/coalesce a page batch under one action.

## Smallest proposed construction

Keep a checkpoint as exact tokens/salt + GDN snapshot + ordered logical FA page
references. Keep device presence separately from the seat, with physical-slot
reuse generations and actual compute/I/O pins. Keep host FA payloads reference
counted so multiple retained checkpoints can share sealed prefix pages. Physical
block IDs alone are not identities. The first cut need not create a universal
radix cache or a new numerical allocator; use the existing exact-prefix match
and pooled allocator where their contracts suffice.

Page registration must cover **every** reuse route (native extension, request
free/preemption, explicit/automatic restore, cache reset), not only the idle-seat
victim loop. A free/reclaimable physical page may retain valid old contents until
reallocated; retaining that knowledge must not count all such pages as pinned
and eliminate usable capacity. Reacquire/pin surviving pages before admitting a
restore, then allocate only missing pages so the allocation cannot evict its own
prospective hits. The exact native-pool interception remains an implementation
check, not a qualified hook in this assessment.

Sealed prefix pages can be reused; a partial tail and speculative target/draft
writes require explicit valid-boundary/version treatment. A page becoming full
is not by itself proof that every lane's committed contents are immutable.
First implement conservative new tail versions (copy-on-write where an old
version is still referenced), not token-subrange DMA or GDN deltas. Do not share
a writable tail between independent requests. GDN plus all required FA page
versions must agree on the chosen boundary before publishing a warm request.
If the newest checkpoint has an unavailable page, an older **complete** matching
checkpoint or cold recompute is safe; scattered FA hits alone cannot authorize
skipping GDN computation.

Host manifest LRU can remain simple initially: dropping a checkpoint releases
its GDN and FA references; shared payloads free only at final reference release
and after I/O completion. No forced backup before device eviction. Cancellation
must unwind only acquired references, after all DMA ranks retire. Page placement
generations supplement, rather than replace, the existing seat epoch/quorum.

## Focused gates before a performance claim

1. A owns N pages; B overwrites k cold unpinned pages; A retains N-k valid pages.
   Demand restore copies exactly k pages plus needed GDN, not all N.
2. A advances after a backup: old sealed prefix host payloads are shared; only
   appended/dirty-tail payloads and a new GDN boundary are stored/charged.
3. Tail-boundary crossings, changed salt/divergent tokens, target/draft accepted
   frontier and old-checkpoint reuse cannot silently match wrong page versions.
4. Slot reuse, abort, delayed TP rank, host LRU and reset preserve pins/accounting;
   a restore cannot evict its own surviving pages during allocation.
5. Combine I/O exclusion with main's round-robin preparation/grants; only runnable
   requests consume that round's grants. Preserve decode/native async behavior.
6. CPU page-lifetime/byte-copy gates first, then bounded native TP2 mixed requests
   comparing unchanged-hot / partial-restore / cold outputs and actual DMA bytes.
   Reuse the C16/C32 rotation protocol only after that passes. No NPU run yet.

Assessment: medium, bounded cache-control work rather than a runtime rewrite.
The principal risk is version/reference correctness at the mutable tail and
async reuse boundary, not copying fewer bytes. Kernel-page granularity versus
native allocator-block granularity must stay explicit; starting at native blocks
avoids silently expanding this into an allocator redesign.


## Implementation and qualification (2026-09-28)

Implemented on `codex/qwen35-incremental-cache`, integrating main `852c106`
without replacing its prefill rotation. Enable `state_cache_incremental: true`
alongside the existing host-cache options. Non-incremental behavior remains
available. Runtime sources live in `src`, not the probe closure.

`cache_pages.py` wraps the **instance-local** native pool allocation/free seams.
Reallocation invalidates weak identities; write admission invalidates the mutable
suffix. Cold valid pages append in LRU order behind genuinely unused/invalid
free capacity, rather than taking native unhashed-free prepend priority. No
native FA-only hash hit is enabled. On restore, free surviving placements are
pinned before holes are allocated. This first cut deliberately does not alias
active device placements, even sealed ones.

`live/runtime/page_state.py` owns refcounted host objects and manifests. Each
native-block object owns its allocation (no partially retained slab charged as
one small page); one batch uses one DMA event, not one event per page. Each GDN
snapshot stays private. Incremental cache actions are serialized, while other
inference remains async. Failed enqueue drains its stream before unwinding;
unknown DMA lifetime retains handles/storage and capacity until engine teardown.
No device-wide synchronization is added to normal cache operations.

**CPU:** 63 distinct tests passed across incremental ownership/bytes/failure
lifetime (13), existing cache/actions/worker/resident gates (40), and main prefill
rotation compatibility/planning (10). These include the real native pool's
100-page / 20-overwrite / 20-restore witness, mutable suffix invalidation through
`LiveStateScheduler._allocate`, host unique-byte accounting, delayed TP receipt,
abort, LRU order and uncertain-DMA storage retention.

**Native explicit byte witness**, `candidate2`, source `39f18a3`, hw3 1/6:
35B BF16 TP2/MTP2/FULL/native async, balanced attention, 6GiB State/rank and
512MiB host/rank. A's4283-token exact checkpoint had three native FA blocks:
`[6,5,4]`; after overwrite it restored as `[6,5,12]`. Only one FA block was
transferred. Per rank, full backup164,810,804 bytes versus partial restore
118,673,460 bytes. Advancing A then backing up again added/transferred only
118,673,460 bytes (new GDN plus tail); the two sealed FA blocks were reused.
Both ranks checked all90 selected State views bytewise after store/load/store,
including the pages *not* copied during restore. All matched. Unmoved hot,
restored hot and independent cold returned the same30 IDs and expected code.
This audit deliberately synchronizes and is not an overlap/performance result.

**Native automatic policy**, `candidate3`, source `ab099a6`, hw3 1/2:
same model/execution, 4GiB host/rank, audit disabled, no manual cache actions.
20 automatic backups initially kept20 seats hot. A was evicted from the seat
arena but its three FA blocks survived in the weak LRU. On return the automatic
load transferred only95,604,788 bytes of GDN/continuation per rank: **zero FA
blocks**. Hot/restored hits4283 tokens; cold and post-host-eviction re-entry hit0.
All four outputs were identical. Final observed operations85 stores /1 load /
49 drops; host4,272,244,560 bytes/rank below4GiB, pending0.

The explicit partial-overwrite witness predates the weak-free-page LRU change;
the latest LRU has the real-pool CPU partial-overwrite gate and the automatic
NPU all-pages-survived gate. The current explicit probe also permits zero holes:
C must overwrite the GDN seat, but unused FA capacity should now protect A's
pages. Do not force a needless overwrite merely to recreate an older witness.
`e9b439e` subsequently tightens uncertain-DMA retention/draining, covered by CPU
fault tests; there is no fault-injected NPU qualification.

Both passing supervisors exited0 and selected cards returned to idle baseline.
Both native shutdowns force-killed one remaining owned process and warned about
one shared-memory object; preserve those warnings. `candidate1` failed before
model execution because the source capsule omitted main's newly required
balanced-attention library. Later capsules include the exact `native.json`
qualified binary; no admission or artifact guard was relaxed.

Compact tracked receipt: `docs/evidence/qwen35-incremental-cache.json`.
Full local/remote evidence: `runs/qwen35-state-lanes/20260928-incremental-cache/`.
CPU logs: sibling `20260928-incremental-cache-cpu` remotely, copied into the
local evidence folder. This establishes functionality and actual byte savings,
not C16/C32 throughput, tail latency, or a new timeline/overlap measurement.
