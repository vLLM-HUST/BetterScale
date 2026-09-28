# Incremental State backup and sparse device residency

Enter before replacing whole-checkpoint transfers with incremental FA backup,
partial device eviction and missing-page restore. This is a source-derived
construction assessment, not an implemented or qualified feature.

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
