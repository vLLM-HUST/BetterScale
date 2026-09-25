# C16 with about20 resident seats: capacity estimate

2026-09-25, research arithmetic, not allocated memory or a performance result.
Fletcher proposed retaining about20 resident seats for16 concurrent requests.
Separate execution width E=16, resident capacity R=20, and a bounded token-page
pool. Graph/activation envelopes should follow admitted execution shapes, not
increase automatically to20 merely because resident IDs range over20 entries.
Resident-sized metadata grows where genuinely needed; transient in-flight
readers can temporarily reduce the nominal four spare seats.

## Assumptions and per-device arithmetic

35B-A3B TP2: target30GDN+10FA, draft1FA; per-rank GDN value16 and128x128 FP32
state, BF16 conv4096channels x5history rows, FA kv1/head256 BF16. Small0.8B TP1:
18GDN+6FA+1draftFA, GDN value16, conv6144x5, FA kv2/head256. Source geometries
are recorded in GUIDE.md and small-model-metadata.json. MTP2 keeps3 recurrent
candidate matrices and ONE speculative convolution history per live seat.

For a simple uniform allocation, retain that complete representation for all
resident seats, hot-idle as well as executing. This avoids a new compaction/
restore protocol. No mandatory copy occurs on request completion.

| Payload |35B TP2, each rank |0.8B TP1 |
|---|---:|---:|
|GDN per resident seat |91.171875MiB |55.0546875MiB |
|16-seat GDN |1458.75MiB |880.875MiB |
|20-seat GDN |1823.4375MiB |1101.09375MiB |
|Extra4-seat GDN |364.6875MiB |220.21875MiB |
|FA+draft bytes per retained token |11264 (11KiB) |14336 (14KiB) |
|Extra4 independent4K histories |176MiB |224MiB |
|Extra4 independent32K histories |1408MiB |1792MiB |
|Extra4 independent256K histories |11264MiB |14336MiB |

These exclude weights, workspaces, allocator alignment, inactive-lane scratch,
continuation/control metadata and speculative token-page tails. They assume
all listed FA lanes retain those token ranges, BF16 KV, and no cross-request
prefix sharing. They are not a complete physical capacity receipt.

At35B TP2, adding four uniform GDN seats plus four independent32K histories
costs about1.73GiB/rank in payload; at256K it is about11.36GiB/rank. The two-rank
aggregate is twice the per-rank bytes, but admission is per-device, not pooled
across chips.20 versus16 is25% more *seat-indexed GDN*, not25% more model memory.

## Token pages must not become max-context seat arrays

A resident seat pins the actual token pages it needs. It must not automatically
reserve max_context pages for every seat. If the total State budget is fixed,
adding four GDN seats consumes about365MiB/rank from the35B token-page budget;
retaining hot-idle histories additionally competes for those remaining pages.
The same nominal20 seats can therefore reach token-page pressure well before
all seats are occupied. Seat admission and token-page admission are separate
constraints. Eviction removes the selected resident's ownership, but actual
pages are reclaimed only after the final shared/in-flight reference retires.

Empty-seat-first is a placement policy, not permission to exceed the page
budget: an empty seat can still require eviction of safely idle cached pages/
residents, or admission deferral if all resources are pinned. Do not displace
active requests merely to maintain20 occupied seats.

Four spare residents are a locality buffer, not a guarantee of20 useful cached
sessions or immunity to eviction. Benefit depends on return order, tool delay,
history length and working set. Active slots, idle retained residents and
retiring-but-pinned residents must be distinguished in future traces.

## First implementation direction, not a new optimization mandate

Keep E and R explicit. Use fixed R for GDN/continuation StateDomains and a
separate admitted token-page capacity. Preserve full candidate representation
in idle seats initially; a theoretical compact idle snapshot would reduce
extra4-seat35B GDN payload from364.69 to124.69MiB, but only after defining accepted
selection, convolution window normalization and safe scratch ownership. That
saving is not worth silently introducing copies or another state representation
into the first migration. It is not implemented or selected here.

No full target-hidden-history sidecache is included. Importing the old Qwen
LiveInference prototype's history-rebuild policy would add context-proportional
memory and invalidate these estimates. No claim that current kernels or root
already accept the E/R separation without metadata/ownership changes.

For the first small single-device test use a scaled-down E=2,R=3 or4 envelope
and force churn, then validate E16/R20 once State semantics pass. The earlier
two-seat story remains a minimal ownership example; E/R separation is the
additional admission test. No NPU launch is authorized by this note.
