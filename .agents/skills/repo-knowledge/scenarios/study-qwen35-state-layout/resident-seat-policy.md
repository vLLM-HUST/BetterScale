# Resident seats, not request-scoped disposable slots

Fletcher's clarification, 2026-09-25. This is the intended new policy, not a
claim that the native block manager or current prototypes already implement it.
It supersedes earlier migration prose that requires every request to release
its seat and copy a checkpoint into another seat for continuation.

- Request completion ends active execution/its lease. It does not discard the
  seat's resident numerical state, prefix identity or valid-state cursor.
- An unrelated request uses an empty seat before evicting a retained hot seat.
- A compatible continuation reuses the matching resident seat in place.
- Replacing a hot resident with an unrelated prefix is eviction. It requires a
  selected victim, reader/writer quiescence, old-identity invalidation and a new
  resident incarnation. It is not an incidental consequence of request finish.
- Graph/transport banks are not resident seats. Root generation, resident
  incarnation and individual request leases must not be conflated.

First story: A runs on seat0 and leaves it hot; B uses empty seat1; C matches A
and continues seat0. Only when no empty seat exists do we exercise unrelated
replacement. Preserve numerical State across the idle gap and prove the hit
avoids prefix recomputation. Cross-seat restore is not the default path.

Compatibility still requires exact represented history/cursors, not merely a
common string prefix. GDN state after token n cannot recover state after m<n.
MTP may retain a pending anchor/last emitted token not yet processed into all
State; resident identity must describe those distinctions. Do not fix this by
silently assigning one cursor to all target/draft/continuation contents.
An earlier-prefix hit needs an actually retained compatible checkpoint or is a
miss. Concurrent branches from one prefix need an explicit sharing/serialization/
copy decision; this policy does not authorize two writers to one mutable seat.

Module-local StateTensor declarations and root/backend physical lifetime still
apply. Seat storage is long-lived; request turnover changes execution authority
without necessarily changing resident contents. A separate checkpoint pool or
CPU offload pool is optional future capacity/branch support, not required merely
because A finished. Do not manufacture migrations to fit a three-domain diagram.

Research-only note change; no runtime implementation or device validation.
