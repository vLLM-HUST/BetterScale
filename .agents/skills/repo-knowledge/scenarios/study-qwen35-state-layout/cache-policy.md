# Automatic resident/host cache policy

Enter when changing automatic backup, device eviction or host admission. The
copy execution and TP quorum remain in [native cache actions](native-cache-maintenance.md);
[continuous-DMA evidence](native-cache-timeline.md) establishes hardware overlap,
not a serving speedup. This policy is separate from numerical execution.

## Accepted rules (Fletcher, 2026-09-27)

- Trigger backup when either resident-seat or shared-page usage reaches70%.
  Their capacity domains remain independent; a seat never reserves max context.
- Current and previous scheduling rounds must both omit the candidate. This is
  a cheap candidate window, **not a writer-completion proof**: also require the
  actual native last_sched_seq/processed_step_seq fence to have retired.
- Backup retains device residency. Choose never/least-recently backed candidates
  round-robin, one automatic background action at a time. In-flight I/O pins
  the exact seat generation and independent references to its FA pages.
- Device capacity reclaim uses oldest eligible resident LRU, irrespective of
  whether it has a host copy. Never force a store before eviction. The inherited
  running-request preemption fallback remains native; do not describe it as a
  new LRU for every running request.
- Host entries have their own LRU, touched on insertion/load/hit. Drop only
  unpinned entries, retaining their byte charge through all-rank acknowledgement.
- Returning requests prefer hot device state. Otherwise, an exact token-prefix
  AND salt host match can restore when seat/pages fit; publish after TP quorum.
  Missing/evicted host copies take native cold recompute. Old host checkpoints
  are immutable earlier prefixes, never proof that newer device contents match.

## Implementation and bounded behavior

Opt in alongside `using_live_runtime` with positive per-rank
`state_cache_host_bytes` and `state_cache_policy: true`;
`state_cache_watermark` defaults0.7. Omitting the policy flag preserves the
previous explicit-action interface. `CachePolicy` in `cache_policy.py` owns
candidate age, backup rotation and automatic restores. `CacheActions.backup`
adds retained-device semantics; legacy explicit `store` still releases device
residency after quorum. No copy worker math or graph initialization changes.

A still-owned, drained request can be backed up using its observed exact
Frontier. Its native page refs are independently pinned through completion,
including cancellation/abort. During I/O, remove that owner from runnable work
and subtract its reserved slot from admission; restore its queue position after
native scheduling. Matching restore/backup waiters are skipped, allowing other
ready requests to run rather than making a pending load head-of-line blocking.
There is no device-wide drain or synchronous copy wait in scheduler code.

Maintenance-only turns use native has_requests/WAKEUP, including when all user
requests have finished. Remember which seat generation/frontier was already
backed: host LRU eviction must not trigger endless idle re-copy/evict cycles.
New device progress makes a new candidate. Oversize objects are not queued.
Automatic receipts are bounded to the latest256 completed operations; pending
futures remain owned until completion. Debug utility lookups are not a durable
operation history. Snapshot includes host identity/salt/cursor/byte charge.

## Validation entry

`tests/test_qwen35_cache_policy.py` exercises watermark domains, two-round plus
writer fences, retained device state, unbacked eviction, host LRU/quorum,
restore/hot preference, capacity reservation for I/O-held owners, bounded
receipt history, and real native page-refcount preservation through abort.
Use the CPU-capable pinned donor runtime with TORCH_DEVICE_BACKEND_AUTOLOAD=0.
These are protocol tests, not NPU numerical validation.

`CACHE_AUTO_POLICY=1` on the existing admitted `native_probe.py` selects
`native_policy_probe.py`: native35B TP2/MTP2/FULL/async, E16/R20,6GiB State,
4GiB host/rank. Workload alone fills both tiers; no manual store/load/drop.
It requires retained hot seats after backup, automatic restore after device
LRU eviction, exact hot/restored/cold chat output, then cold recompute after
host LRU eviction. Immutable capsules live under
`runs/qwen35-state-lanes/20260927-cache-policy/`. Never infer a PASS merely from
this recipe; read the run receipt and qualification boundary below.

## Native qualification receipt (2026-09-27)

`attempt2` passes on hw3 physical4/5, real native async `step_with_batch_queue`,
TP2/MTP2/FULL. No manual store/load/drop calls:

- Fill20 completes20 automatic backups while all20 device seats remain hot.
- Workload churn evicts A from device, while its exact host checkpoint survives.
  Automatic return loads it:979 cached tokens, versus cold0.
- Unmoved hot, restored, independent cold, and re-entry after host eviction
  all produce the same30 IDs and exact expected access code. Host-evicted
  re-entry reports0 cached tokens, proving the recompute fallback.
- Before the last re-entry:85 store /1 load /49 drop completions, all TP2 quorum;
  host36 entries,4,272,244,560 bytes/rank below4GiB; pending0.
- Process exit0. Release sample shows NPU4/5 at3420/3421MiB, no worker owners.
  Native teardown did force-kill one remaining process after its grace period
  and resource_tracker reported one shared-memory object; retain these warnings,
  do not claim all workers exited voluntarily. No foreign work was displaced.

`attempt1` was stopped during startup because its fresh capsule omitted the
previously compiled native libraries. `attempt2` explicitly copies those same
qualified libraries from candidate8; no rebuild/operator change or live-runtime
patching. Both attempts are retained. The automatic driver uses normal chat
closure, not the known forced-post-EOS warm/cold counterexample.

CPU gates:38 tests passed across policy/actions/worker/resident/native-page
lifetime; two additional policy tests then passed with the actions suite
(22 tests), covering bounded completed-history and independent watermarks.
The final source differs from the admitted snapshot only by bounding the
completed receipt history to256 (deque and snapshot-list conversion), covered
by those final CPU tests; numerical/scheduling/transfer source is unchanged.

Limits: this is lifecycle/correctness qualification, not throughput or C16/C32
performance evidence. The workload numerically exercises completed-request
idle backups; still-owned but drained backup, abort during copy and runnable
slot reservation have CPU protocol/refcount coverage, not a forced TP2 device
fault/concurrency qualification. No claim of arbitrary token-history parity,
persistent host cache across engine lifetimes, or transport-failure recovery.

For the completed C16/C32 × D1/D2 matched pressure matrix, host-LRU turnover
and the retained C32D2 tail-latency regression, read [cache-pressure.md](cache-pressure.md).

For the accepted incremental-backup / partial-page-eviction direction and its
source-derived construction boundaries (not yet implemented), read
[incremental-cache.md](incremental-cache.md).
