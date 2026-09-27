# Qwen3.5 scheduler cache-maintenance prototype

Explicit experimental actions, not an automatic cache policy or a public serving
option. The original research route below is separate from the new optional
native `models/qwen35` Worker/AsyncScheduler integration (see end).

The small-model research `Scheduler(root, maintenance=cache)` owns completion
publication. `cache.store(seat, HostStateKey(session, generation))` pins an idle
resident and its shared FA pages; the copy worker waits for the current producer
stream, enqueues D2H on its copy stream, and wakes a host future on completion.
Only a successful scheduler reap publishes the host checkpoint and evicts the
source. `cache.load(key, seat)` reserves the destination and shared pages first;
its numerical state is invisible to prefix matching until the H2D receipt.

Payload uses the existing `TorchHostStateBackend` and declared State domains:
all GDN candidates + extended convolution, all target/draft FA pages, and
continuation metadata. `resident_epoch` is placement identity, not portable
numerical state: restore keeps the destination's new epoch. Stable addresses
remain unchanged; physical seat and page IDs may change. Host capacity includes
in-flight allocations, not just completed snapshots.

I/O pins do not consume execution width. Seat admission and page-pressure
reclaim exclude those pins. A request awaiting a matching load yields admission
to unrelated requests. Cancellation only records intent: payload and device
ownership survive until completion. Stale/failed completion quarantines the
operation rather than pretending memory is reusable. Partial enqueue failure
must drain that copy stream before rollback; failed drain retains the buffers.

The driver must pump maintenance even without model work:

```python
cache.wait(timeout=30)  # host completion notification, no device-wide barrier
scheduler.tick()       # one scheduler owner consumes the receipt
```

`probe.py` supplies that driver explicitly. Call `scheduler.close()` before
closing/reactivating the root. All submissions, reaps and cancellation use one
scheduler thread. No request thread may call these methods concurrently.

## Verification

CPU-only (torch available, NPU autoload disabled):

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src python -m pytest -q \
  prototypes/qwen35-cache-maintenance/test_maintenance.py \
  tests/test_live_qwen_residents.py tests/test_live_qwen_scheduler.py
```

`probe.py` requires an admitted single NPU and the pinned Qwen3.5-0.8B real
weights at `$CAPSULE/model`. It checks source-seat reuse, different destination
pages, byte-exact restoration, MTP continuation versus cold MTP and independent
target-only execution, maintenance-only progress, and cancellation after enqueue.
Use the repository's hw3 selected-device lease/foreign-owner guard. This file
is not permission to initialize a device outside that protocol.

## Deliberate limits

- TP1 only: no multi-rank completion quorum, partial-rank restore, remote store,
  persistence, prefix tree, host LRU, automatic hit lookup or eviction policy.
- Only idle/completed residents can be stored, not active preemption. The caller
  chooses the complete continuation boundary and final-writer stream.
- This research scheduler completes numerical waves synchronously; the native
  asynchronous scheduler and its delayed completion fences are not integrated.
- The transfer engine adds no device-wide synchronize. Existing research-root
  `clear_seat`/generation do synchronize: this is **not** a proof of production
  overlap, tail latency or throughput. A transport failure may require root
  teardown; this prototype does not recover an unhealthy device in place.

## Native async / TP2 expansion

`native_probe.py` exercises the actual AsyncLLM/EngineCore batch queue, FULL
graph and MTP2 on35B TP2. The optional production-source closure lives in
`models/qwen35/cache_{actions,worker,engine}.py`, enabled only with
`additional_config.state_cache_host_bytes` (per rank). Scheduler utility
`state_cache` supplies explicit store/load/cancel/drop and deferred wait.
Rank receipts wake the native scheduler even without computation.

See [native integration evidence and boundaries](../../.agents/skills/repo-knowledge/scenarios/study-qwen35-state-layout/native-cache-maintenance.md).
The original TP1 research limits above are historical, not native qualification.
Native FIFO/preemption policy remains unchanged: pending loads may head-of-line
block waiting requests. No automatic eviction policy or performance claim.
Device/receipt failure fails closed and may require whole-engine teardown.

The native probe defaults to a valid multi-turn chat boundary and compares an
unmoved hot twin, relocated hot continuation, and independent cold execution.
`CACHE_BYTE_AUDIT=1` additionally reads all selected device views back to CPU
before each rank receipt; this is a diagnostic barrier, **not** a performance
configuration. `CACHE_LEGACY_FIXTURE=1` preserves the forced-post-EOS fixture:
it exposed a warm/cold discrepancy also present without offload, not a transport
failure. Do not remove that counterexample or generalize the normal-chat PASS
into unrestricted post-EOS numerical equivalence.

For a bounded timeline, copy `native_profile.py` beside the driver and set
`CACHE_PROFILE=1 CACHE_BYTE_AUDIT=0` under the normal admitted launcher. Capture
excludes initialization and uses B256 rather than the unprofiled B1024. Native
rank-bound profile RPCs retain raw CANN files beneath `$CAPSULE/profiles` plus
host control/receipt JSONL. See [the observed timeline and interpretation](../../.agents/skills/repo-knowledge/scenarios/study-qwen35-state-layout/native-cache-timeline.md):
first-use D2H did not overlap model kernels despite asynchronous completion.
