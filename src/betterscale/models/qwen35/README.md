# Qwen35 default serving: resident State + balanced attention

This is the qualified integrated serving route, not the earlier standalone live
execution loop. Model loading, forward arithmetic, FULL capture, MTP2 and async
EngineCore scheduling remain BetterScale's baseline. Owned `StateTensor`
declarations, resident leases and shared regular-attention pages replace State
storage/ownership. No external `livemodule`, experiment directory, observer Worker
or runtime-generated source import is required.

## Prepare once, outside any running service

Use the pinned Linux/aarch64 CANN9.0.1 / vLLM0.25.1 / Ascend0.25.1rc1 environment
and qualified native payload from this source build. Published PyPI0.5.1 does not
contain this route. Do not upgrade donor dependencies as part of installation.

```sh
python -m betterscale.models.qwen35.runtime \
  --source /path/to/pinned-vllm-ascend --output /path/to/new-qwen35-runtime
```

The source directory must contain the built`vllm_ascend/` package, including its
native libraries and custom-op vendor. A source-only checkout is rejected. The helper accepts either pristine
commit9bf964cb4b87c8cd0d6852c41a55b3c29711fa95 or the exact qualified adaptation.
It copies the package, applies the bundled four-file patch if needed, then verifies
all pinned Ascend source files. Unknown or mixed inputs fail closed. It never
replaces an existing output or edits the shared installation. The system`patch`
executable is needed for pristine input. Core commit752a3a504485790a2e8491cacbb35c137339ad34
and runtime versions are independently checked by the Worker.

After obtaining selected-device leases and fresh idle-card admission:

```sh
python -m betterscale serve-qwen /models/Qwen3.5-35B-A3B \
  --devices 0,1 --port 8000 \
  --qwen35-runtime-dir /path/to/new-qwen35-runtime \
  --cache-dir /path/to/task-cache --served-model-name qwen35-moe
```

The default `--runtime auto` recognizes the Qwen35 MoE model configuration and
selects resident State plus balanced target decode/verification attention.
Explicit `--runtime live` remains compatible; Qwen27's default route is unchanged.
No feature environment variables or external attention-library path are needed:
the complete distribution includes the qualified `libbs_fia_cp.so` and verifies
its digest before launch and again in the Worker. Missing or mismatched payload
fails closed, never silently falls back. For a diagnostic native-attention control
only, set `BETTERSCALE_CONTEXT_PARALLEL=0`; resident State stays enabled.
Single-request capacity3, prefill/mixed and draft retain native attention.

The launcher configures the real`betterscale.qwen35_worker.Worker`, native async
scheduler subclass, FULL4096/MTP2 and required environment before device startup.
It does not claim or acquire accelerator ownership itself. The default total State
budget is26,038,239,232 bytes/rank, E16/R20 and262144 context. Change the total via
`--state-budget-bytes`, not resident-count × context pages. The resident declaration
is1,912,095,920 bytes/rank; remaining bytes fund shared FA pages. The default admits
16,720 physical128-token pages. Explicit page counts remain rejected. `--execution-seats` controls native
`max_num_seqs`; `--resident-seats` independently controls retained State rows.
The current configurable envelope is1..36 execution rows, with at least as many
resident seats. For C16 retain E16/R20; the C32 system trial uses E36/R36, not
E16/R20 and not40 resident seats. Changing rotation depth does not automatically
multiply seats. Direct vLLM entry uses `--max-num-seqs` and
`additional_config.state_resident_seats` (default execution+4). The same values
reach allocation, host-byte accounting, metadata, MTP addressing and capture keys.
E36/R36 native TP2/FULL/async passes144 requests, including36 distinct-key hot
continuations and their exact-token independent cold oracles; see
`docs/evidence/qwen35-configurable-seats.json`. It retains2,004,992 shared FA tokens
at the default total budget. This correctness gate is not a new performance claim. No CPU KV offload or connector is enabled.

One convolution window and three recurrent candidates serve each resident. A known
output-length budget prevents queued later frames from overwriting terminal State;
next-turn affinity waits for its native writer fence. Arbitrary earlier checkpoint
or EOS/stop-string rollback is not supported. Preemption invalidates the entire
seat; no offload is implied.

The September26 sourcea8abd05 passed long/hot/preemption gates and one C16/900s SWE
observation:414.8011 output tokens/s/chip versus a fresh same-pair369.1728 control.
See repository knowledge's`study-qwen35-state-layout/baseline-execution-parity.md`
for original evidence and scope. Packaging/entry qualification is separate from
that immutable timing result; do not relabel it as a newly measured release.

Experimental automatic State caching: set `state_cache_policy: true` and a
positive per-rank `state_cache_host_bytes` in `additional_config`, alongside
`using_live_runtime`. `state_cache_watermark` defaults to0.7 (seat OR page
usage). Backup is best effort after two unscheduled rounds AND writer retirement;
it keeps device state hot. Device reclaim does not require a host backup. Host
capacity uses LRU with asynchronous all-rank drop acknowledgement. Exact host
hits restore asynchronously; absent copies recompute. In-flight I/O remains
pinned and other ready requests may run. This is opt-in, not a released default
or a measured throughput claim; see the repo-knowledge cache-policy scenario.

The integrated configuration passed the September27 C16/900s observation at
442.98 output tokens/s/chip, decode P9062.91 tokens/s/user and TTFT P95588.66ms.
This is one combination point, not a new paired comparison or a quality score.
Default selection and packaging reuse its unchanged numerical/State/attention
programs and exact kernel artifact; changing the entry default is not a new
performance measurement. See `docs/evidence/qwen35-balanced-default.json`.

## Prefill round-robin candidate (not the historical benchmark policy)

The fairness branch reserves ready decode/MTP demand, then greedily distributes
remaining tokens from a rotating prefill start. Eligible waiting requests share
that ring; full execution seats, writer fences and native allocator failures are
not bypassed. This changes grant policy, not numerical kernels or State lifetime.
`fair_schedule.py` adapts only the class-local, source-pinned native schedule's
three grant/skip seams; an unknown native method fails closed. No installed donor
file or global native Scheduler is rewritten. See `prototypes/prefill-fairness`
for tests and the qualification boundary. Earlier throughput numbers above do
not measure this changed policy. Joint Conv integration remains pending.

### Incremental State cache candidate

With the experimental host cache enabled, `state_cache_incremental: true` opts
into shared host FA pages and on-demand missing-page restoration. No predictive
prefetch is performed. GDN/continuation stays an exact private snapshot; sealed
prefix FA blocks are shared between host checkpoints, and each mutable boundary
block gets a fresh version. Native allocator blocks, not token fragments, are
the transfer unit. Epoch metadata remains destination-owned.

Device residency is a weak index over the existing native pool: releasing an
idle seat makes its blocks reclaimable, but only actual reallocation or a write
invalidates their identities. Restores pin surviving free blocks before allocating
holes. This cut does not share writable device pages across active requests and
serializes incremental maintenance transactions; unrelated inference remains
asynchronous. Host capacity counts unique payloads, including pending stores and
drops until TP quorum. The existing non-incremental mode remains available.

This is a correctness candidate, not a measured performance improvement. See
repo knowledge `study-qwen35-state-layout/incremental-cache.md` for qualification.
