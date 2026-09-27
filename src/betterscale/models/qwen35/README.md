# Qwen35 resident State serving

This is the qualified State-only integration, not the earlier standalone live
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
  --runtime live --devices 0,1 --port 8000 \
  --qwen35-runtime-dir /path/to/new-qwen35-runtime \
  --cache-dir /path/to/task-cache --served-model-name qwen35-moe
```

The launcher configures the real`betterscale.qwen35_worker.Worker`, native async
scheduler subclass, FULL4096/MTP2 and required environment before device startup.
It does not claim or acquire accelerator ownership itself. The default total State
budget is26,038,239,232 bytes/rank, E16/R20 and262144 context. Change the total via
`--state-budget-bytes`, not resident-count × context pages. The resident declaration
is1,912,095,920 bytes/rank; remaining bytes fund shared FA pages. The default admits
16,720 physical128-token pages. Explicit page counts or other seat counts are rejected
by this qualified entry. No CPU KV offload or connector is enabled.

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
