# Explore PD storage and head-major transfer

Source-only research, 2026-10-01. BetterScale baseline
852c10663e703f853c81435d6fd89a6c8affdef3; Mooncake reference
0d1a8040faebb7c127c8901840a38c2ff57e80c5, inspected at
/workspace/Mooncake-reference on hw180. No runtime install or device test.

## Fletcher's intended boundary

- TP2 P instances, D pool initially DP8EP8; one active writer per request.
- Dense target KV streams incrementally; target GDN checkpoint at turn end,
  not each decode step. MTP persistence is not required; bootstrap remains open.
- DRAM/SSD cache, not all-session HBM mirrors. Do not replicate a session across
  every P instance. D has a definite attention-group owner.
- Explore per-head token chunks to avoid TP2/TP1 rearrangement. Extra bandwidth
  can be preferable to packing cost. This is a direction, not a measured result.

## Observed seams

BetterScale AttentionState declares [page_tokens, local_kv_heads, head_dim].
patches/qwen_fia/host_metadata.cpp describes paged K/V as
[pages,128,kvheads*256]. TP2 local_kv_heads=1 masks the distinction between
token-major and head-major. TP1 heads=2 does not. A reshape alone cannot change
the interpretation. Kernel reads AND cache writes need qualification before
claiming direct common-layout consumption or unchanged attention performance.

Mooncake source:
- mooncake-store/include/real_client.h: batch_put_from_multi_buffers;
  get_into_ranges with source/destination offsets, registered destination memory;
  snapshot reads fail on expired lease rather than silently requerying.
- mooncake-store/include/replica.h: replica_num=1, preferred_segments,
  hard/soft pins, optional group_ids. Groups describe lifecycle, not atomic
  multi-object publication or attention-group ownership.
- allocation_strategy.h: preferred allocation can fall back to other segments;
  preferred_segments is not a strict owner-placement guarantee.
- python/mooncake/async_store.py wraps synchronous methods in run_in_executor.
  This is host nonblocking, not evidence of device-stream overlap.
- Ascend Direct local_copy_engine.cpp has H2D/D2H batch-copy paths and async
  fallback for unsupported batch copy. Hardware/driver/registration compatibility
  and completion/event semantics remain untested on hw180.
- Store has standalone-service mode and DRAM/SSD support. LICENSE-APACHE.
  Preserve upstream license/notices if borrowing.
- real_client.cpp is 8615 lines and store_py.cpp 3475 at this pin: broad forks
  incur meaningful comprehension cost; inspect only narrow seams initially.

## Current inference and next discriminator

Prefer a BetterScale session/State adapter over Store + Transfer Engine before
forking Store. Use immutable sealed token/head chunks and a turn manifest;
do not make a whole growing request one repeatedly rewritten object. Tail
chunk policy and group eviction still need design. Byte objects do not establish
valid target State or execution ownership.

First determine whether the pinned attention read/write ABI can directly consume
paged head-major data. Then qualify range/multi-buffer transfers, lease/eviction
races and snapshot publication on CPU; accelerator overlap is a separate admitted
probe. Do not silently upgrade BetterScale's pinned CANN/donor for Mooncake.

## First CPU prototype

Read [the layout probe](../../../../../prototypes/pd-kv-layout/README.md) for
the 22 passing byte-layout tests, CPU timing limits and the installed Mooncake
import/shutdown failure. Native Store, NPU overlap and kernel head-major support
remain unqualified; do not repeat model setup before resolving that preflight.

Fletcher subsequently accepts sub-second CPU conversion as a first-cut tradeoff;
do not make zero-reorder head-major kernels a prerequisite for the connector.
The isolated generic Mooncake0.3.13.post1 wheel imports libcuda.so.1 and is not
a CPU-only escape from the installed NPU wheel failure; see the prototype README.

CPU Store blocker subsequently resolved using the mooncake-hust documented
mooncake-transfer-engine-non-cuda0.3.13.post1 in an isolated venv. The real
multi-buffer/range/lease roundtrip and clean shutdown pass; see the prototype
README and store-receipt.json. This does not fix the old NPU package or qualify
PCIe/SSD. Fork HEAD is c992ba75; the tested published wheel is not a fork build.
Configure lease TTL explicitly: upstream tests and installed defaults differ.


## Current handoff boundary (2026-10-01)

Mainline is now DRAM by Fletcher's decision; disk-native ranges are deferred
under https://github.com/vLLM-HUST/mooncake-hust/issues/3 (published in Chinese
with explicit approval). Do not let SSD block PD implementation.

The prototype README now owns session publication/fencing, selected GDN/conv
TP-layout oracles, real Store P-D-P-D receipts, and admitted NPU pinned-staging
results. Read its final sections rather than interpreting the historical initial
blocker above as current status. SQLite and opaque Store checkpoint strings are
NOT production distributed ownership or model resume evidence. Actual target
State CPU binding has a separate test. NPU0 20/80/320 MiB staged roundtrips passed;
no compute overlap or model qualification is established by those timings.

Today's new hw180 machine is explicitly assigned entirely by Fletcher, without
the former host's lease service. This dated authority does not waive future
shared-host admission. Preserve installed runtimes and qualified donor pins.
