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

### MTP discard is a cache-validity problem, not just an omitted payload

Observed at pinned core752a3a504 and Ascend9bf964cb (both submodules now checked
out exactly, no shared runtime changed):
- vllm/model_executor/models/qwen3_5_mtp.py constructs an independent
  full_attention decoder layer and consumes normalized target hidden states
  concatenated with token embeddings.
- vllm_ascend/spec_decode/llm_base_proposer.py: propose -> set_inputs_first_pass
  consumes target hidden states and common attention metadata; the draft pass
  copies common seq_lens and slot_mapping into its own metadata banks.
- Owned apc_boundary.install_draft_boundary corrects the lookahead token at a
  prefill boundary; it does not create absent draft-prefix KV.

Inference: dropping MTP storage remains sound as a design choice, but resuming
with absent prefix KV and unchanged full-context draft lengths is unsafe. Need
an explicit invalid/cold draft state. A target-only fallback is the correctness
reference; a separately bounded suffix-only draft context is a candidate that
requires address/mask/position qualification and acceptance-rate measurement.
Neither path exists in the currently qualified MTP2-only serving entry. Target
KV alone does not contain the historical hidden-state inputs needed to recreate
identical full-prefix draft KV cheaply. Do not claim warm draft equivalence.

### Model gate environment is not yet restored

Current container has CANN9.1.0, torch_npu2.10.0.post4 and differently pinned
installed donors. Qualified BetterScale needs CANN9.0.1 / torch_npu post2 plus
its explicit vLLM/Ascend source and native payload pins. Source submodules alone
are not a built runtime. Fletcher was asked for a prior image/artifact location;
component probes can proceed, but do not bypass validation to label a model run
as baseline-compatible. P-only/no-MTP and Qwen35 DP/EP are new qualification
surfaces even after the old environment is restored.

### Pinned upstream connector seams worth borrowing

Read narrow seams rather than forking the 2059-line Ascend hybrid connector:
`upstream/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`.
Its SupportsHMA path handles MambaSpec address groups, a P-side N-1 truncation
for D last-token recomputation, and delayed freeing until transfer completion.
These are useful frontier/lifetime precedents. Its save_kv_layer/wait_for_save
are no-ops and request_finished handoff is P->D, not our bidirectional durable
session manifest/DRAM lifecycle. Owned resident GDN candidates are not ordinary
native Mamba blocks; do not register all candidate storage as the selected state.

`kv_pool/ascend_store/backend/mooncake_backend.py` wraps multi-buffer Store puts
and gets, but `put` logs errors and does not return an acknowledged result to
its caller. Our manifest publication must require positive completion; borrowing
this wrapper unchanged would erase that correctness boundary. Use lower-level
Store results or an explicitly failing wrapper. This is a local integration
assessment, not a claim about the correctness of the upstream system's cache-
miss policy. Our prototype already fails closed on Store put errors.
