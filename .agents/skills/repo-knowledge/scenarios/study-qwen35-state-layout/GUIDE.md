# Study native Qwen3.5 state layout

Enter when reasoning about Qwen35 hybrid physical state, group ownership,
logical versus kernel blocks, or CPU connector registration and restoration.
The original native-layout observations remain source/CPU research, not a
connector repair plan. Fletcher subsequently authorized an independent worktree
and State implementation; enter [implementation-first-cut.md](implementation-first-cut.md)
for the declaration/allocation cut and its explicit qualification limits. The
scenario itself does not grant new hardware authority. For serving qualification
use the neighboring
`adapt-qwen35-moe` and `optimize-qwen-hybrid-serving` scenarios.

For the approved optional execution path and current graph-lifecycle evidence,
read [owned-execution.md](owned-execution.md) before extending the runtime entry.

For shared-page pressure, active request preemption, and the C16/R20 scheduler
frontier, read [seat-scheduler.md](seat-scheduler.md).

## Latest policy correction: hot resident seats

Read [resident-seat-policy.md](resident-seat-policy.md) before using any earlier
migration census or acceptance story: request finish does not free resident
State; warm continuation stays on the matching seat; unrelated overwrite is
explicit eviction. A separate checkpoint pool is not yet required.

## Accepted research boundary — 2026-09-25

Fletcher explicitly rejected secondary development on this native layout.
Understand and preserve its behavior; do not turn the connector defects below
into an implicit repair plan. Read [the LiveInference comparison](liveinference-comparison.md)
for the existing exclusive-State-lane protocol and its current implementation.
That research decision did not authorize a new allocator or offloader. The
later implementation uses LiveInference's existing State backend, rather than
repairing this native heterogeneous layout.

For the subsequent authorized ownership-replacement research, read
[ownership-handoff.md](ownership-handoff.md): it distinguishes actual adopted
LiveModule execution from semantic State allocation and records the first cut.

## Evidence envelope — 2026-09-25

Scope: Qwen3.5-35B-A3B, BF16, TP2, MTP2, native hybrid allocator used by the
September24 BetterScale FULL matrix. Do not generalize these byte counts to
27B, another TP degree, speculative depth, or state dtype.

Local inspected artifacts (outside this Git tree):

- Runtime root: `/workspace/my-ascend-workspace/runs/qwen35-moe-mtp-256k/`;
  `local-native-runtime1` supplies connector and runner source;
  `local-candidate-runtime1` supplies the executed FULL runner. AST comparison
  of `_allocate_kv_cache_tensors` and `_reshape_kv_cache_tensors` found both
  methods identical between these two snapshots. This does not establish
  whole-runtime equivalence.
- Core reference commit: `752a3a504485790a2e8491cacbb35c137339ad34` in
  `/workspace/my-ascend-workspace/runs/rp-legacy/20260903T155041Z-layout/rp-upstream-0.25.1/core`.
  Inspect frozen runtime files rather than assuming current upstream matches.
- Model: `/workspace/models/Qwen3.5-35B-A3B/config.json`, recorded model
  snapshot `712cf74392b05026a6db2bf213d343747d1f6d45`.
- FULL capsule:
  `/root/my-ascend-workspace/runs/qwen35-parallel-matrix-20260924/full-tp2-ep2-capacity32-source`.
- Original investigation:
  `/root/my-ascend-workspace/runs/betterscale-c32-investigation/20260925/`;
  `CACHE-MECHANISM.md`, `CPU-CONNECTOR-COMPATIBILITY.md`, and `FINDINGS.md`
  retain cache lifecycle, integration gates, and C32 measurement boundaries.

The grouping and physical layout below are **source-derived**, not a dump of a
live NPU allocation. [cpu-observation.json](cpu-observation.json) records a
synthetic CPU fixture with real per-rank byte geometry, N=4, and a nonzero
storage offset. It is not a DMA or service-correctness result.

## From layers to allocations: groups are not independent allocations

There are 30 target GDN layers and 10 target FA layers, plus one draft FA layer.
In `vllm/v1/core/kv_cache_utils.py`, `_get_kv_cache_groups_uniform_page_size`
chooses width 11 and partitions GDN by `layers[i::3]`:

```text
                  slot 0       slot 1        ... slot 9      slot 10
GDN group 0       g[0]         g[3]              g[27]       absent
GDN group 1       g[1]         g[4]              g[28]       absent
GDN group 2       g[2]         g[5]              g[29]       absent
FA group          f[0]         f[1]              f[9]        f[10]
                  |            |                 |           |
raw allocation    A0           A1                A9          A10
```

`g[]` and `f[]` are ordinals within each type, not model layer indices.
GDN groups have **10/10/10 real layers**, not 11/11/8. Three absent positions
represent padding in group accounting. `_bucket_layers_by_page_size` shares
one allocation among layers in each column: **11 raw allocations**, not 41.
The FA-only last slot retains the hybrid format.

Independent group block tables obtain distinct live IDs from the common block
pool. Shared address space is deliberate: two groups must not own the same
live physical block ID simultaneously. Same slot does NOT mean simultaneous
live layer states overwrite each other.

## Inside one allocation: three slabs, not interleaved pages

Per logical block, per rank:

- `C = 40,960 bytes`: GDN convolution history, 4096 channels × 5 × BF16.
  History length is kernel 4 minus 1 plus speculative depth 2.
- `S = 1,048,576 bytes`: GDN SSM, 16 × 128 × 128 × FP32.
- FA K and V each occupy S: 2048 tokens × 1 local KV head × 256 × BF16.
- Common padded page accounting is `P = C + 2*S = 2,138,112 bytes`.

For N physical blocks, offsets below are relative to the aligned raw tensor,
NOT necessarily `untyped_storage().data_ptr()`:

```text
byte offset  0                    N*C                   N*(C+S)              N*P
             |---------------------|---------------------|---------------------|
raw slabs    | C0 C1 ... C(N-1)     | S0 S1 ... S(N-1)     | V0 V1 ... V(N-1)     |
GDN view     | convolution          | SSM state           | padding             |
FA view      | padding             | K cache             | V cache             |
             |---------------------|---------------------|---------------------|
                              SSM and K are TRUE byte aliases
```

This schematic is not to scale. Each S or V block is 25.6× one C block.
Each GDN member of a slot has identical view offsets; ownership is selected by
its group block table. GDN and FA interpret the middle bytes with different
dtypes/shapes. Conv versus SSM/K versus V are **disjoint intervals of one
storage**, whereas SSM versus K is a **true overlapping alias**.

Runner `_reshape_kv_cache_tensors` constructs these contiguous slabs. FA
`get_kv_cache_shape` expands 2048-token scheduler blocks into 128-token kernel
blocks: K/V shape `[N*16,128,1,256]`. A kernel row is 64 KiB; a logical K or V
block is 1 MiB. GDN conv and SSM first dimensions remain N.

Logical block b therefore occupies these separate intervals:

```text
conv:   [b*C,                 (b+1)*C)
SSM/K:  [N*C + b*S,           N*C + (b+1)*S)
V/pad:  [N*(C+S) + b*S,       N*(C+S) + (b+1)*S)
```

It is **not** `raw[b*P:(b+1)*P]`. P is accounting, not this tensor's per-block
contiguous stride. Actual runner N may exceed scheduler minimum N; slab starts
must use allocation/view geometry, not silently substitute the minimum.
With transfer enabled an alignment prefix/suffix can also exist in backing
storage; neither belongs to the logical block grid.

## What the exact connector gets wrong

Installed `vllm_ascend/simple_kv_offload/worker.py`:

1. `register_kv_caches` deduplicates by storage pointer alone. That correctly
   merges some layer aliases but also discards disjoint conv/SSM/V intervals.
   Which interval survives depends on registration order; it is not always SSM
   that disappears.
2. `_build_block_views` interprets `shape[0] >= logical_num_blocks` as logical
   blocks and uses `stride(0)`. For FA this copies kernel rows, only 1/16 of
   the required logical span, with wrong block indexing as well as capacity.

The colocated CPU probe executes those exact AST-extracted statements, without
initializing the worker or importing NPU modules. For one 8,552,448-byte slot
(N=4), GDN-first registers only 163,840 bytes; FA-first only 262,144 bytes.
The fixture also checks SSM/K aliasing and an independent three-region block
1→3 copy with neighbors and alignment guards unchanged. That last check is an
address oracle, **not a connector fix or production roundtrip**.

A repair cannot be justified by merely adding offsets to a dedup key. It must
reconcile logical block geometry, truly overlapping ranges, disjoint slabs,
group ownership, worker mirror capacity, and scheduler accounting. Copying all
three regions is a conservative address model; choosing group-specific payloads
is a separate design decision, not accepted here.

## State validity is a second boundary

Native Mamba manager retains convolution history and recurrent checkpoints,
not an ever-growing copy of every historical activation. Its align mode has
running/speculative state blocks and checkpoint reuse; a cache hit needs a
matching GDN boundary as well as the FA prefix. FA bytes alone are insufficient.
BetterScale's capsule adds accepted-state selection, fresh-state migration,
MTP +1-token hash identity and delayed publication. These are not extra copies
of all historical hidden states; don't multiply allocation bytes by all views.

FULL graphs with stable cache addresses and dynamic block tables are compatible
in principle with external copies into those addresses. But the capsule's
`apc_boundary.py` currently asserts connector is None. Also, connector
`get_finished` at target-forward exit can precede later draft/APC writers.
This is an **unqualified ordering boundary, not a demonstrated race**: eligible
stores may refer to already-immutable earlier state. Prove the final writer for
each stored generation, load-before-migration ordering, and accepted-state
identity before removing a guard. Reference pinning prevents reallocation,
not writes into still-live state. No NPU roundtrip or C32 gain is claimed.

## Reproduce the bounded observation

Use a CPU-capable torch interpreter; this helper disables device autoload
before importing torch. It never writes to the runtime. AST extraction fails
loudly on incompatible source; failure-signature assertions intentionally stop
matching after a connector fix and require reassessment.

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /path/to/python probe_layout_cpu.py \
  --runtime /path/to/local-native-runtime1 --output /path/to/result.json
```

After note/helper edits, run `../grow-repo-knowledge/validate` from this
scenario and the CPU probe. Hardware qualification needs its own authorized
probe/admission; do not turn this research workflow into an implicit launch.

For MTP candidate selection, boundary checkpoints, seat departure and prefix-hit
resume, read [checkpoint-transition.md](checkpoint-transition.md). It corrects
the shorthand that hash insertion necessarily follows device postprocess.

Before designing new candidate-State ownership or lifecycle machinery, read
[dspark-ownership.md](dspark-ownership.md): existing module-local numerical State,
continuation-owned proposals, capacity domains and composed root lifetime.

For the proposed small-model TP1 first vertical, State census, activation
handoff and complete checkpoint/seat-turnover acceptance, read
[first-state-vertical.md](first-state-vertical.md). Model metadata is pinned;
no weights or device qualification are implied.

For16 concurrent requests with about20 hot resident seats, read
[resident-capacity.md](resident-capacity.md): separate execution/residency/page
capacities, per-rank payload estimates and token-pressure admission limits.

For why heterogeneous pooling still wastes GDN space, read
[padding-versus-pooling.md](padding-versus-pooling.md): padding arithmetic versus
free-capacity fungibility, and the separate-domain condition for lane savings.
