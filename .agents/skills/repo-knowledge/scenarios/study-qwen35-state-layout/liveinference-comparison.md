# LiveInference reference: understand the donor, do not extend its layout

## Fletcher's decision — 2026-09-25

Preserve the native Qwen35 layout as observed compatibility behavior. Do not
build secondary development on its shared union storage or start repairing the
connector around that representation. Study the existing LiveInference State
protocol first. This does **not** authorize porting LiveInference into BetterScale,
replacing the allocator, or building an offloader.

## Exact reference and historical status

Read-only reference checkout: `/root/my-ascend-workspace/LiveInference`, clean
at `05ac15419c0e73650e687ceb9daffeb7874865f0` during inspection.

- `engine-redesign/livemodule/0011-homogeneous-simd-state-blocks.md`
  (2026-08-25): accepted architectural direction; explicitly rejects donor
  common-BlockPool union rows as a foundation. Its implementation receipt
  generalizes native-block multiplicity and fixed prefixes.
- `src/livemodule/core/state_tensor.py`: current declarations, logical/native
  lowering, complete-allocation validation and cross-domain exclusivity.
- `src/livemodule/runtime/grouped_state.py`: current same-schema coalescing,
  exact backing spans, padding accounting and backing leases.
- `src/livemodule/runtime/state_transfer.py`: current realized-handle/range
  contract. Endpoints must agree on generation, lane ID and layout identity;
  ranges must fit their extents. Layout identity is recipe-owned, not a generic
  inference from raw pointer or tensor shape.
- `src/livemodule/runtime/host_state.py`: current logical-ID copy lowering,
  including fixed prefix and physical-block multiplicity. Its inspected enqueue
  path handles CPU/CUDA; do not describe it as a qualified Qwen35 Ascend offloader.
- `src/livemodule/llm/qwen35/gdn.py`: convolution and recurrent state are
  separately registered StateTensors; inactive speculative lanes get explicit
  scratch prefixes rather than racing on one null row.
- `tests/test_grouped_state_backend.py`: existing tests exercise disjoint
  same-backing lanes, logical/native multiplicity, leading pages, lease-protected
  retirement, domain/dtype separation and padding-aware admission. Tests were
  inspected, not rerun in this research turn.

## The distinction that matters

```text
native donor:      physical slot = union(group-specific State interpretations)
                  safety depends on common-pool exclusion of live block IDs

LiveInference:    logical row = product(exact State lanes within its domain)
                  each distinct State owns an exclusive physical span
```

This is not a ban on sharing an allocator or underlying allocation. Current
GroupedStateBackend can coalesce compatible lanes into one backing. It cannot
make their byte ranges overlap. Sharing allocation identity and sharing State
contents are different things.

| Question | Native Qwen35 studied here | LiveInference contract |
|---|---|---|
| What identifies contents? | Group + block ownership selects an interpretation of aliased bytes | Exact State identity + logical block ID + within-block coordinate |
| Can unrelated States use ID 7 concurrently? | Only with common-pool exclusion preventing overlap | Yes: each lane addresses its own exclusive span |
| Can K and SSM overlap? | Yes, under group ownership exclusion | Not if they are distinct StateTensors; validator rejects overlapping intervals |
| How does 2048→128 lower? | Must reconstruct runner/kernel layout | Explicit `physical_blocks_per_logical_block=16` can express it |
| Where are fixed scratch/sentinel pages? | Representation/path dependent | Explicit `leading_physical_blocks`, charged and excluded from ordinary logical IDs |
| What is transferred? | Connector currently guesses from registered views | Exact realized lane and explicit validated source/destination ranges |
| May allocation die during a copy? | Must recover lifecycle rules across owners | Backing leases prevent retirement; operation completion still needs its own proof |

For lane i the current declaration makes these quantities explicit:

```text
native_bytes_i = product(block_shape_i) * sizeof(storage_dtype_i)
logical_bytes_i = native_bytes_i * physical_blocks_per_logical_block_i
fixed_bytes_i = native_bytes_i * leading_physical_blocks_i
first_native_row(i, b) = leading_physical_blocks_i
                         + b * physical_blocks_per_logical_block_i
```

Current code also has separate StateDomains and shared indivisible
StateCapacityUnits: coupling capacity does not merge address spaces. Do not
mistake the original single-domain formula for a universal one-pool policy.
Grouped-backend admission charges lane/allocation padding as well as payload.
No capacity gain follows automatically: exclusive lanes deliberately give up
some heterogeneous slack reuse. Recalculate with the actual State census and
retention policies rather than promising better memory efficiency.

## What this does NOT solve by itself

`engine-redesign/livemodule/0029-cpu-offload-state-tensor-research.md` is dated
2026-08-28 and explicitly pre-implementation. Later current source does contain
host and transfer primitives, so neither its old absence claims nor its list of
research packages is a current implementation inventory. Its safety questions
still apply:

- A byte-complete restore is not proof of a semantically usable checkpoint.
- State generation is not the same thing as request/slot lease generation.
- Store must follow the final accepted-state writer; pinning only prevents
  retirement, not mutation by another writer.
- Restore completion, complete required-lane visibility and metadata readiness
  must precede graph consumption. A cache hit is not readiness.
- Native Qwen35 still needs its MTP accepted-state, +1-token identity and
  checkpoint-boundary semantics understood; cleaner storage cannot erase them.

The preceding CPU three-region copy in this scenario remains an independent
oracle for understanding the donor's bytes. It is **not a proposed transfer
layout, an adapter implementation plan, or a reason to remove connector guards**.
No source/runtime changes or hardware experiments were made for this comparison.
