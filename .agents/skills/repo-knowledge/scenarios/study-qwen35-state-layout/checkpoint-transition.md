# Qwen35 MTP: candidate, checkpoint, departure, resume

Source-only follow-up to ownership-handoff.md, 2026-09-25. Same runtime/capsule
identities. No runtime edits, device imports, NPU probes or numerical validation.
A small CPU integer calculation checked the example below; it is not a kernel
execution or a recorded request trace.

## Three identities, not one seat

1. Request identity chooses a stable **control seat** in `device_apc`.
   `columns[seat]` and `selection[seat]` remember which running state to consume.
2. Request/group block-table entries choose the actual GDN/FA **physical rows**.
   GDN contents do not reside in the control-seat arrays.
3. Prefix hash plus group identifies a retained **checkpoint/cache entry**.
   Its lifetime can outlast the request and its seat.

Thus current native GDN is not already a clean seat-owned storage domain.
Seat-owned live State would be a redesign, not a renaming of current storage.
Temporary unscheduling preserves the control seat. Finished/preempted/resumed
request notifications remove its map entry; a resumed request is treated fresh.
A different request with the same prefix need not inherit the same seat.

## Follow one request

### 1. Admission and first execution

MambaManager.find_longest_cache_hit searches backward for a matching aligned
checkpoint, not every preceding GDN snapshot. Earlier table entries can be null.
The hybrid coordinator also requires compatible FA coverage. Mamba align
allocation reserves a running row plus speculative rows; it does not allocate
one historical GDN snapshot for every visited token position.

`device_apc.prepare` derives source column from `(computed-1)//B` for a fresh
request, or remembered `columns[seat]` otherwise. Fresh selection is 1;
continuing selection comes from the seat. Destination is
`ceil((computed+scheduled)/B)-1`. If columns differ, the selected source state
is copied into the destination and selection becomes 1. Cold computed=0 has
no source; initial-state metadata, not a checkpoint copy, governs initialization.

### 2. Verification leaves candidate state, not necessarily a copied winner

`service_metadata.py` sends accepted selection and block-table-derived slots
to convolution and recurrent kernels. Recurrent candidates occupy multiple
rows; convolution stores a sliding history with speculative extension.

After sampling, `device_apc.postprocess` counts non-minus-one output entries.
This accepted count includes the ordinary output/bonus position; do NOT treat
it as the number of accepted draft tokens alone. The Ascend fused postprocess
then decides whether an aligned checkpoint needs materializing. When no
boundary needs a copy, the running winner can remain selected by an index;
there is no unconditional whole-state copy to a unique committed row.

### 3. Boundary checkpoint is not necessarily the latest accepted state

For the inspected V1 Ascend kernel, let B be block size, c previous computed,
q scheduled, d draft count, a accepted output count:

```
r = c + q - d
new_computed = r + a - 1
boundary = floor(new_computed / B) * B
copy iff boundary >= r
bias = boundary - r
checkpoint_column = boundary / B - 1
```

Example B=2048, c=2046, q=3, d=2; prepare chooses running column 1:

| a | new computed | checkpoint action |
|---|---|---|
| 1 | 2047 | No new boundary copy |
| 2 | 2048 | Copy boundary2048 with bias1 to column0 |
| 3 | 2049 | Same boundary2048 copy; running state continues beyond it |

For convolution, copy the source row's history slice starting at bias into the
checkpoint row's beginning. For recurrent state, copy the full state from
`table[running_column+bias]` to `table[checkpoint_column]`. These are different
selection operations, not two identical memcpy ranges. Unused convolution tail
must not be mistaken for meaningful committed history.

The accepted-selection output resets to 1 when running and destination columns
coincide. Otherwise it remains the accepted selection. Consequently progress
count, running-state selection and checkpoint position are three separate facts.

### 4. Index publication is not physical readiness

`KVCacheManager.allocate_slots` can invoke coordinator.cache_blocks during
allocation, capped by known request tokens. Capsule BoundaryScheduler further
caps by available +1-token hashes. Therefore earlier shorthand “publish only
after postprocess” is NOT an exact account of native host hash insertion.

MambaManager tracks cached_blocks_this_step and refuses another request's
same-step hit. The device path separately orders copies: runner global_stream
waits sampling_done_event, invokes postprocess, which records _mtp_apc_done;
next execution waits that event before shared-table/count preparation.

These are separate protections. A future protocol should distinguish reserved/
indexed entries from recoverable entries and retain completion dependencies.
This inspection does not prove all async scheduler/abort interleavings or a
cross-worker publication quorum. Do not claim such proof from the presence of
one event or hash. CPU connector integration remains unqualified.

For MTP, prefix ending at B also needs token[B] in its identity because draft
inputs are shifted. `apc_protocol.extend_hashes` requires that lookahead token
to be known. Matching target tokens alone is insufficient for the draft KV.

### 5. Request leaves; checkpoint can remain

Scheduler _free_request_blocks can defer return when an in-flight step may
still write (subject to defer_block_free configuration). Its deferred queue is
fenced by processed step sequence. This branch is a source fact, not evidence
that every launch enables it.

BlockPool.free_blocks drops references. Refcount-zero hashed blocks remain in
the free queue and hash index until eviction; release is not immediate erasure.
Unhashed blocks are preferred for reuse. Control-seat removal does not delete
retained checkpoints. Cache retention is opportunistic, not a lifetime promise.

### 6. A new seat resumes the prefix

Suppose a new request hits B=2048: its Mamba table ends with the cached row at
column0. The manager pins hits and allocates writable running/candidate rows.
For a positive continuation up to the next boundary, prepare chooses source0,
destination1, fresh selection1 and copies the checkpoint into running storage.
FA keeps its token-page coverage; it is not restored by this GDN copy.

The next GDN execution consumes the new writable state, not the old request's
seat. The semantic requirement is to prevent running updates from mutating a
checkpoint still shared by prefix hits. Keep checkpoint ownership independent
from the new seat and preserve completion-before-consumption.

## What transfers to the new State design

Preserve: exact conv+recurrent checkpoint semantics; selected accepted state;
compatible FA/draft prefix identity; immutable/shared checkpoint versus mutable
running state; final-writer/readiness fences; delayed reuse of in-flight rows.

Do not preserve as architecture: common-pool exclusion between GDN and FA;
GDN history represented by token-column lists with null prefixes; relocating
running state merely because a padded token block column changes; pointer-table
layout inference; synthetic token counts used to drive a general postprocess
kernel as a precopy primitive.

Candidate conceptual domains are live seat/candidate State, retained GDN
checkpoint State, and FA token-page State. This does not freeze their APIs,
capacity ratios or copy policy. There may be an index-only commit when safe;
do not require an extra copy every step just to fit a diagram.

LiveInference has `Qwen35GDNStateProgram` plus a concrete CUDA precopy program
with StateTensor lanes and accepted-state/anchor inputs. The inspected Ascend
tree has no matching named Qwen GDN state implementation. That is useful
semantic reference, NOT a drop-in qualified Ascend replacement. Its precise
boundary behavior and recipe wiring need comparison before reuse.

Next research hinge: map these three domains to actual StateTensor declarations
and the existing capacity/lifecycle API without reintroducing a single global
block-count authority. Include checkpoint budget/eviction and failed restore;
State allocation alone does not implement those policies.
