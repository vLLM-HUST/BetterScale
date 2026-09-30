# Finite host-scheduled EP waves

Fletcher/Lumi design, 2026-09-30. This is a **new experimental protocol**, not a
modification or qualification of the old persistent ABI. The executable policy
is `wave_protocol.py`; the bounded-enqueue host loop is `wave_server.py`.
The existing serving entry remains unchanged. A native device/IPC Backend is
not implemented by these files; the current executable bindings are CPU oracles.
Do not launch the old persistent server and label it this protocol.

## Agreed scheduling policy

Every expert owner holds its contiguous expert-ID shard at **every** target and
physical draft layer. Owners independently scan all client READY descriptors.
No global EP wave, attention-rank barrier, prefill/decode priority policy, or
resident device scheduler is introduced.

Starting from the rotating cursor, select the first eligible READY client as
primary. A decode primary may take already-ready same-layer decode passengers
in circular order. Freeze the membership immediately; never wait to fill.
Prefill and mixed primaries are singleton waves, and cannot be passengers.
Advance the cursor to the primary's successor, not the last passenger's successor.
Capacity is one maximum-prefill input-token width, not twice that width. Whole
requests must fit; do not split them to fill spare space. Count MTP verification
**token rows**, not request count. Routing scratch must cover `tokens * topk`
even when all routes land on one owner; average EP balance is not a bound.

Round-robin gives service opportunities, not equal time or a strict latency SLO.
We explicitly accept prefill bubbles and a next PULL that outlasts current DOWN.
Do not add priority/preemption to hide this admitted case.

## Client publication and result ownership

One immutable live request per client/owner mailbox:
`(client, generation, layer, token_rows, kind)`. Device route data retains original
`(token, topk_slot)` identities and expert IDs. Client hidden/IDs/probabilities
are stable until retirement. Producer writes payload and descriptors completely,
then release-publishes READY; a consumer must observe a coherent descriptor with
matching generation, not infer visibility from a host enqueue.

The client publishes to all owners before waiting, computes its local shared
expert once, then collects routed outputs. A native binding must preserve this
order under graph replay and changing routing. This Python policy does not move
per-token route IDs to the host; native input grouping belongs to the finite
PULL/preparation operation on the server.

Each server groups its local routes by expert within the wave's one layer.
Flattening `(layer, expert)` into a catalog ID is optional; layer identity must
never be lost. Each server returns unweighted BF16 route outputs; the client
weights and reduces in original top-k order in FP32 before casting once. No
owner-local partial weighted sums are introduced. Each original route has one
writer. An owner with no routes still completes the generation (zero work is not
absence of a response). This first contract notifies all owners, not a dynamic
expected-owner-mask optimization.

Client destinations are `[token, original_topk_slot, hidden]` in a stable owned
bank, not `[token, all_global_experts, hidden]`. The protocol admits one live
request per mailbox, so the bank cannot be overwritten by a newer generation
until the current one is consumed. Bootstrap carries IPC capabilities privately;
READY should not introduce untrusted arbitrary addresses.

## Two return modes, one retirement contract

**Push:** DOWN writes owner scratch, then a finite PUSH scatters route outputs
into the client destination bank. Publish per-request DONE only after writes
are visible. Owner output scratch may be reused then; client destination/input
ownership still lasts until client consumption and ACK.

**Pull:** DOWN writes a server-owned output bank. Publish OUTPUT_READY plus
`(wave_sequence, slot, request_generation)`; the client asynchronously pulls its
routes, then acknowledges **after the pull and consumption complete**. The owner
must retain that output slot until every member has acknowledged. Metadata
receipt is not retirement. Two output slots provide ping-pong buffering; a slow
client can pin a slot and exert backpressure. Never overwrite to keep Cube busy.

`Scheduler` conservatively reserves two unified wave slots; each may own separate
input/intermediate/output allocations. Push releases a slot only after PUSH
completion; pull releases after all member ACKs. This bounds native lifetime
requirements without claiming minimum memory or maximum possible overlap.
Do not alias UP intermediates, DOWN output, next-wave PULL input or in-flight
return storage. A later native binding may separate pool lifetimes only with
explicit dependency and reuse proofs.

Repeated observation of an unchanged READY is normal, including the last retired
READY that remains in the mailbox. Mutated live descriptors, skipped generations,
stale ACKs and output-before-DOWN are rejected. EOF names the exact fully retired
generation. Cancellation is fail-stop, not successful drain: quiesce outstanding
owned device operations and coordinate peer abort before freeing IPC mappings.

## Finite operations and stream order

```text
Cube:     UP(n) ----------- DOWN(n) ---------------- UP(n+1)
Vector:   PUSH(n-1)  GATE(n)  PULL/prepare(n+1)  PUSH(n)
```

- UP(n) waits for its PULL and previous DOWN.
- GATE(n) waits for UP(n); DOWN(n) waits for GATE(n).
- Next PULL/metadata preparation is enqueued after current GATE, not before it.
- Next UP waits for both next PULL and current DOWN.
- PUSH(n) waits for DOWN(n); its vector-stream position follows next PULL if
  one was admitted. A long next PULL may therefore delay current PUSH; accepted.
- If there is no next wave or no free slot, **skip prefetch and return now**.
  No next request is needed to flush the last result.
- In pull mode, query DOWN completion and publish output metadata. Output remains
  pinned during independent client reads; no server PUSH is submitted.

These are queue dependencies, not claims about physical AIC/AIV overlap. Native
kernel placement, stream execution and host launch overhead require observation.
The host owns the loop; all device operations are finite. `tick()` queries
completion without device synchronization, freezes a ready snapshot and enqueues
at most one compute wave plus one lookahead PULL. Two reserved slots bound work.

## Executable evidence and next implementation gate

Run from the worktree:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_expert_wave*.py' -v
```

15 CPU tests cover primary rotation, same-layer decode batching, singleton large
waves, no wait-to-fill, immutable membership, capacity and generation validation,
push-versus-pull slot lifetimes, empty-owner completion, final-wave flushing,
long-PULL bubbles and the two-stream dependency order. NumPy exercises grouped
FFN route scatter/gather for EP2/EP4, layers0/40, five changing generations,
nonuniform probabilities, skew and independently progressing owners against a
per-token matrix oracle. Push and pull results agree exactly within each EP
configuration; grouped versus independent FP32 paths use a numerical tolerance.
This is **not Qwen real-weight, BF16, IPC visibility, graph replay, or NPU speed
acceptance**. The simulated engine durations are arbitrary dependency fixtures.

Before serving integration: bind coherent READY/ACK snapshots and finite native
routing/GMM/gate/return operations; verify actual device lifetimes with a bounded
multi-client EP leaf. Require changed real-weight outputs, empty/skewed owners,
replay reuse, cancellation/drain and push/pull exact-generation parity. Then
compare modes on the same admitted hardware/protocol. No return-mode performance
winner has been selected, and the old persistent path remains the rollback.
