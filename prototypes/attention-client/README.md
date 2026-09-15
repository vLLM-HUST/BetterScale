# Attention-side continuation for a shared expert server

Owned here: vLLM Qwen layer splitting, per-batch continuation and returned
contribution accounting. The server in
`/workspace/my-ascend-workspace/prototypes/pull-expert-server` is independently
owned; do not modify or assume it implements a real FFN yet.

## Current server boundary (September 15)

Read its CONTRACT.md before wiring a transport. Its eight-slot IPC envelope is
INT32 row-coded arithmetic, MAX_ROWS=8, WIDTH=64, exact generation matching.
It is NOT compatible with real BF16 hidden2048/128-expert Qwen without a new
server-side envelope. No distributed top-k reduction is currently promised.
This client therefore must first use a reference transport and expose the gap,
not silently cast model activations into the oracle packet.

## Intended integration

Native Qwen3MoeDecoderLayer.forward owns input norm/residual, attention,
post-attention norm, then MoE. Yield AFTER post-attention norm. Keep normalized
input and residual live until all routed contributions retire. Shared expert
(if the actual model has one) runs locally from that same normalized input.
Resume the next layer only after weighted routed contributions and shared output
are ready. The residual is not added twice: native next-layer RMSNorm owns it.

An attention lane owns positions, metadata, KV pages and all graph IO addresses
through its lifetime. Do not reuse model_runner's mutable singleton input batch
or output buffer for another lane while one is suspended. Disjoint request
lanes may progress, but successive decode steps of one request may not overtake.
Native engine scheduler remains owner of token/KV allocation and sampling;
this prototype's layer scheduler is not a replacement token scheduler.

Qwen3-30B-A3B has no shared expert by default. Never claim measured shared-branch
hiding from this model; optional synthetic shared work needs a separate oracle.
No TP/EP collectives may remain inside independently progressing attention lanes.
Start TP1; expert-server workers form a distinct group, not native DP2EP2 lockstep.

## Gates

1. CPU continuation oracle: weighted top-k, duplicates, stale replies, row shape,
   out-of-order completion, bounded admission, cancellation drain and fairness.
2. Native Qwen layer split with reference FFN, exact same metadata/KV and outputs.
3. Finite asynchronous small-model inference with per-lane retained contexts.
4. Server integration once BF16 payload/output, routing and completion contracts
   exist. True concurrent progress, error drain and FULL per-layer replay require
   explicit validation; CPU/reference passes are not distributed qualification.

Current files and checks are an isolated prototype, not installed Worker hooks
or new defaults. Do not patch out native DP coordination globally.

## Updated objective / receipts

Fletcher explicitly does not require fixed P/D attention pools: every attention
server should independently serve either phase. Keep FULL attention segments for
both where possible; remote expert dependencies form the explicit yield boundary.
Reference transport is authorized for model-side qualification. The delivered
server is integrated at its declared integer-protocol boundary below; this does
not qualify real remote neural FFN or end-to-end serving.

- native1: fixture failed because native ForwardContext.moe_layer_index was
  already consumed by reference forward. Not a model arithmetic failure.
- native2: snapshot/restore that exact invocation cursor; two-layer reduced dummy
  Qwen, TP1,13 native calls (16/9/2/1/128/65/256 row shapes), original-vs-split
  output and complete KV bytes all exact. Prefill/mixed/decode all included.
- graph3: entire per-layer attention half captured, three fixed-context replays
  for each distinct row count; all native output/KV checks pass. Expert execution
  remains the original local MLP. Fixed-context capture is not a dynamic-length
  qualification or proof of prewarmed production catalogs. Capture happens in
  the probe, not a proposed online serving path.

Raw receipts: `runs/attention-client-20260915/{native2,graph3}` and sibling
`*-checks.json`. All use bounded admitted single local card7 and frozen sources.
`continuation.py` is a CPU ownership oracle, not the device reduction kernel;
`scheduler.py` exercises ready-lane progress with a nonblocking transport contract.
The later metadata and independent-lane gates below now pass. Real neural
transport still requires the server to supply its BF16/quantized row contract.

### Native metadata graph updates

metadata4 rejected PrefillNoCache's None block table at native full_graph_fia;
metadata5 then exposed absent update_stream under eager runner initialization;
metadata6 exposed native global _ATTN_KEYS_BUFFER retaining the previous layer.
metadata7 scopes paged metadata, bank-private IO/metadata/handles/workspace,
layer-key cache and update stream. Original native output and raw KV checks pass,
including later same-shape graph reuse with changed input positions/lengths.
Capture-on-miss is an oracle convenience, not the final startup policy.

### Delivered server integration boundary

Workspace commit3532418 and ATTENTION-CLIENT-HANDOFF.md were received from
Fletcher. Source tree matches that commit for prototypes/pull-expert-server.
`server_contract.py` translates client routing-slot ownership into its eight-slot
INT32 envelope, exact generations and weighted top-k retirement. Tests cover
out-of-order expert results, packet-slot wrap, stale DONE and refusing BF16 input.
This is CPU ABI/reduction validation against the delivered arithmetic contract;
the server's already-passed two/three-card evidence is not recast as this client's
hardware integration. Real BF16 FFN remains outside that server's current ABI.

### Independent native lanes (lanes12)

The two-layer TP1 dummy Qwen experiment now retains two **disjoint** KV snapshots,
separate attention graph banks and separate forward contexts: 16-row prefill and
one-row decode. Both banks are built before the interleaved episode. Every layer's
attention half replays its complete graph; the local reference MLP runs on another
stream behind an input-ready event. A completed output is exposed only after its
completion event, and the scheduler does not block on an unready lane.

The fixture deliberately withholds prefill's first expert reply. Decode completes
B0 → B1 → final norm before prefill can advance past A0; prefill then completes.
Both outputs and every private KV tensor match their original native snapshots
exactly. See `independent-lanes-result.json` for the submission/retirement order.
The enclosing native run also passes all13 output/whole-KV-byte checks and10 reused
attention stages. `stream-scheduler-result.json` preserves the earlier single-lane
native graph plus asynchronous local-MLP gate.

This establishes dependency isolation, **not parallel GEMM throughput**: the
reference MLP uses one stream, the delayed reply is intentional, and these are
frozen model snapshots rather than a live HTTP scheduler. The general probe still
captures unseen banks on demand; a serving integration needs an explicit finite
startup catalog/admission policy. No release Worker path is changed.

Native KV is a tuple of typed tensor views over raw byte allocations. Python
`deepcopy` follows typed Storage and rejects the Char/BF16 alias. Snapshot each
logical K/V tensor with `clone`, keep the tuple structure, and patch both native
KV and implementation cached K/V references during capture. This fixture has
ordinary separate K/V; do not apply that cloning rule to heterogeneous compressed
state pools whose alias relationships carry semantics.


### Joint IPC acceptance (ipc3)

`ipc/` connects our externally supplied packet producer to the unchanged delivered
server binary. Local cards6/7 pass two episodes of the same captured client/server
graphs with changed input plans:16 packets, eight lane retirements, exact integer
outputs and exact weighted top-k plus local shared-stand-in reduction. Padding
and guard checks pass; both cards are released. The successful receipt is
`ipc/result.json`; frozen source and loader/binaries are in
`runs/attention-client-20260915/ipc-v3`, logs in `ipc3`.

Only one client is qualified by this new joint test. The delivered server's
separate three-card result remains its own cross-source batching evidence.
Our originally queued three-card test was cancelled before launch because card5
was occupied; the two-card test suffices for this adapter boundary. ipc2 exposed
a missing AIV ELF metadata section before any task execution; ipc3 fixes it.

**Next real integration boundary:** normalized BF16 model rows plus native route
weights → device packet producer → real routed-expert GEMM → device weighted
reduction → next attention-layer replay. The server currently has no real GEMM
consumer or BF16 contract. Neither the exact native reference test nor the integer
IPC test may be reported as that end-to-end path already working.
