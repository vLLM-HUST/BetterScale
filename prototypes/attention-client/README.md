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
Reference transport is authorized for tonight; final expert-server integration
remains part of the goal when Fletcher forwards its handoff. Do not mark the
whole goal complete on local-reference results.

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
Next gates: bank-private native metadata task updates and isolated graph IO,
then independent request/layer progression; integrate external server only after
its real payload and result contract is supplied.
