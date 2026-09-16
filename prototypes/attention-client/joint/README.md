# Attention2 + Expert2 BF16 dummy closure

Scope: Qwen3-30B-A3B's full layer dimensions, two layers, dummy weights.
Two independent TP1 native attention engines and two disjoint 64-expert servers.
No fixed prefill/decode role; each attention engine runs both phases.

This first neural closure deliberately uses host control messages and CPU route
counts, with IPC D2D for weights/activations/results. It is NOT the persistent
integer server with its arithmetic silently replaced, nor a performance candidate.
The delivered server remains untouched. This staged reference consumer establishes
real GEMM, multiple-server completion ownership and native numerical checks before
moving its control loop into the persistent device service.

Attention uses the previously qualified per-layer FULL graphs and native metadata
updates. Expert compute uses native BF16 grouped matmul/SwiGLU; each server owns
its shard. Both server receipts are required before consuming/reusing a source.
IPC output backing remains owned until the next generation is submitted.

Raw expert contributions retain top-k slots; the client combines them in one
weighted reduction after both servers return, rather than reducing independently
and introducing a different cross-server reduction order. Native routing decides
expert IDs/weights. The reference and candidate use identical dummy weights.

Prototype-only costs include local reference experts on attention ranks, startup
weight export copies, CPU route inspection and synchronous shadow comparisons.
None belongs in a claimed four-card serving throughput number.

## Acceptance — September16

Final capsule: `runs/attention-joint-20260916T025937Z/run/measurements`.
Local physical cards0/1 are attention,2/3 experts. All four are released.
`result.json` is the compact receipt; `summarize.py <measurements>` checks paired
submission/retirement identities against both expert-server logs.

- Full layer dimensions: hidden2048,32 Q heads,4 KV heads,head_dim128,
  128 routed experts,top-k8,expert intermediate768; two layers, BF16 dummy weights.
- Each attention engine passes12 native forward shadows:16/32-row prefill and
  one-row decode, with exact final outputs **and exact KV tensors**. Each has six
  attention banks and24 layer replays; the final six forwards run a sealed catalog.
- Every layer submission routes to BOTH expert owners. Each server processes48
  jobs, from both clients, using two shared per-layer GEMM graphs. Second-client
  weights are explicitly compared to the first client's weights before sharing;
  the second copy is discarded, not kept as a second expert model.
- In the sealed episode,9 of client0's12 outstanding layer intervals overlap a
  client1 interval. This is a host-visible independent-inflight witness, NOT an
  assertion of concurrent device GEMM or a latency measurement.
- Native reference and candidate begin from the SAME pre-forward KV snapshot.
  Reference output/state are saved, then KV is restored before candidate execution;
  leaving reference writes installed could mask missing candidate KV updates.
- Startup has an episode-start gate after both clients seal their banks. There is
  no four-rank barrier per layer, per decode step or per request completion.

Attention uses original native router semantics, including BF16 routing weights.
Servers use native grouped matmul → SwiGLU → grouped matmul with64 local experts.
Bounded256-row packed input is padded to a fixed graph extent; padding computes
zero-input rows on the last expert but is never returned as a live contribution.
Routing pack and output slot scatter are eager outside the expert compute graph.
This conservatively simple padding is NOT an efficient large-prefill policy.

The first attempt (`joint1`) failed during startup: importing attention internals
before native Worker startup triggers a donor circular import. Pass bootstrap
handles through the lightweight common module, letting native engine construction
import the Worker normally. `joint2` passed the initial closure; `joint3` added
same-pre-state KV restoration. The final run additionally aligns the measured
episode start, because otherwise one engine can finish before the other warms.

## Reproduce

From the repository root on this configured lab host:

```bash
bash prototypes/attention-client/joint/run.sh 0,1,2,3
python3 prototypes/attention-client/joint/summarize.py <capsule>/run/measurements
```

The launcher freezes sources and uses the existing selected-card lease/occupancy
supervisor. It depends on the pinned lab runtime and the previous validated
admission helper capsule; it is not a portable installer or public Worker entry.

## Remaining before performance work

The first BF16 consumer is intentionally host-driven, not an extension of the
persistent INT32 server. CPU routing descriptors, synchronous payload publication,
shadow copies/reference FFNs and uncoalesced source jobs remain. In particular,
sharing weights is qualified; **cross-source neural GEMM batching is not**.
A device-driven compute consumer needs its own protocol and arithmetic acceptance
before these reference costs are removed. Full48-layer real weights, long contexts,
large prefill capacity and real online scheduler integration are also unqualified.
