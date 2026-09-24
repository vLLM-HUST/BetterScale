# MTP expert execution qualification

Run `multi.py` with the pinned runtime and installed MOD package, under the host's
admission/lease/foreign-owner guard. It calls packaged `Deployment`; only request
generation and assertions live here. Default C64/128 requests exercise257..32769
input lengths, 4096-frame boundaries, mixed target/draft work, APC reuse and128
forced output tokens. Sampling uses real MTP acceptance, never synthetic rates.

This is an execution/liveness check, **not** a semantic-quality score, AgentX
Frontier point or isolated throughput comparison. The separate native-engine
retrieval inquiry owns the observed repeated-7 failure. Keep it visible without
requiring a generated-text oracle for communication/generation correctness.

Acceptance requires41 native routed shadows per attention rank;164 total bytes
of local routed metadata and no expert payload; successful native MTP acceptance;
zero output-length/request errors; two directly launched persistent kernels per
owner; exact client/owner generation equality; clean all-role exits and the
external supervisor's device-release observation. Server graphs are not admitted
as a substitute for the direct resident launch contract.

`synthetic_probe.py` separately qualifies the explicit benchmark-only sampler at
TP1/TP8 against an independent CPU oracle and cross-rank agreement. Fletcher stopped
the AgentX comparison before completion and switched to `vLLM-HUST/swe-prefix-reuse`.
Do not resume the old matrix or relabel its partial data.

`benchmark.py` uses installed Deployment for candidates, and a tiny native
observation Worker for DP8EP8/TP8EP8 controls. Controls prove all41 target/draft
layers have32 local experts and a complete nonoverlapping256-expert union;
eight independent replicas are not an admissible baseline. Both paths include
the qualified async feedback ownership repair and use **real** MTP2 acceptance,
query4096, FULL decode and native262144 context. The SWE client uses unchanged
prepared input deltas and actual generated token IDs, not synthetic acceptance.
This is fixed-shape serving, not correctness of generated tool calls or SWE solving.

The session relay retains `agentx_router.py` as its historical filename but now
routes by SWE `cache_salt` when no correlation header exists. New session plays
round-robin, later turns stay on their rank; request bodies/SSE bytes are unchanged.
Both native DP and separated groups use the same loopback relay. `test_router.py`
checks salt affinity and exact body/stream preservation. Keep server-side APC
metrics alongside client occupancy/turn coverage; correct IDs alone do not prove
cache hits. Each900s C64 run owns a fresh admitted eight-device deployment.


## Two/three-card real-expert leaves

Use `ep_leaf.py` for a client plus EP2 owners; `--devices` is an explicit physical
client,owner0,owner1 map. Never infer physical IDs0..2 from an owner count.
`--sources 2 --shared-client-device` colocates two independent clients on the
first card, retaining separate generations; this tests co-batching with EP2 on
three physical cards, not four-rank scaling. EP4 requires a separate allocation.

`batch_leaf.py` uses three distinct physical cards: client0,client1,one owner.
Only real target layer0 and physical draft40 are loaded, so this bounded E1 leaf
is legal; it does not loosen the product's full-model E2/E4 requirement. Supply
a source-owned layer build with `--sources-per-wave 1` or7. Ready-source admission
must produce exactly432 waves under cap1 and fewer under the current cap7 fixture;
wave/generation checks are in the leaf, not a guessed performance score.

Example commands **inside the external selected-device admission scope**, with
CANN environment and `BETTERSCALE_EXPERT_EXTERNAL_WATCHDOG=1`:

```bash
python ep_leaf.py /absolute/model --output /absolute/new-ep \
  --build /absolute/ep2-cap7-build --owners 2 --devices 4,5,6
python ep_leaf.py /absolute/model --output /absolute/new-ep-multi \
  --build /absolute/ep2-cap7-build --owners 2 --devices 4,5,6 \
  --sources 2 --shared-client-device
python batch_leaf.py /absolute/model --output /absolute/new-layer \
  --build /absolute/layer-cap7-build --devices 4,5,6
```

Device IDs above are examples, never reservations. The supervisor starts owners
before clients, watches failures, bounds completion and terminates only children.
The external guard owns leases, hardware observation and process-group cleanup.

2026-09-24 local receipts: workspace `runs/qwen35-expert-mtp/20260924-local-leaves`.
All four accepted runs pass, all roles exit0 and4/5/6 return to idle. Layer cap1:
432 frames/432 waves; cap7:432/326 (106 paired). EP2 one-client:28 numerical/replay
cases; EP2 two-client:29/client,126 frames per owner,94/95 waves (32/31 paired).
Worst relativeL2 across these independent unfused full-output comparisons is
0.004572 (gate0.02). This establishes bounded real-weight FULL reuse/co-batching,
not whole-model MTP semantics, EP4, or a serving-performance gain. Layer bursts
use hot8, EP multi burst uses broad routing; only the final repeated-burst output
is compared, in addition to every changed-input case. Preserve these populations.
