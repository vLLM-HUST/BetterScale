# Investigate PCP KV prefetch behind donor matrix work

Scope: source/retained-profile investigation plus a bounded, synthetic two-card
transport/compute interference probe. No serving defaults, PCP ownership or
production graph changes. A visible overlap in a timeline is not proof of free
bandwidth or of improved serving throughput.

## Existing windows in the pinned donor

- `attention/context_parallel/dsa_cp.py::_forward`: prefill launches
  `_maybe_all_gather_o_proj_full_weight` after Q projection/RoPE, before WKV,
  indexer/compressor and sparse attention. `forward` waits for these handles
  in `_switch_o_proj_to_full_weight` before the output projection consumes them.
  The existing async helper is `distributed/utils.py::all_gather_async` and
  uses native `all_gather_into_tensor(..., async_op=True)`. This is already a
  working example of early issue / late consumption, not an unused comm slot.
- `attention/dsa_v1.py::_mla_prolog_multistream` (DP/native DSA) explicitly
  pairs Q-A GEMM with KV quantization; Q norm/quant with KV GEMM; Q-B GEMM with
  KV norm/RoPE/scatter. Its Vector-side budget is already occupied.
- `ops/fused_moe/fused_moe.py::_forward_shared_experts` schedules shared gate/up
  against dispatch, its activation against routed GMM2, and shared down against
  combine. Inserting a KV collective into this region adds competing traffic,
  not another independent full-bandwidth communication engine.

## Which matmuls deserve the first experiment?

1. Current-layer Q projections: independent of old KV payload; deadline is
   indexer / attention consumption. Indexer history must be ready before QLI,
   while compressed attention KV may have a later deadline at sparse attention.
2. Output projection and FFN of layer L: can prefetch **old prefix** of L+1,
   which already exists. Newly appended L+1 KV cannot be fetched before it is
   produced. Large prefill shapes offer a longer window than tiny decode GEMMs.
3. Pure-Cube-looking BF16 output projection is a stronger initial resource
   candidate than assuming all quantized or grouped GEMMs leave Vector free.
4. Routed GMM and shared-expert windows require a full-section test including
   existing EP traffic, routing/unpermute and fused quant/activation. A plain
   GEMM microbench does not qualify them.

`tp046-matmul-inventory.json` is extracted from retained run046 rank0 native
TASK/COMPUTE_TASK_INFO, not from the aligned visualization. It includes shapes,
reported blockNum/mixBlockNum and overlap with native AllGather envelopes.
`inspect_profile.py DB --out FILE` reproduces it read-only. These envelopes
include queue/wait time: intersecting them does NOT prove bytes moved throughout
that intersection. mixBlockNum describes the reported kernel launch, not a PMU
measurement of Vector saturation.

Notable entries: prefill QuantBatchMatmulV3 [516,1024] -> 32768 reports24/48
blocks, GroupedMatmulSwigluQuant reports24/48, GMM down24/24; BF16
TransposeBatchMatMul and MatMulV3 report24/0. One QuantBatchMatmulV3
[516,4096] ->4096 launch reports24/0 with79.6% of its aggregate task duration
intersecting AllGather envelopes. Do not infer the corresponding standalone
slowdown from this profile; it contains no counterfactual execution.

## Prototype protocol

`probe.py` runs W8A8 NZ projection-like shapes plus a BF16 output-projection-like
shape, at small and prefill row counts. For each it tests AllGather and owner
Broadcast, at16/64MiB receive capacity, under four programs:

- compute alone;
- communication alone;
- serial compute then communication;
- separate captured compute/communication blocks with stream-event fork/join overlap.

The collective uses already-contiguous fixed buffers; there is no paged State
packing, KV conversion or new-token cache production. Same-rank compute data is
identical between modes; output is compared exactly with standalone execution,
and receive values checked. Each captured block contains32 repeated operations to amortize host submission.
Three alternating-order repetitions measure total block time plus separate
compute/communication event spans, normalized by32. These are contention /
throughput-block measurements, not single-layer latency predictions. Events
are recorded OUTSIDE graph capture; independent compute/transfer streams fork
from one event and join before reuse. Native HCCL dependencies
stay intact. No special Vector core quota is requested.

Interpret both compute slowdown and joint completion. If M0/C0 are isolated
compute/communication times and M1/C1 their concurrent spans, then `M1/M0 - 1`
measures compute-side interference (event-span proxy). Net benefit is matched
serial block time minus overlap block time. Do not add an overlap interval to saved time or
claim no interference merely because joint execution is faster.

Two cards, synthetic payloads, cached/repeated tensors and unrelated jobs on
other NPUs do not establish TP8 HCCS throughput or actual donor GMM behavior.
AllGather and Broadcast have different per-rank/network byte counts; do not
rank their efficiency solely by the common receive-buffer size.

## Integration invariants, before any production implementation

- Transfer source must be published old-prefix state; owner mutations, slot
  reassignment and eviction must not race the read. Preserve recurrent tails.
- Helpers wait only at their actual consumer; do not place an early host fence
  that serializes the entire candidate compute window.
- Preserve identical collective order across participating ranks. Moving a KV
  collective onto another stream/group does not create extra physical bandwidth;
  check ordering/deadlock and interaction with TP/EP communications explicitly.
- Scratch reuse requires a credit after the previous layer's last KV reader.
  If prefetch starts after that attention finishes (during output projection or
  FFN), a single cross-layer buffer may suffice. Earlier lookahead overlapping
  both readers needs distinct storage; do not blindly reuse one scratch buffer.
- AllGather means shards from every rank. If one owner holds the desired whole
  KV, Broadcast or bounded owner-to-helper pulls may be the right transport;
  the observed AllGather overlap is a scheduling lesson, not an ownership model.
- Measure owner pack + transfer + consumer wait + any compute slowdown, plus
  scratch reserved memory. Accept only a net improvement in the target section.

Prior native evidence: workspace `prototypes/hcomm-overlap/README.md` measured
1.15–1.18x improvement for a two-card FP16 double-buffer direct-HCOMM toy pipeline;
that is feasibility evidence, not a number transferable to donor W8A8/EP.
`prototypes/remote-kv-pull/BROADCAST_CONTRACT.md` records a compact64K-capacity
C4 wire of18.363MiB and fair leaf-layer peer/broadcast controls. Whole-layer
broadcast did not automatically win even when its standalone bandwidth did.

Primary background: Huawei's
[HCCL kernel API ordering](https://www.hiascend.com/doc_center/source/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0801.html)
distinguishes prepare/commit/completion dependencies; its examples do not qualify
our installed AIV backend or promise universal communication/GEMM overlap.

## Execution boundaries encountered

- Local device6 became occupied during admission; the launcher failed closed.
- The first communicator attempt rejected HCCL_CONNECT_TIMEOUT=90 (minimum120);
  corrected before any measurement.
- In-graph timing events could not be read on this runtime (`event recorder null`,
  507000). The current probe uses external timing around captured blocks instead.
  The same local attempt also lost device7 to a foreign process; no local timing
  is accepted. The selected resources were released, not the foreign process.
- hw3 is used for the bounded two-card follow-up because no clean practical
  local pair remains. This exception does not authorize occupying an unrelated
  gang; the same remote ~/tp8.lock and30-second admission apply.

## Accepted bounded observations (2026-09-13)

All36 cases on hw3 physical2/3 completed; both ranks passed output checks for
all432 records/rank. The local physical0/1 follow-up completed12 cases each for
explicit AIV and unset/default expansion mode (144 records/rank/arm). INT8 weights
report actual format29 (NZ); BF16 weights format2. Repeated graph compute is
compared to same-input standalone output, and collective receive values are
checked. Payloads stay constant in these timing tests; this is not a changing-KV
publication/lifecycle oracle. Each program warms before measurement.

The local same-pair AIV arm,16MiB AllGather receive capacity:

| GEMM proxy | M | Serial block /32, ms | Overlap block /32, ms | Net reduction | Compute slowdown across ranks |
|---|---:|---:|---:|---:|---:|
| Q-B-like INT8, K1024/N32768 |516|0.5916|0.4574|22.7%|139.8–142.0%|
| Q-B-like INT8, K1024/N32768 |4096|1.3481|1.0109|25.0%|11.4–12.1%|
| Dense-like INT8, K4096/N4096 |4096|0.6846|0.4547|33.6%|84.0–87.5%|
| BF16 projection-like, K1024/N4096 |4096|0.5537|0.4523|18.3%|156.6–164.7%|

**Useful overlap and material compute interference coexist.** Larger Q-B compute
can amortize traffic much better than the small shape. Even the BF16 matrix-only
proxy suffers under sustained communication: separate Cube/Vector resources do
not prove zero memory-system or scheduling interference. These timings alone do
not identify the physical cause; no PMU attribution is claimed.

The same local pair with expansion mode unset still exhibits interference. For
Q-B M4096, net reduction is19.6%, compute slowdown19.7–21.9%; for BF16 M4096, net
reduction18.1%, compute slowdown131.3–166.6%. This is sequential AIV/default
observation with three alternating serial/overlap trials within each arm, not a
hardware-level proof that all interference comes from AIV or that default means
a specific SDMA kernel. The larger change was program shape, not simply unsetting
one environment variable.

Broadcast is asymmetric: local AIV Q-B M4096 gives1.717→1.186ms normalized block
cost, with source compute +2.1%, receiver +31.7%. BF16 M4096 source +1.6%, receiver
+138.7%. Source and helper must both be measured; watching only the owner can hide
helper slowdown. Broadcast sends16MiB from one source whereas two-rank AllGather
here exchanges8MiB contributions to form16MiB; these are NOT equal-byte backend
speed rankings.

The broader hw3 matrix includes64MiB. Example Q-B M4096 AllGather:2.599→1.726ms,
but compute slows90.4–91.7%. Do not extrapolate “16MiB partly hidden” to unlimited
context capacity. Cross-layer prefetch needs an actual available compute window
and bounded byte budget.

Results (rank medians retained, not only aggregate wins):

- [`hw3-pair23-result.json`](hw3-pair23-result.json)
- [`local-aiv-result.json`](local-aiv-result.json)
- [`local-default-result.json`](local-default-result.json)

The raw local capsules are under `runs/kv-prefetch-overlap-hw3-20260913-pair23`
and `runs/kv-prefetch-local-20260913/{aiv,default}`. Remote original:
`/workspace/my-ascend-workspace/runs/donor-kv-prefetch-overlap-20260913-pair23`.
The launcher and admission/during/release records preserve hardware co-residency;
selected NPUs returned to baseline, foreign jobs on other cards were preserved.
No timing evidence from the failed local probes is accepted. The local8-card
window did not remain free (6/7 occupied), so no8-rank result is claimed.

Recompute summaries with `summarize.py CAPSULE --out RESULT`; use
`--expect-cases 12` for the local reduced matrix. Invoke `probe.py` only under the
host's shared lease plus fresh selected-card health/HBM/process admission; it
does not reserve devices itself. Current shapes are representative rather than
a replay of actual donor GMM routing, and the32-operation blocks deliberately
create sustained traffic. They cannot be substituted for the latency of a single
prefetch inserted into one real layer.

### Decision supported by the evidence

Proceed with a **bounded old-prefix prefetch window**, not indiscriminate
communication alongside every matmul. The first whole-section candidate is
layer L output projection / FFN overlapping L+1 old-state transfer, or an earlier
same-layer Q projection if its indexer/attention deadline leaves enough room.
Keep the original native overlap and charge any new contention against net saved
time. A plain-GEMM win does not qualify an MoE section whose GMM, activation,
unpermute and EP communication are already jointly scheduled.

Outstanding before adoption: real owner-State packing/published-tail protocol,
real grouped/fused MoE interaction, single-layer deadline and eight-rank fanout,
whole-root graph scratch lifetime and memory budget. No production integration
has been performed or authorized by this exploratory result.
