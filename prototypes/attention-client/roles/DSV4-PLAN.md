# DSV4 expert-separated serving: placement investigation, 2026-09-16

Status: source/header investigation only, no DSV4 remote-serving qualification.
The separately evolving expert GEMM implementation is not changed by this plan.

## Capacity: start with A2 + E6 on eight 64-GiB 910B devices

Header-only `weight_census.py` records exact checkpoint payload bytes in
`dsv4-weight-census.json`. Config's expert_dtype=fp4 is NOT the deployed storage:
these checkpoints contain INT8 expert matrices and FP32 scale/offset tensors.
Hidden4096, intermediate2048,256 experts,top-k6,43 target layers. One complete
layer's routed payload is6.015625GiB (6GiB INT8 +16MiB quant metadata).

| Routed payload | All servers | E4 average | E5 average | E6 average |
|---|---:|---:|---:|---:|
| Target43 layers |258.671875GiB|64.667969GiB|51.734375GiB|43.111979GiB|
| Target +0731 DSpark3 |276.718750GiB|69.179688GiB|55.343750GiB|46.119792GiB|

E4 does not fit even target-only before workspaces. E6's contiguous per-layer
ownership43/43/43/43/42/42 gives maximum46.480103GiB/server with DSpark3,
leaving17.519897GiB of nominal64GiB, NOT verified allocatable workspace.
E5's maximum52 experts/layer costs56.208496GiB with DSpark3; possible on weights
alone but only7.791504GiB nominal margin. Prefer E6 as first bring-up, not proof
of optimal throughput. A3+E5 remains an alternative after workspace measurement.
The older W8A8 checkpoint has ONE mtp layer; including it gives264.6875GiB routed,
not276.71875. Do not silently interchange the draft inventories.

0731 non-routed payload is16.295715GiB total including shared experts and draft.
This is an approximate replicated A-rank checkpoint floor, not measured runtime
residency: loader conversions, graph buffers and KV must be accounted separately.
No routed checkpoint tensor should ever be staged in client HBM. Do not expand
INT8 experts to BF16: the four-card Qwen prototype's arithmetic is not this model's
production quantized contract.

## Native integration boundary

Pinned source: runtime vllm_ascend/models/deepseek_v4.py, DeepseekV4MoE;
DSpark source: deepseek_v4_dspark.py. Keep native attention/MHC/residual and
scheduler on two attention devices, initially two TP1 sources rather than TP2
(to avoid accidental replicated routed outputs and extra TP reduction).
Retain native gate/router semantics on A: first3 target layers use hash routing;
other layers use sqrtsoftplus, correction bias, normalization and output scale1.5.
Draft is not hash routing. Preserve SwiGLU limit10 and native W8A8 arithmetic.
Send routed IDs/weights with the established hidden payload protocol.

Shared experts stay local to A. Split submit from wait/collect so shared expert
compute can be scheduled between them; wrapping the entire blocking remote call
and then computing shared experts would lose that opportunity. Preserve native
output scaling and exactly one shared-output addition. Initially disable mixed
placement and EPLB; native EP world is not the six-server transport group.

The current Qwen implementation is deliberately fixed:2 layers,2 sources,2 owners,
128 experts,64 experts/owner,top-k8,BF16. It cannot be enabled for DSV4 by flags.
Required work here: native role-specific weight loader, explicit owner/local-expert
lookup (256 is NOT divisible by6),six-peer window/collect lifecycle, layer-specific
weight addresses, W8A8 payload contract, and graph banks with shared scratch and
persistent IO. Preserve existing two-source service topology; generalize owner
count independently. Never use expert_id//64 or native floor-divided EP ranges.

## Bounded route to qualification

1. CPU-check256-expert ownership coverage/no duplicates and both checkpoint
   inventories; define quantized kernel ABI without modifying the other task's code.
2. One actual DSV4 MoE layer: normal routing plus hash case, shared addition and
   quantization oracle, initially small rank/device fixture. Measure graph memory.
3. A2+E6 target-only role loading and short native generation; measure each role's
   READY/capture peaks before enabling DSpark3.
4. Draft-inclusive numerical/quality gate, then matched-work throughput and
   aligned timeline. Report expert queue latency, A waiting, shared overlap and
   per-server expert skew, not just storage fit.

The checkpoint fit supports trying six expert servers. It does not establish
that two attention devices feed them efficiently or that separation beats DP8.
EOF/registration remain startup/shutdown work; no host RPC in normal forward.
