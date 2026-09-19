# Qwen27 TP2: row-projection MC2 and standalone gate/up investigation

Research prototypes, **not shipped product behavior**. Evidence lives under
`runs/qwen38-tp2-serving/` (ignored); current product is98f507167b5a4832481a2187e6dcb5b5e5afc85b.
The exact product identity is also retained in the matched model capsule launcher.

## The two different opportunities

-64 gate/up projections are column parallel: BF16 `[N,5120] @ [17408,5120]^T`.
 They do **not** normally end with AllReduce. Treat their kernel/layout efficiency
 separately from communication fusion.
-64 attention/GDN output and64 MLP down projections are row parallel:
 localK3072/8704, output5120. These are natural MatmulAllReduce candidates.
-Current pinned vLLM-Ascend's `enable_fused_mc2`/`enable_prefill_mc2` references are
 not proof of a working dense projection switch: inspect the actual leaf dispatch.
 Historical `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` release notes are not a current
 capability check.

## Bounded operator observations (2026-09-19, hw3 910B2)

All tests use leased subsets and cleanup receipts. Device5 is standalone GEMM;
6/7 is TP2. ND/BF16, fixed random weights, changed input generations, FP32 oracle,
independent graph banks. Timing samples are graph-event envelopes, **not** a claim
of model or service acceleration. Profiles are separately collected.

### Gate/up

`gate-up-shapes1` PASS:1024 native624.5–624.7us,1536 native977.0–979.9us.
At1536, splitting1024+512 (1074–1088us),768+768 (1021us),3x512
(1270–1281us), and contiguous KxN weight (1012us) were all slower, including concat.
Shape sweeps covered512,768,1024,1152,1280,1408,1536,1792,2048.

`gate-up-profile1` passed native export and TraceLoom; twelve GEMMs, two per shape.
1024 is MatMulV2;1280/1408/1536/1792/2048 are MatMulV3; all report24 blocks.
1536's kernel time is higher than a linear extrapolation from1408 or backward from
1792. This supports a shape-selection/tiling hypothesis, **not** identification of
an internal tile size or proof that V3 itself causes the slowdown.

`gate-up-layout2` PASS: native vs one-time FRACTAL_NZ29 weight:

| tokens | ND, two order blocks (us) | NZ (us) | split output columns incl. concat (us) |
| --- | --- | --- | --- |
|1024|648.06 /649.96|618.17 /617.67|782.17 /783.22|
|1536|995.64 /993.86|875.05 /879.82|1014.61 /1019.13|

This is about12%1536 microkernel-envelope improvement, not12% service gain.
NZ is a native capability: `vllm_ascend.utils.maybe_trans_nz`, and BF16 conversion
requires weight_nz_mode2. The model prototype instead restricts conversion to64
MLP gate/up weights, checks exact ND roundtrip, and leaves other projections alone.
No per-forward conversion is deliberately added. `gate-up-layout-profile3` confirms
exactly one GEMM for the NZ arm, consuming ND/FRACTAL_NZ; no separate conversion
kernel is present. Both1024 and1536 keep their respective V2/V3 family. Its second
unprofiled1536 comparison is985us ND vs885us NZ (~10% benefit). ABBA model
checks completed; see the closed NZ result below. Do not enable a global weight-format change.

### MatmulAllReduce: positive capability, unqualified performance comparison

`matmul-allreduce5-fused` PASS both ranks forN1/1024/1536 andK3072/8704,
with four changed generations and alternating independent banks. The communicator
was established with a real collective before obtaining the MC2 handle. Rank0
medians (two timing blocks):

| N | K3072 fused us | K8704 fused us |
| --- | --- | --- |
|1|108.26 /106.19|103.67 /105.37|
|1024|681.92 /683.62|856.89 /857.67|
|1536|1000.57 /1002.68|1198.43 /1194.20|

Do **not** report a gain against the failed split controls below.

Failures retained, not erased or absorbed by relaxed tolerances:
-`matmul-allreduce1`: mixed PG/MC2/oracle group; split control failsN1536,K3072,
 bank0 generation2. RMSE~1.00 rank0/~1.73 rank1, far beyond BF16 error.
-`matmul-allreduce2`: separate MC2 group; stuck in capture pending-event-query wait;
 cancelled owned supervisor, cleanup retained. ReceiptRUNNING reflects interruption,
 not successful execution.
-`matmul-allreduce3-split`: isolated split process + CPU/Gloo oracle + correctness
 phase fences still fails the same generation with the same errors.
-`matmul-allreduce4-split`: retaining intermediate GEMM output does not fix it.
-`matmul-allreduce6-raw`: current-stream raw HCCL in-place also fails identically;
 PG side-stream overhead alone is not an adequate explanation.
-`matmul-allreduce3-fused`: without a real communicator warmup, first eager MC2 call
 times out with561000 / `HcclAllocComResourceByTiling ret=4`. Adding communicator
 bootstrap produced the passing5-fused capsule; do not bypass resource initialization.
-`matmul-inplace1` PASS: no HCCL, captured GEMM+in-place multiply2+consumer, two banks,
 six changed generations across all six shapes. A generic claim that GEMM cannot
 tolerate a later in-place consumer is not supported.

Next discriminator for the split failure is intermediate-value observation and/or
out-of-place communication. The native pinned NPUCommunicator inherits
DeviceCommunicatorBase's in-place PG implementation; the separate PyHccl helper
has an out-of-place API. Do not confuse those two call paths.

## Files and live experiment state

-`probe.py`: row GEMM fusion/control capsule; every run freezes its own source.
-`gate_up_probe.py`: shape/layout/split envelope and numerical checks.
-`gate_up_profile.py`: short native shape boundary profile.
-`inplace_probe.py`: single-device in-place-consumer discriminator.
-`nz_gate_up.py`: one-time model-local gate/up packing, not product patch.
-`nz_compare.py`: same-pair ND/NZ/NZ/ND candidate-only model comparison,
 cold1024 prompts and1536 first chunks of2048, APC enabled, no MTP.
-`gate-up-model1` failed before server start because trace.json was missing;
 `gate-up-model2` includes the original fixed trace and is the active replacement.

Public reference for native operator semantics:
https://ascend.github.io/docs/sources/pytorch/api_doc.html
Installed2.10.0.post2/CANN9.0.1 probe evidence, not the older API support matrix,
is the graph-compatibility authority here.


## Updated findings after real tiling and paired MC2 probes

### NZ is closed, not an optimization to ship

Fletcher explicitly redirected away from weight-format tuning on2026-09-19.
`gate-up-model2` ND/NZ/NZ/ND completed and released6/7. Rank0 period means:
ND1536=338.32/338.43ms, NZ=334.32/334.12ms; NDdecode=27.275/27.305ms,
NZ=29.026/28.973ms. The prefill saving does not justify the decode regression.
No product format change, duplicate-weight cache, or further NZ experiments.
Full compact per-rank receipts and `timing-summary.json` are retained in the capsule.

### Actual native ND tiling recovered and a controlled panel ablation passes

`gate-up-tiling1`: physical5, native ND BF16, four msprof-op profiles. Each has
24-core PMU and actual input_tiling.bin. Instruction TimelineDetail remains partial;
this is not a reconstructed instruction timeline. `summarize_tiling.py` decodes only
verified V3 TCube200-byte and L2 prefixes. V2's160-byte ABI is left opaque.
V3 dumps allocate288bytes; installed C++ MatmulTilingData is280bytes. Wrapper compile
asserts both280 and TCube200; never force a288-byte struct interpretation.

| M | kernel | Cube tile | L2 panel grid | panel rows x columns | weight set/panel | median L2 read hit |
| --- | --- | --- | --- | --- | --- | --- |
|1408|V3 key65536|128x256x64|1x5|1408x3584|35MiB|92.94%|
|1536|V3 key65536|128x256x64|3x2|512x8704|85MiB|90.25%|
|1792|V3 key65536|128x256x64|1x5|1792x3584|35MiB|93.90%|

1536 and1792 share stepKa/Kb4, A/B double buffering2, L0C1.1408 instead
uses stepKa8/depthA1=16. PMU active counters overlap, so do not sum them. Actual
kernel source is installed ops_nn/ascendc/mat_mul_v3, NOT adjacent Mc2 host code.
Prior native-FFN worktree's tune-native-bf16-ffn scenario/PMU.md supplied the proven
capture/decoding method; its4096x27648 results are not this TP2 shape's evidence.

`panel_kernel.cpp` is a small launcher of that installed unmodified native base
kernel/template. One binary accepts either captured native tiling or modified L2
fields50..54; operands stay ND, no format conversion or extra scratch.
`gate-up-panel1` all12 changed-input/bank checks and input/guard checks pass:
stock969us, same-template original tiling983us, modified1x5panels949us.

`gate-up-panel2` is the2x2 M-panel/N-panel discriminator; all20 checks pass.
Two order-block medians inus:
- stock990.76/980.78;
- same binary native3x2:997.33/1006.60;
-1x2:957.04/957.39;
-3x5:958.02/955.89;
-1x5:952.51/952.07.
Both removing M splits and narrowing N panels help in this case; neither a simple
"only cache capacity" nor "only tile count" explanation is established.1x5wins
~3.5–5% against the matched recompiled original tiling and~2–3.4%against stock
across these two bounded cohorts. No model/serving gain claimed or product patch.

### Qualified fusion comparison now exists, but the single-call failure remains

`matmul-allreduce7-outplace` was invalid: PyHccl's implicit stream was outside capture.
`8-outplace` passes the explicit capture stream but reproduces the single-call
failure at1536/K3072/generation2. `9-tap` confirms the pre-communication GEMM is
correct (RMSE~.00166) when the reduction result fails. It is not a GEMM output error.
`10-pair`, two successive projections per graph, passes24 generations, both outputs,
both banks, both ranks and all six shapes. Do not pretend this explains the lower-
level single-call failure; it establishes a different working protocol envelope.

`11-paired`: same process, separate **bootstrapped** PG/MC2 groups, two projections
per graph,24 changed generations, all1152 output checks pass across two ranks.
Unprofiled timings use split/fused/fused/split; units areus per projection including
communication and consumer. The old free-text receipt scope mentions separate
processes; structured separate_mc2/copies fields and frozen source show the actual
same-process protocol. Future receipt wording is corrected.

| tokens | localK | rank0 split | rank0 fused | reduction |
| --- | --- | --- | --- | --- |
|1024|3072|720.54|658.78|8.57%|
|1024|8704|926.65|847.98|8.49%|
|1536|3072|1059.39|970.69|8.37%|
|1536|8704|1356.05|1178.13|13.12%|

Rank1 agrees (~8.3–8.5%, down1536~13.5%). Atone token fusion is much worse:
~82us vs48–57us onrank0. The first model-local prototype tried a Python threshold of512 tokens.
**That did not preserve decode:** Dynamo specialized the branch while compiling
one1–2048 range, so generated code also ran MC2 during decode. The partial
`matmul-allreduce-model1` completed split/MC2 rounds show decode forward
26.083→28.339ms despite prefill improvements. The obsolete run was cancelled;
selected devices6/7 were reclaimed. Its comparison receipt remains RUNNING because
supervisor termination bypassed the child finalizer; it is NOT a completed ABBA.

`matmul-allreduce-model2` replaces the traced Python branch with an opaque
custom-op leaf that reads owned GDN wave metadata during eager/capture. Fletcher's
chosen policy is **all prefill/mixed, including small waves; pure decode native**.
No token-size threshold. Both routes retain ND weights; MC2 communicator bootstrap
happens once before capture. Qualification adds16/32/128-token prefill, mixed and
1/4/8-request decode, with a short profile containing both prefill and decode.
This is still experimental, not a shipped product change.

Installed MC2 source's910General path runs chunk GEMMs then HCCL Commit per chunk,
Wait/Finalize after the last one; it is a device-side producer/communication pipeline,
not magic elimination of synchronization. Confirm the actual model timeline before
attributing measured savings to overlap versus kernel/communication-path changes.


### Qualified owned-wave model route (model2)

Same-pair split/MC2/MC2/split completes; all four HTTP/step cohorts PASS and
admission exit0/reclaimed. `summarize_model.py` preserves each round and both ranks;
`docs/evidence/qwen-mc2.json` is the compact public ledger. Rank0 forward means:

| tokens / kind | split ms | fused policy ms | time reduction |
| --- | --- | --- | --- |
|16 prefill|52.300|54.555|-4.31%|
|32 prefill|55.455|57.786|-4.20%|
|128 prefill|71.044|72.494|-2.04%|
|512 continuation|147.619|143.943|2.49%|
|1024 prefill|232.142|223.026|3.93%|
|1536 first chunk|330.714|315.317|4.66%|
|1 decode|25.690|25.750|-0.23%|
|4 decode|27.826|27.825|0.00%|
|8 decode|30.329|30.413|-0.28%|

1536 remains the first chunk of2048,512 its continuation. No new native baseline
or SWE throughput claim. Mixed arrivals pass but differ across rounds, so no
mixed speedup headline. Fletcher chose all prefill/mixed without a size threshold;
small-wave regression is visible, not hidden by an automatic fallback.

`profile_mc2.py` exports native data through TraceLoom and validates both ranks:
32/1/1/1024/1/1 schedules have128/0/0/128/0/0 MatmulAllReduce kernels. The six
steps retain16FIA each and177/305 GEMMs (including LM head) according to route.
This proves actual decode capture/replay exclusion. CPU `test_qwen_mc2.py` also
executes one dynamic compiled leaf with changing metadata at the same tensor shape,
protecting against recurrence of the model1 branch-specialization bug.
