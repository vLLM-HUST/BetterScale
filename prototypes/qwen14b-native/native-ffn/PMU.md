# Per-core on-board msprof op comparison

2026-09-15, local910B2 physical7/CANN9.0.1, BF16 X4096x5120, NZ W5120x27648.
Compare native MatMulV3 and native SwiGlu separately with v7 pair and v8 full
buffer kernels. No down GEMM. Stateless leaf replay under msprof op, first
matching invocation; NOT whole-FFN graph timing. Each profile uses its own
process and profiler replay state/cache conditions. No quality claim.

Invocation: msprof op --aic-metrics=TimelineDetail,Default --replay-mode=kernel
--warm-up=0 --launch-count=1 --kernel-name=<prefix> --application='<python> app.py
--mode <mode>'. Prefixes qwen_bf16_native_swiglu*, MatMul*, SwiGlu*. The retained
app uses the same deterministic dummy setup; three application calls warm the
unprofiled path, but profiler selection captures the first match rather than
skipping those calls. Do not call this a warm-graph capture.

## Partial success, explicitly bounded

Hardware PMU CSVs contain24 AIC rows for each matrix kernel and48 AIV rows for
both fusion kernels and native SwiGlu. BasicInfo, PipeUtilization, Memory/L0/UB,
L2Cache, ArithmeticUtilization and ResourceConflictRatio parsed successfully.
TimelineDetail FAILED on all four: kernel context/argument dump failed, then no
available dump to parse. Tool printed0success/1failed for the detail path despite
exit0 and usable PMU CSVs. These are counters, not instruction timelines; never
invent temporal overlap diagrams from them. Full raw tree~909MiB remains local.

Earlier v1 torch-npu MSTX range selected no kernels; v2 direct library range
returned0 and was failed closed. Prefix selection fixed collection. Do not
repeat those failed MSTX wrappers without a new hypothesis. Two changes in v2
also removed --dump=off; do not attribute v1 solely to that flag.

## Observed per-core medians

| metric | native matmul |2slot fusion|full-buffer fusion|
|---|---:|---:|---:|
|AIC Cube-active|3296.48us|3330.71us|3339.04us|
|AIC MTE2-active|3092.99us|3978.93us|4855.49us|
|AIC L2 read hit|93.54%|70.18%|66.09%|
|AIC wait-id9 (return credit)|not applicable|13.21us|absent by design|
|AIV Vector-active|separate kernel147.21us|257.30us|249.62us|

Pair credit wait min12.89/max14.01us across24 cores. Pair Vector wait-id8
(ready) median3322.97us. In full-buffer mode ready waits are spread over flags
8..15, so its id8 alone is NOT comparable. Native SwiGlu task217.56us. Pair task
4246.84us; full task4914.40us. These durations are profiler-scope, not serving.

GM->L1 logical transfer count552960KiB per AIC in all three matrix variants;
Cube arithmetic-active time is similar, while transfer-active duration and L2
read-hit differ materially. Do not call GM->L1 bytes physical HBM bytes. The
fusion does not simply execute more counted matrix-input transfer volume.
Different ordering and simultaneous Vector traffic remain plausible causes.

Evidence rejects the proposed dominant buffer-credit-wait bottleneck in this
probe. It instead localizes the gap toward memory-feed efficiency/locality.
It does NOT isolate an exact causal share between paired gate/up weight order,
Cube/Vector competition and other backend policies. Independent counters overlap;
scalar stalls and pipeline times cannot be summed into a critical path. A future
change should target that concrete uncertainty, not expand ring size again.

## Deliverables / exact source

All artifacts under `/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/`:
`opprof-v3-source`, `opprof-v3-local7/measurements`, `opprof-v3-local7/export`.
Export includes per-core-report.html, cube-memory-counters.svg and a59KiB zip
of all32 original CSVs. SVG bars are active-duration counters, NOT timelines.
`PMU-SUMMARY.json` retains minimum/median/maximum with row counts and exact
native kernel identity. `summarize_pmu.py <measurements> <output>` regenerates
small reports without reading multi-megabyte raw dumps. Local7 release receipt
exists; no other tasks touched. Paired source03ea4dc, full-buffer sourceb163149;
exact build binaries remain build-v7-pair and build-v8-full with identity receipts.

## Recover actual native tiling, not just adjacent Mc2 source

The native dump retained a288-byte input_tiling.bin even though instruction
TimelineDetail failed. Decoding the installed actual TCubeTiling prefix (50
int32 fields) and first5 L2 fields gives NATIVE-TILING.json. Shape/core count
and exact panel coverage agree with the profile; trailing runtime/padding bytes
are deliberately not interpreted.

Actual native cube: M128/N256/K64, stepKa/Kb4, depthA1/B1=8, dbL0A/B2,
dbL0C1. L2 panels:4 M panels *4 N panels; each8 M blocks *27 N blocks,
thus1024 rows x6912 output channels. calOrder0 selects diagonal within-panel
assignment. The real installed source is ops_nn/ascendc/mat_mul_v3, not the
previously inspected ops_transformer/ascendc/3rd/Mc2 variant. Its base kernel
traverses L2 panels in alternating N direction across M panels and calls
UpdateBasicIndex for calOrder0. The native65536 key records the selected variant;
no assumption of a K-shift optimization is needed here.

A native panel's weight set6912*5120*2 bytes=67.5MiB, reused over1024rows. Our
paired row-major traversal sweeps all13824 paired channels (both halves total
270MiB weights) over roughly one256-row block before repeating for more rows.
Logical GM->L1 bytes can be equal while a different fraction hits L2. This is
concrete working-set structure consistent with the PMU gap, not a timed causal
ablation. Simply saying distant addresses are slow was imprecise; reuse distance
and active set matter, not address distance alone.

Next smallest candidate: retain tested256x128 paired GEMMs and two-slot local
micro-pipeline, but schedule panels of1024rows x3456 paired channels. Both gate
and up together cover6912 channels=67.5MiB weights, matching native panel size.
There are4 row panels *4 paired-channel panels, each4 M blocks *27 paired N
blocks. Apply diagonal assignment within this panel and snake panel traversal;
do not repeat the old slab-local swizzle-only test. No new tensor transpose or
quantization is required. This proposal has not been implemented or timed.
