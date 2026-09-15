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
