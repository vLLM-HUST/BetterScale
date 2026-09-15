# Borrow native L2 scheduling without losing paired SwiGLU streaming

Native reference: NATIVE-TILING.json, recovered from the actual MatMulV3
profiler dump, plus installed ops_nn/ascendc/mat_mul_v3 base-block/kernel source.
Adaptation is not a claim of identical full native scheduling: keep paired
256x128 GEMMs, rather than native128x256, and keep existing pair-local credits.

NATIVE_PANEL=1 uses1024 rows x3456 gate/up paired channels per panel. Combined
weight footprint is67.5MiB, matching native1024x6912 ordinary output panels.
Each panel has4 row tiles x27 paired channel tiles. Global linear work ownership
remains core+round*24; remap that logical index through row-panel, channel-panel,
within-panel diagonal assignment, with channel-panel reversal on odd row panels.
No global rendezvous is introduced at panel boundaries; physical cores advance
independently through their own tasks. Last row panel uses its actual tile count.

Build: NATIVE_CM=256 NATIVE_CN=128 NATIVE_CK=64 NATIVE_V3=1 NATIVE_PAIR=1
NATIVE_PANEL=1, NATIVE_SLOTS=2, NATIVE_FULL_BUFFER=0. Control switches remain
available; no product defaults or installed packages changed. Working ring6MiB;
matched performance tests allocate the same larger envelope for both candidates.
No new transpose, quantization, arithmetic order within an individual dot product,
or request/state protocol. Existing CANN attribution remains.

## Verification

CPU enumerated all1..16 row-tile counts, proving a bijection over all paired
channel tiles and valid coordinates. Local7 dummy smoke128/257/769/1025 rows,
VC256/512, guards, immutable inputs, and three changed-input FULL replays all
passed.1025 exercises a partial second1024-row panel. Whole FFN numerical
comparison passed512/4096 with native down, including changed-input replay.

Six alternating ND/NZ/candidate trials, ten FULL replays each, local910B2:

| rows |old pair FFN|panel FFN|native ND(panel cohort)|native NZ(panel cohort)|
|---|---:|---:|---:|---:|
|512|0.951ms|0.885ms|1.041ms|0.885ms|
|4096|6.297ms|5.758ms|6.116ms|5.917ms|

4K reduces8.6% vs old pair,2.7% vs matched NZ,5.9% vs matched ND.512 ties NZ.
This is a bounded leaf FULL-graph result, not a model/serving improvement or a
universal winner across shapes. Full samples are in RESULTS.json. No additional
repeated test is justified merely to decorate this gain; extend only for a new
shape, integration, or identified variance risk.

## Independent per-core PMU corroboration

msprof op Default metrics, one matched kernel per process, warm-up0, kernel
replay.24 AIC and48 AIV counters retained. This is NOT an instruction timeline
or the whole-FFN throughput test. Prior TimelineDetail export had failed, so no
unchanged detail retry was requested.

| median counter |old pair|panel|native GEMM|
|---|---:|---:|---:|
|AIC Cube active|3329.37us|3304.51us|3295.59us|
|AIC MTE2 active|3963.23us|3238.44us|3156.84us|
|AIC L2 read hit|71.72%|94.14%|93.71%|
|AIC wait-id9|13.88us|13.20us|not applicable|

The changed traversal recovers cache hit/feed efficiency without materially
changing arithmetic-active time or buffer-return waits. This strongly supports
the locality interpretation of this ablation; counters are overlapping, not
additive latency buckets. Profile task duration old4018.76us, panel3521.54us,
native GEMM3657.94us; native excludes SwiGLU here. Use whole-FFN table above for
fair composite timing rather than mixing scopes.

## Artifacts

Root `/workspace/strengthen-dsv4/runs/qwen-native-ffn-20260915/`:
`build-v10-panel`, `panel-v10-source`, `panel-v10-local7`,
`opprof-panel-source`, `opprof-panel-local7`. Frozen source/binary identities,
admission/release receipts, raw samples, parsed counters retained. Both tasks
released local7. `opprof-panel-local7/export/` contains compact per-core HTML,
active-duration SVG (not a timeline), and original CSV zip.
