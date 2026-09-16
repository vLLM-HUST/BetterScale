# Small-row, many-expert GEMM: checkpoint and scheduling evidence

2026-09-16. Service checkpoint: `c208d5b`, tagged and pushed as
`checkpoint/expert-service-work-conserving-20260916`. See `PERSISTENT.md`
for correctness, protocol limits and conditional performance, not a claim of
DFC parity. This investigation does not change the released worker.

## Same-card cache and useful-row controls

Run `runs/actual-gmm-control-20260916T064520Z` completed on device 1.
`gmm_schedule_probe.py` uses the already qualified actual-count CATLASS adapter,
BF16 whole NZ weights, H=2048, intermediate=768, capacity=512 routed rows.
Independent random expert weights and per-expert chain oracle pass for both
layers (rtol=0.02, atol=2e-5). Each measurement has 8 warmups and five trials of
32 graph replays. Chain includes native capacity-512 SwiGLU; no IPC, packing,
mailbox polling or serving scheduling is included. Full trials are preserved in
`gmm-scheduling-result.json`.

Times below are median microseconds, same-layer / alternating two disjoint
layer weight catalogs:

| Experts | Rows/expert | Up | Down | Up + SwiGLU + down |
|---|---|---|---|---|
| 16 | 2 | 42.88 / 45.54 | 28.87 / 28.54 | 79.54 / 141.47 |
| 64 | 2 | 288.84 / 289.90 | 60.88 / 149.32 | 455.48 / 456.33 |
| 64 | 4 | 285.96 / 290.52 | 59.70 / 148.27 | 454.88 / 456.52 |
| 64 | 8 | 291.12 / 291.35 | 60.36 / 150.40 | 461.77 / 459.83 |

**Observed:** isolated repeated down is not representative of the whole chain.
Changing only the weight-reuse sequence takes broad down from about 61 to 149 us,
near the persistent service's measured down time. **Inference:** cache residency
explains much of the apparent adapter/service gap; this is not proof of an HBM
roofline. Four times the useful rows costs almost the same broad-expert math:
work-conserving cross-source batching remains valuable without intentional waiting.

## Existing scheduling is not naive expert-per-core

Installed CATLASS `gemm/kernel/grouped_matmul_slice_m.hpp` statically stripes
M/N tiles over cores and rotates the starting core across groups. Default
L1(128,256,256) gives 384 up tiles and 512 down tiles for 64 small-row experts:
16 and 21/22 tiles per AIC respectively. Preload/L1 and L0 buffering already
exist, as does K-start staggering. Actual M is rounded to hardware alignment;
a nominal M=128 tile does not imply computing all 128 rows for a two-row group.
Earlier smaller-M and wider-N probes did not deliver useful broad-route gains.

In the sequential homogeneous instrumented run
`runs/remote-dfc-control-20260916T063512Z`, median per-wave spread between
fastest and slowest AIC routine durations is about 6.5–9.1 us (24 cores), not
hundreds of microseconds. This supports deprioritizing dynamic work stealing for
this balanced case, not a conclusion about every skewed route distribution.
For interleaved slots, classify phases by command generation/event joins; odd/even
command numbers do NOT reliably distinguish up/down.

## DFC has two distinct pipeline levels

Pinned source: `upstream/vllm-ascend/csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/`.

* `dispatch_ffn_combine_bf16.h` uses preload and L1/L0 buffering, and sets
  `epilogueGranularity = expertPerRank - 2`.
* `dispatch_ffn_combine_bf16_kernel.hpp::GMM1` publishes a cross-core completion
  after the first expert segment, then completes the remaining experts.
* AIV `DispatchAndCombine` consumes that segment for SwiGLU, publishes readiness,
  and handles the second segment separately. GMM2 waits for the required segments.
* The AIC entry still calls GMM1 before GMM2. This is NOT simultaneous up and down
  GEMMs on the same cube cores. Source establishes the mechanism; it does not
  quantify the overlap achieved by our separately built lab DFC provider.

Our persistent prototype has two request slots, but within a slot currently
waits for all up, then all SwiGLU, then all down. Request double buffering is
not equivalent to DFC's intra-wave segmented AIC/AIV pipeline.

## Next bounded experiment

Prefer a two-segment up-to-SwiGLU publication prototype over a new dynamic GEMM
scheduler. Reuse CATLASS and device actual-row counts; publish only after all
producer cores have completed the segment. Preserve generation and slot ownership,
including zero-hit groups and skew. Measure same-work whole-chain latency and
actual overlap, not sums of isolated warm GEMMs. Segment barriers and AIV
contention may erase gains, so retain the current path as the control.

Reproduce the cache/rows probe on an idle single card:

```bash
ACTUAL_GMM_PROBE=gmm_schedule_probe.py \
  bash prototypes/attention-client/device-service/run_actual_gmm.sh 1
```
