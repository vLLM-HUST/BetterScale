# Why broad expert routing is slower

## The measured bottleneck is GEMM, not remote payload packing

The 2026-09-16 four-card profile `runs/remote-dfc-control-20260916T043335Z`
uses the exact synthetic route/weight control described in `DFC-COMPARISON.md`.
`analyze_route_spread.py` joins each server cycle to its device-recorded source
**generation**, then to the client's workload record. It does not infer batch
membership from adjacent timestamps. Every measured warm cycle selected one
source, not two-source coalescing.

At32 input rows per source, medians per expert server are:

| Stage | Four local experts (hot8 globally) | 64 local experts (128 globally) |
| --- | ---: | ---: |
| Two native GroupedMatmul tasks, summed | about64us | about464us |
| Remote input pack | about35us | about35us |
| Output scatter | about15us | about13us |

The broad16 case has approximately the same462us GMM cost as broad32. Thus
roughly400us of the route-spread penalty is in the two GMMs, not a newly expensive
pack/scatter. `neural_prepare` includes device polling for a new source; do not
charge its entire task duration to routing computation. Profiler durations are
attribution evidence, not replacements for the unprofiled stage-latency table.

Each BF16 expert has6MiB of gate/up weights and3MiB of down weights. A broad
single-source32-row cycle gives each local expert only2 routed rows, versus32
rows/expert in hot8. The logical live weight working set rises from36 to576MiB
per server, excluding the fixed final padding expert. These are weight sizes,
not measured HBM traffic: caches and kernel tiling determine actual transfers.

## Layout is part of the gap, not a complete explanation

The server originally supplied ND weights to native grouped matmul. The fused
DFC control supplies NZ. An opt-in `DEVICE_SERVICE_WEIGHT_FORMAT=NZ` converts
server weights once, before capture; raw IPC inputs and GMM outputs stay ND.
The qualified four-card run `runs/remote-dfc-control-20260916T044052Z` passes
all48 outputs bitwise against the same oracle. Six warm observations per case:

| Routing / rows per source | Earlier ND client stage | NZ client stage |
| --- | ---: | ---: |
| broad /16 | 608.6us | 542.1us |
| broad /32 | 643.3us | 586.6us |
| hot8 /16 | 195.6us | 194.8us |
| hot8 /32 | 236.7us | 236.6us |

The subsequent NZ profile `runs/remote-dfc-control-20260916T044206Z`
confirms the location of the improvement: broad32 GMM pairs fall to398–402us
per server (from roughly464us); pack stays34–35us and scatter about13us.
Hot8 GMM pairs remain65–67us. The48-cycle generation join succeeds on both
servers despite the profiler stop warning; this export covers the finite episode.

This is about9–11% less broad-case latency, not DFC parity. The earlier DFC
numbers remain416.5/423.2us, subject to the topology/batching/provider boundaries
in the comparison note. NZ remains opt-in: this control does not qualify the
native full-attention two-layer model with NZ or make it universally faster.

The single-card `gmm_layout_probe.py` is a contract probe, not a surrogate for
remote-service performance. It checks both GMMs and SwiGLU with replay-time group
changes after all-padding capture, and tests ND/NZ/transposed-NZ storage. A local
32-replay block actually made NZ slower for the full pair, even though its first
GMM alone was faster. Do not extrapolate one matmul or that block to the service;
the four-card observation has its own producer/consumer, cache and execution
conditions. Raw results retain these contrary observations.

## Bootstrap readiness is separate from device work readiness

Rejected NZ run043443Z produced zeros/mismatches on the first client calls and
subsequent MTE errors during teardown. Weight-copy acknowledgement previously
released clients before server format conversion and48-cycle capture. A new
`server_ready` message is sent only after the prepared server replay is queued;
clients wait once before their first real submission. This avoids consuming
bounded device polling time on host startup. The changed-boundary run passes;
we did not weaken numerical checks or increase polling limits. The failed run
is not a timing result. `weights_loaded` still independently protects bootstrap
buffer ownership. This is startup control, not a per-wave host handshake.

## Remaining structural differences to test, not silently assume fixed

- Our observed service cycles did **not** batch both sources. DFC sees both
  sources in one synchronous EP wave. Broad32 then supplies four rather than
  two rows per expert and need not revisit all weights for a second independent
  source cycle. Measure batching under sustained arrivals before claiming the
  expert-server architecture retains DFC's weight reuse.
- Native standalone GMM/SwiGLU/GMM are separate graph tasks. DFC's GMM block
  scheduler rotates core starts across experts, disables weight L2 caching for
  small-M groups, and has AIV/AIC group readiness plus SwiGLU cross-core signals.
  See pinned `dispatch_ffn_combine_bf16_kernel.hpp`, GMM1/GMM2 and SwiGLU. Those
  source mechanisms are concrete candidates, not measured attribution of every
  remaining microsecond. Its global EP synchronization is not compatible with
  independent sources by simply copying the whole kernel.
- Our128 layer/expert groups and final-group padding are different from DFC's64
  local groups. Preserve this cost until a changed implementation is qualified.

## Inspect / reproduce

```
DEVICE_SERVICE_PARALLEL=1 DEVICE_SERVICE_TIMING=1 DEVICE_SERVICE_PROFILE=1 \
  bash prototypes/attention-client/device-service/run_remote_dfc.sh 4,5,6,7
# Use the pinned runtime with CANN initialized:
python prototypes/attention-client/device-service/profile_export.py <run>
python3 prototypes/attention-client/device-service/analyze_route_spread.py <run>
bash prototypes/attention-client/device-service/run_gmm_layout.sh 3
```

`<run>/analysis/route-spread-costs.json` preserves selected generations, source
counts and per-cycle costs. `<run>/analysis/attention2-expert2-provider-clock.json.gz`
is the four-device TraceLoom display. These are same-host provider timestamps,
not an independently calibrated inter-device clock fit. Never send the raw JSON
to Codex desktop. The unprofiled NZ run and single-card probes do not have timelines.
