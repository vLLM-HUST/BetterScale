# Independent-source coalescing and live-row GEMM

This is an opt-in **bounded prototype**, not a published worker change or a claim
of DFC parity. Two clients independently publish frames; two expert servers pull
them. A server combines ready rows with the same (layer, expert), rather than
running a GEMM per source. Different layers remain distinct groups.

## Scheduling and ownership

`DEVICE_SERVICE_COALESCE_POLLS=16` allows sixteen additional device poll rounds
after at least one source is ready. It is **not sixteen microseconds**. Consume
both if available, otherwise consume the available source when the budget expires.
Completed sources are excluded. No per-wave host rendezvous is required.
The old forced-pair diagnostic remains separate and mutually exclusive.

Server capture completes before server_ready; clients then acknowledge readiness,
and servers enqueue their finite service graph. The burst fixture may prequeue
both client graphs at this initial rendezvous. That tests a common burst, not
arbitrary online arrival synchronization. The deliberately late-source fixture
tests the opposite case. Each source frame still waits for both exact-generation
server completions before reuse.

## Actual rows, not a native-GMM shape loophole

`DEVICE_SERVICE_ACTUAL_COUNTS=1` enables whole NZ weights and the thin
`actual_gmm.cpp` adapter. Device-produced cumulative group ends determine live
rows. Empty catalogs return; leading/trailing empty groups are trimmed without
copying weights. GEMM does not compute synthetic padding or write inactive rows.
This explicitly defined capacity/live contract does **not** pass a short final
group end to native npu_grouped_matmul, whose documented contract requires full M.

Scope: BF16, 64/128 groups, capacity at most512; matrices2048×1536 and768×2048.
Pack writes live rows only. SwiGLU still traverses capacity512; we have not
eliminated all padded vector work. Buffers, weights, group arrays and loaded
binaries outlive graph replay.

The adapter reuses external CANN CATLASS grouped-M scheduling and matrix tiles.
`ACTUAL_GMM_DFC=1` additionally selects pinned donor DFC's FIXPIPE tile through
an argument adapter; it needs the newer external CATLASS include tree used by
the lab provider. No upstream matrix implementation is copied into this repo.
Those headers retain their CANN Open Software license. This is Ascend-only,
and not a self-contained portable package.

## Evidence, 2026-09-16

Paths below are relative to /workspace/strengthen-dsv4/runs.

- `actual-gmm-control-20260916T051430Z`:14 leaf cases, including empty,
  one-expert512-row skew, layer switch, input preservation and inactive-tail/
  outer canaries. Same graph consumes changing device counts.
- `attention-device-joint-20260916T051919Z`:12 post-preparation dummy model
  forward checks across two clients; output and KV bitwise exact. Two layers,
  full Qwen dimensions; not real-weight serving.
- `remote-dfc-control-20260916T051539Z`:48 route-control outputs pass
  rtol.02/atol2e-5; worst relative L2 0.000162. Not generally bitwise identical.
- `remote-dfc-control-20260916T052017Z`:common prequeued burst,24/24 paired
  cycles on each server,48 outputs pass; worst relative L2 0.000293.
- `remote-dfc-control-20260916T052128Z`:source1 starts20ms late; source0
  completes its24 jobs in16.265ms. Ready work is not held for the absent source.
- Counterexample `remote-dfc-control-20260916T051837Z`:without common startup,
  the same closed-loop sources produced zero paired cycles. Coalescing support
  does not guarantee arrivals overlap.

- Final FIXPIPE burst `remote-dfc-control-20260916T053439Z`:48 outputs
  passed,24 paired cycles/server. Device receipts verify mode flags13 and the
  bounded wait count, rejecting stale queue binaries. Immutable matrix binaries
  are stored in its actual-build directory.

- Final standard late-source gate `remote-dfc-control-20260916T053608Z`:
  all48 outputs passed; source0 completed24 jobs in16.012ms before the20ms
  delayed source. Both servers verified flags13 and16 bounded poll rounds.
  See [compact receipts](actual-counts-result.json).

### Performance is not yet DFC parity

In the profiled single-source broad32 control, standard CATLASS live-row GEMMs
sum to419–456us/server versus398–403us for the earlier native-NZ control.
Hot32 improves to34–38us of GEMM. Broad, tiny expert groups remain expensive.
Common-burst paired cycles process twice the live work, but this is not an
equal-resource end-to-end throughput comparison against two-card DFC.

The FIXPIPE leaf variant (`actual-gmm-control-20260916T053352Z`) also passed
all14 cases, but did not solve this: broad gate/up remains about279us; down is
about77us versus70us for the standard tile. Merely selecting DFC's tile is not
the same as reproducing its fused scheduling and communication pipeline.

Profiled common-start episode wall time includes profiler/startup rendezvous;
do not publish that value as steady service throughput. Timelines use provider
clock alignment, not a new cross-device clock calibration.

## Reproduction

Build queue with `OUTPUT_DIR=runs/attention-device-actual-build bash
prototypes/attention-client/device-service/build.sh`; build standard matrix
adapter with `bash prototypes/attention-client/device-service/build_actual_gmm.sh`.
Then set:

```bash
DEVICE_SERVICE_PARALLEL=1
DEVICE_SERVICE_ACTUAL_COUNTS=1
DEVICE_SERVICE_COALESCE_POLLS=16
DEVICE_SERVICE_BURST=1
DEVICE_SERVICE_SOURCE_BUILD=$PWD/runs/attention-device-actual-build
ACTUAL_GMM_BUILD=$PWD/runs/attention-actual-gmm-build
```

Export those variables before run_remote_dfc.sh with four idle device IDs.
Add DEVICE_SERVICE_SOURCE1_DELAY_MS=20 for the delayed-source gate.
Set DEVICE_SERVICE_PROFILE=1 for msprof. New capsules freeze matrix binaries
and record their digests; earlier runs above predate that snapshot addition.

The finite48-cycle graph, at most24 jobs/source, bounded queue polling, host
attention continuation and no production cancellation remain explicit limits.
Native GMM stays the default until broad-route performance is improved.

### Optional DFC tile build

```bash
CATLASS_INCLUDE=/workspace/my-ascend-workspace/stateharbor/native/ascend/csrc/third_party/catlass/include \
ACTUAL_GMM_DFC=1 OUTPUT_DIR=$PWD/runs/attention-actual-gmm-dfc-build \
bash prototypes/attention-client/device-service/build_actual_gmm.sh
```

The default standard tile uses installed CANN headers. The DFC variant additionally
uses pinned upstream/vllm-ascend headers, including its custom vector-layout copy
specialization; do not mix arbitrary CATLASS versions. It remains an experimental
comparison, not the selected faster implementation.

Final FIXPIPE common-burst timeline: `runs/remote-dfc-control-20260916T053439Z/analysis/attention2-expert2-provider-clock.json.gz`.
Per-server steady23-cycle medians (source-wait excluded):
- server0: GEMMs 470.00us, pack-to-DONE 567.18us for two sources.
- server1: GEMMs 423.54us, pack-to-DONE 519.16us for two sources.
