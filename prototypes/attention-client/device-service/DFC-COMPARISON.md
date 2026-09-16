# EP2 fused DFC versus the separated expert stage

## Measured scope

Fresh local910B2 controls on2026-09-16. Both use BF16 H2048, M768, E128,
top-k8, two sources with1/16/32 tokens each, max receive capacity512, and FULL
captured execution without profiler. Both consume exactly the same synthetic
hidden rows, expert-dependent weights, expert IDs, and uniform route weights.
Routes are prepared in advance; **gate/top-k is excluded from both timings**.
This differs from the earlier whole-MLP table in TIMING.md.

| Routing | Tokens per source | EP2 DFC | Remote client stage | Remote / DFC |
| --- | ---: | ---: | ---: | ---: |
| balanced sweep | 1 | 279us | 189us | 0.68 |
| balanced sweep | 16 | 416us | 609us | 1.46 |
| balanced sweep | 32 | 423us | 643us | 1.52 |
| hot8 | 1 | 272us | 162us | 0.60 |
| hot8 | 16 | 290us | 196us | 0.68 |
| hot8 | 32 | 318us | 237us | 0.74 |

Balanced IDs are `(arange(N*8).reshape(N,8)+source*8)%128`: at1 token the two
sources collectively touch16 experts, all on owner0; at16/32 they cover all128.
Hot8 IDs are `[0,1,2,3,64,65,66,67]` on every token: four experts per server.
These are controlled routing endpoints, not samples of trained Qwen routing.
Weights use a shared random base with an expert-parity .5/1 multiplier. The
independent oracle detects owner/index mistakes while avoiding a full weight
all-gather. It is not a representative trained-weight quality evaluation.

The conclusion is **conditional**, not universal parity: the current separated
implementation approaches/exceeds this DFC build's latency on concentrated routes,
but is46–52% slower on the16/32-token broad-expert cases. Changing route spread
changes both expert packing and GEMM work; these measurements alone do not prove
which particular kernel explains the gap. Do not attribute it to PCIe/HCCS or
network bandwidth without the corresponding profile.

The subsequent [route-spread investigation](ROUTE-SPREAD.md) attributes the broad
penalty primarily to native GMM, records the NZ control and startup-readiness fix,
and preserves the remaining batching/fusion gaps.

## Fairness boundaries

- DFC has two cards, each both source and expert owner; remote has two attention
  client cards plus two expert servers. This is NOT equal-resource throughput.
- DFC processes both sources in synchronous EP waves; remote may choose one or
  both independently. Its reported time is each client's submit/result graph,
  not wall time to complete a synchronized two-source cohort.
- DFC uses five trials of64 prequeued graph replays, retaining the slower rank per
  trial. Remote uses six warm individual observations/case (first of four per
  client discarded). Balanced1 remote ranges175–325us: do not overstate its median.
- Device sets differ: DFC1,3 and remote4–7, same host. Selected cards passed
  admission and foreign-owner checks; unrelated devices may still be active.
- Receive capacity is512 in both, but remote's128 layer/expert groups include a
  second inactive model layer and padding on the final group. DFC uses one64-expert
  shard per rank. This preserves the current candidate rather than hiding its cost.

See [machine-readable result](dfc-comparison.json) for exact run paths/ranges.
Remote48 outputs are bitwise equal to the independent BF16 oracle. DFC passes
rtol.02/atol2e-5 and normalized L2<.01; observed worst relativeL2 is below.0003.
Different fused arithmetic is reported, not silently called bitwise identical.

## The DFC binary is a lab A2 port, not the pinned donor package

The pinned donor910B build omits BF16 DFC (`csrc/build_aclnn.sh` lists it under
ascend910_93). Initial run042609Z fails with missing
`aclnnDispatchFFNCombineBF16[GetWorkspaceSize]`.

An existing LiveInfer/stateharbor A2 build contains the API and910B kernels.
Its API takes **an extra xActiveMask argument** before group. Pairing that library
with the pinned donor's older BF16 Torch binding shifted all later arguments and
caused host SIGSEGV in run042836Z. This is an ABI mismatch, not a device performance
result. The qualified run043010Z loads the A2 build's matching Torch extension AND
OPP directory in its isolated process. Installed donor packages are untouched.

`run_dfc.sh` records the OPP path; its exact current lab provider is:
`/workspace/my-ascend-workspace/stateharbor/build/lib.linux-aarch64-cpython-312/livemodule/arch/ascend/_native/`.
Do not call this the performance of the stock vllm-ascend release. Requalify the
API/header pair if rebuilding that mutable lab artifact.

Remote042813Z completed its numerical checks but the supervisor rejected a
late unknown-owner row; its timings are excluded. The subsequent fresh
admitted042852Z run exited0 and supplies the table. Queued042724Z was cancelled
before launch to switch from occupied2,3 to idle1,3; no watcher was left behind.

## Reproduce

```
bash prototypes/attention-client/device-service/run_dfc.sh 1,3
DEVICE_SERVICE_PARALLEL=1 DEVICE_SERVICE_TIMING=1 \
  bash prototypes/attention-client/device-service/run_remote_dfc.sh 4,5,6,7
python3 prototypes/attention-client/device-service/compare_dfc.py <dfc-run> <remote-run>
```

The remote control reuses the production-path prototype's existing kernels,
DeviceExperts lifecycle and native GMM; only the bank's gate is replaced with
fixed routing to match DFC's input boundary. It still has the same bounded24-job
per-source protocol. No serving default is changed.

### Weight address scope clarification (2026-09-16)

The remote burst reuses two separately allocated layer weight ranges, with equal
dummy values, alternating layer0/1; the DFC leaf replays one range. This is not
fresh weight generation per request. It is an unmatched address/cache condition,
not a demonstrated explanation for the whole latency gap. See EARLY-RETURN.md.
