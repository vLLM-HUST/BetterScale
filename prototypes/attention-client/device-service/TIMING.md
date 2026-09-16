# Local versus separated expert: current prototype is not faster

2026-09-16, same local910B2. The native oracle's eager timeline has1.5–1.8ms
expert envelopes, but much of that is host launch delay. Comparing that directly
to a remote captured graph would manufacture a speedup. A fresh one-card native
FULL MLP graph control and an unprofiled four-card remote run give this instead:

| Actual token rows | Native local FULL MLP | Remote client FULL graph |
| --- | ---: | ---: |
| 1 | 119.5us | 211.1us |
| 16 | 136.8us | 665.4us |
| 32 | 155.7us | 1089.8us |

Both cover gate/top-k through weighted expert output. Native uses TP1 local
experts, two layers with full Qwen30B-A3B dimensions and dummy BF16 weights. Its
10 trials/shape each average64 prequeued replays. Remote uses Attention2+Expert2;
external events bracket individual client graphs, excluding input staging copy,
with16 sealed decode observations and4 per prefill shape. No profiler is active
in either timing run. All local graph outputs and joint forward/KV shadows remain
exact. See [machine-readable receipt](timing-result.json).

This is a stage-cost diagnostic, NOT equal-resource throughput or a precisely
matched end-to-end speedup ratio. The local control reuses one input; the remote
fixture uses actual successive layer inputs and retains native oracle gaps. It
has not tested another request advancing attention while this request awaits its
experts. The present host scheduler admits only one lane per client. Attention
GEMMs have moved off the card, but that fact alone is not faster request progress.

## What the profile identifies

The earlier same-host profiled run (`20260916T032439Z`) lets us separate compute
from return formatting. Device traces associate each server cycle with source
generations; the following use only single-source sealed waves. Medians per server:

| Rows | Two server GMMs | neural_complete |
| --- | ---: | ---: |
| 1 | 65.2–66.1us | 22.35–22.44us |
| 16 | 62.9–65.7us | 225.6–226.0us |
| 32 | 62.9–66.2us | 441.8–443.9us |

These are profiled operator spans, not unprofiled timing-row components. The
server's near-flat GMM time partly reflects its fixed512-route padding: it does
not prove efficient useful-token batching. `neural_prepare` includes waiting for
requests and cannot be charged entirely to packing.

The output path has an observable avoidable cost: one AIV core serially scatters
and synchronizes each top-k row. Both servers return the entire `[N,K,H]` shape,
writing zero for nonowned slots; zero fill uses scalar UB stores. AtN32,K8,H2048,
each server exposes1MiB BF16 output and the client pulls2MiB, even though half the
slots per server are unowned. The input pack likewise rereads a hidden row for its
expert routes. This is correctness-first transport/formatting, not a mature MoE
communication kernel or evidence of a hardware bandwidth ceiling.

A useful next optimization target is the output protocol and parallel pack/scatter,
not replacement of native GMM. Server-side weighted partial token reduction could
reduce returned volume, but changes summation order and needs its own numerical
contract. Dense zeros can also be avoided without that semantic change. Actual
multi-request overlap/batching must then be measured before making service-gain
claims. No such optimization is silently included in this result.

## Reproduce

```
bash prototypes/attention-client/device-service/run_local_expert.sh 0
DEVICE_SERVICE_TIMING=1 bash prototypes/attention-client/device-service/run_joint.sh 0,1,2,3
python3 prototypes/attention-client/device-service/summarize_timing.py <local-run> <remote-run>
python3 prototypes/attention-client/device-service/compare_expert_profile.py <profile-run>
```

Timing is opt-in and does not alter default protocol or publication. Lab sources
and selected-device admission are frozen in each run capsule. Cards were released
after both runs. The external service binary and published worker are unchanged.
