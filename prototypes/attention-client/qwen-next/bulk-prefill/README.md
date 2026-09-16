# Large-prefill cross-source batching probe

The 32-row serving fixture is **not** a suitable capacity target for a prefill
expert server. This isolated experiment widens input frames and matched server
workspace to 1024 rows/source, without changing the released 32-row serving ABI.

## What ran

Three physical 910B devices: two publishers and **one** expert owner. Qwen3-Next
geometry H=2048, intermediate=512, top-k=10, 512 experts; the server holds 128
experts for each of two layers. Synthetic activations and weights, actual NZ
GEMM and cross-device input reads. This is not the full A2/E4 model.
Both sources publish before admission. Same-layer tasks become one wave;
different-layer tasks remain two. No intentional batching delay was added.
For equal-work comparisons, solo 2N and dual N+N use identical token values and
routing multisets; only source partition changes. IDs broadly cover experts.

Run `bulk-prefill-20260916T163223Z` passed 102 cases, all three processes exited
normally. Six repetitions alternate case order; medians exclude the first two.
Selected first/last owned routes were checked, maximum relative L2 9.28e-6.
This is **not** all-output, full-state or model-quality validation.

| Same-layer rows/source | Total input rows | Server span | us/input token |
|---:|---:|---:|---:|
| 128 | 256 | 965.81 us | 3.773 |
| 256 | 512 | 1121.81 us | 2.191 |
| 512 | 1024 | 1581.65 us | 1.545 |
| 1024 | 2048 | 2549.62 us | 1.245 |

Solo 1024: 1568.93 us; dual 512: 1581.65 us, within 0.82%. Bulk cross-source
batching retained single-source efficiency in this bounded case. Dual 1024
on different layers: 3016.16 us; same-layer span is 15.5% shorter. This comparison
includes intended same-layer weight reuse, not solely scheduling savings.

The span is the first coordinator command start through last completion,
including pull/pack/GEMM/activation/return and internal gaps. It excludes client
reduction, attention, host rendezvous and admission before the first command.
The denominator counts input tokens although only one quarter of the model's
experts are served. **Do not label this complete MoE or serving throughput.**

Large batches are not free: from dual 128 to dual 1024, median fetch envelope
77→431 us, pack 56→297 us, return 119→418 us; up 477→727 us, down 214→282 us.
Overlapping phase envelopes cannot be added as a latency decomposition or read
as per-core utilization. Data handling deserves separate examination next.

## Workspace and lifecycle

The frozen variant resizes input, route maps, both staging slots and output
buffers together. At 1024/source, source frames are 6 MiB each, raw output areas
42 MiB each. Hidden payload starts at int32 word 11264, beyond the route IDs.
Coordinator retains two slots' IDs in int16 UB storage, validates full int32 IDs
before narrowing, and uses a 64 KiB transfer buffer. Each vector worker retains
only its own strided route-map entries. Native scalar stack size stays unchanged.
This probe uses continuous, fine-pack, early-down/return and route-pull mode;
resident/move-quantum alternatives are not enabled or qualified by it.

A source frame is not reused until server completion, source-specific EOF and
server stop/unmap acknowledgements. IPC capabilities stay in process memory.
This is one large ready frame per source, **not** a dynamic arbitrary backlog
queue. Source admission capacity, coalesced batch size and internal pipeline
workspace should remain distinct design budgets in a future serving integration.

## Reproduce

From repository root (original experiment base `23845df`):

```bash
python3 prototypes/attention-client/qwen-next/bulk-prefill/build.py \
  runs/bulk-prefill-1024-build --rows 1024
bash prototypes/attention-client/qwen-next/bulk-prefill/run.sh 1,3,4
python3 prototypes/attention-client/qwen-next/bulk-prefill/report.py \
  runs/<printed-capsule>/results/result.json runs/<printed-capsule>/analysis
```

Use three available devices; the launcher retains bounded subset admission and
foreign-occupancy monitoring. Build and source are frozen together in each run.
`result.json` here is the bounded numeric receipt. Full samples and independent-
origin compressed coordinator phase timeline remain under the run's `analysis/`.
These are not cross-device clock-aligned msprof traces.

### Important failed-probe lesson

The admission helper snapshots sibling Python modules and prepends that directory
to `PYTHONPATH`. Calling it beside the old runtime silently paired a widened
binary with **32-row Python allocations**, causing memory corruption/507011.
Those failed runs do not measure the algorithm. This launcher copies admission
into its own directory and workers assert the imported runtime path before NPU
initialization. Do not mix binary, allocation geometry and client payload offsets.
