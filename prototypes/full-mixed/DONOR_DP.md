# Native donor DP+EP baseline

This study uses the release pins, not our FULL-mixed, draft-graph or cross-step
optimizations. `donor_dp.py` follows the pinned native offline DP example: one
LLM client per DP rank, per-rank request shards, native engine scheduling and EP.
`donor_dp_worker.py` observes dispatch and device-event windows. It includes the
existing K5/TP8 **LCM capture-size startup repair**; no installed package or pinned
source is edited. Dummy-only checkpoint-layout repair is not a real-model change.

## September 13, hw3: DP8TP1 versus TP8DP1

Both use eight 910B2/HCCS cards, real DeepSeek-V4-Flash-0731-w8a8, EP8, eager
DSpark K5, target FULL_DECODE_ONLY, shared-expert overlap, max model length16384,
8GiB KV per card, and prefix caching disabled. Decode has16 global requests;
prefill has8. Greedy synthetic-token cohorts ignore EOS and generate a fixed
output length. This is a path/throughput study, **not agent-workload or quality
acceptance**. Native scheduling and speculative acceptance remain in the timing.

| Configuration | DP8TP1 | TP8DP1 |
|---|---:|---:|
| Max active requests per engine | 2 | 16 |
| Prefill token budget per engine | 1024 | 8192 |
| Aggregate budget | 8192 | 8192 |
| Target capture buckets | 6,12 | 24,48,96 |
| Attention builder | DSA | DSACP |
| FlashComm1 | Off (required for TP1) | On |
| Model-loading allocation (GiB) | 49.49 | 41.21 |

Completed capsules: hw3 `dp8tp1-052-real` and `tp8dp1-053-real` under
`/workspace/my-ascend-workspace/runs/strengthen-dsv4-full-mixed/`.
Local receipts: `runs/hw3-dp8-052/` and `runs/hw3-tp8-053/`.
Both exited0 and released all eight cards. Eight relevant installed source files
were byte-compared with the pins; all matched. This is not a blanket runtime
identity claim. DP8 four-layer dummy run051 passed before real weights.

### Unprofiled measurements, two rounds

| Measurement | DP8TP1 | TP8DP1 |
|---|---:|---:|
| Full-occupancy decode target forward | 44.06 / 44.37 ms | 52.85 / 54.97 ms |
| Full-occupancy inter-target cycle | 60.79 / 59.90 ms | 67.79 / 69.99 ms |
| 16×(128 input,128 output): completion | 4.657 / 5.176 s | 5.013 / 4.792 s |
| Same cohort: output throughput | 439.7 / 395.7 token/s | 408.5 / 427.4 token/s |
| 8×(4096 input,16 output): completion | 3.668 / 3.621 s | 3.357 / 3.587 s |
| (8192 +7×256 input),16 output each | 5.159 / 5.123 s | 2.136 / 2.035 s |

Full-occupancy rows use the first10 consecutive FULL forwards with16 actual
requests and96 total target query rows, checked on every rank; report the median
of eight within-rank medians. Initial prefill and drain are excluded. This is a
matched **query-count** window, not identical generated histories. Device-event
forward spans can include queued waits; cycles include host, sampling and draft.

The DP cycle is10–14% shorter in these windows, but cohort output throughput has
**no stable winner** in two rounds. Do not convert fixed-query speed into an
acceptance-independent output-token speedup. The skew penalty is measured with a
1K local budget: DP needs eight chunks for the long request, while TP can use its
whole8K budget. It does not establish DP's best tuned prefill performance.

### Capacity: distinguish ownership from the reported bound

DP has eight independent request/KV pools; TP's DSACP state is replicated across
its eight ranks. Native cache reporting at these configs gives178080 equivalent
tokens per DP engine versus27066 for the single TP engine. These are group-aware
**max-length concurrency equivalents**, not flat token-slot capacities, nor a
measured1.42M-token residency test. Do not advertise their52.6× aggregate ratio as
pure DP savings: the differing prefill budgets change sliding-window per-request
admission memory. Active seats are also explicitly limited to16 globally here.
DP's model-loading allocation costs8.28GiB more per card; it is not free capacity.

### Mechanisms to inspect in the profiles

- `model_runner_v1._sync_metadata_across_dp` exchanges CPU token counts and
  graph mode every target step in this non-disaggregated MoE configuration.
  `_post_process_cudagraph_mode` takes the minimum: one NONE rank makes all NONE.
- A2/EP8 selects ALLGATHER, not MC2 (`_select_a2_moe_comm_method`). Without
  FlashComm1, `_prepare_with_dp_group` pads hidden/router tensors to the maximum
  DP token count and all-gathers; finalize reduces/scatters back to owners.
  Thus attention can remain local while MoE communication retains skew costs.
- There is also a communication-precision asymmetry: the no-FlashComm1 DP
  prepare route gathers BF16 hidden states before quantization, while the
  W8A8 FlashComm1 EP route quantizes before gathering INT8 plus scales. Native
  rank0 decode profiles corroborate it:49152-element AllGather entries are
  BF16-only in DP, whereas TP has INT8 entries as well as its other BF16
  collectives. This is a candidate optimization, **not a measured speedup**;
  do not compare counts across profiles with different wave counts.
- The native offline DP engine checks global completion every32 iterations.
  Completed ranks execute dummy passes for EP; traces include the resulting
  post-request drain, often to32/64 calls. Do not count those as real decode
  tokens or interpret the whole-window median as occupied service performance.
- Profiles are separate from timed cohorts and visibly slower under tracing.
  Use them for mechanism, not the throughput table above.

## Reuse

Use the existing lease/admission launcher, fresh capsule and unchanged pins.
`probe.py --donor-dp 8 --tp 1 --spec --budget 1024 --kv-gib 8 --real
--profile-after` selects the DP runner before importing the optimization extension.
The TP control is `--donor-dp 1 --tp 8 --budget 8192` with the other flags unchanged.
Keep all eight cards visible: Ascend maps local devices from native DP rank×TP.
Do not also hide one device per DP client, or pass `LLM(data_parallel_size=8)`
through the unsupported single-process offline interface. TP1 must turn off
FlashComm1. Keep all clients alive until other ranks finish their EP work.

After exit0, copy the small receipts and run
`python3 prototypes/full-mixed/donor_dp_report.py CAPSULE/engine`.
For profiles, parse with native `torch_npu.profiler.profiler.analyse` offline,
then use `profile_tools` on each window separately. Preserve provider databases
and rank/device maps; link only compressed exported timelines. The affine clock
fit is display-only candidate alignment with the existing50us holdout gate.
