# Qwen MTP: direct-pool FULL graph prototype

This directory is **not** enabled by the packaged service launcher. The service
adapter is experimental. Bounded real-model TP2/MTP2/APC/mixed-FULL service
checks passed on2026-09-20; this is not product admission or SWE qualification.

## Archived exploration snapshot (2026-09-21)

This branch preserves the MTP/N+2 prototype and its supporting leaf changes,
followed by the mixed-GDN layout and solve/WY fusion experiments. It is not a
production release. Further fusion is deferred in
[BetterScale #1](https://github.com/vLLM-HUST/BetterScale/issues/1).

`MTP_GDN_WY_MODE=original` remains the default: the new fused solve/WY path
improves the measured small mixed waves but regresses long waves. Do not turn it
on universally. The current candidate's earlier layout fusion remains unchanged.
Read the [fusion evidence and controls](../../../.agents/skills/repo-knowledge/scenarios/qualify-qwen-mtp/solve-wy-fusion.md)
for the final matched envelope, numerical checks and compiler-workspace caveats.
The [MTP scenario](../../../.agents/skills/repo-knowledge/scenarios/qualify-qwen-mtp/GUIDE.md)
records the later service/SWE qualifications separately from the early results below.

## What is being transferred

LiveInference `05ac1541` already implements the important state contract:
previous acceptance includes the anchor (1..3); it selects one of three retained
GDN candidates, even if the next query has only one token. The correction/bonus
is the next anchor, not an already processed target state. Convolution history
uses a temporal offset instead of three matrix-state slots. Metadata bank reuse
and candidate-state lifetime are separate ownership problems.

## Original path → hook → experiment

- Native GDN builder produces speculative and non-speculative request metadata.
  `service_adapter.py` replaces that same builder leaf with fixed-capacity mixed
  metadata; it does not introduce a Worker.
- Existing BetterScale publication owns two host/DMA/device-read banks.
  `service_metadata.py::MTPFrame` stages host-authorized fields in one slab, then
  takes accepted-count feedback directly from the native runner's device tensor.
- Existing GDN `_forward_core` is the numerical hook. `mixed_core.py` combines
  chunk prefill and speculative recurrence while both write the same K-V state
  pool directly. Token routing is fused into preprocessing and output stores;
  there is no full-state gather/scatter or K-V/V-K conversion during execution.
- Native MTP proposer, rejection sampler, request scheduler and APC boundary
  migration remain native. The synthetic verifier in the operator protocol
  probe is an oracle stimulus, **not** a replacement service sampler.
- Pure verification keeps the ordinary row-parallel path; prefill/mixed retains
  the existing MC2 policy. Target and native draft graphs need not be one graph.

## Bounded checks

- `candidate_state_probe.py`: two graphs,24 continuation waves, independent CPU
  recurrence, all candidate states, conv history, slot0, request permutations,
  empty lanes and query1 after acceptance3. `DEVICE_FEEDBACK=1` also checks
  device feedback/cursor/anchor advancement; no host accepted-count publication
  after bootstrap. Stagewise CPU checks distinguish BF16 convolution rounding
  from state-selection errors.
- `mixed_state_probe.py`: dynamic prefill/verification composition, warm/cold
  state, two graph banks. `USE_PUBLICATION=1` exercises the service publication
  slab and device feedback mapping; `PURE_VERIFY=1` checks its small-graph branch.
- `compare_recurrence.py`: same-host native V-K versus owned K-V recurrence
  correctness and short graph replay control. State-layout conversion is outside
  execution/timing. The current C8 MTP kernel is slightly slower, so do not infer
  an end-to-end improvement from the no-MTP results.
- `service_smoke.py`: real TP2, MTP2, FULL, APC align, repeated prefix and new
  requests arriving during decode. Requires leased devices and the frozen
  capsule's `service_adapter` bootstrap at the existing Worker import.
  APC replay must itself exceed two1536-token blocks because MTP drops the
  last matched block. Warm C1/4/8 timings count output tokens over complete
  cohort wall time (including prefill), **not** decode-only throughput or SWE.

Read repo-knowledge's `scenarios/qualify-qwen-mtp/GUIDE.md` for observed
receipts, rejected experiments and remaining integration boundaries. Device
admission and cleanup are external to these probes, not bypassed by their entry.

## Observed end-to-end control

Same hw3 pair, MTP2 in both arms, APC actual hits82944/170587 queried tokens
in both. Median complete-cohort output tokens/s after one warm cohort and three
measured repetitions: C1 native38.78 / owned42.30; C4 71.35 /80.05;
C8 93.90 /104.93. These include prefill and are not decode-only measurements.
The native arm carries only the necessary SD/V1 Mamba postprocess ABI bridge.

Frozen receipts: `runs/qwen-mtp-service-20260920-v7/evidence/receipt.json`
and `runs/qwen-mtp-native-20260920-v2/evidence/receipt.json` (local artifacts,
not packaged). Target captures26graphs/2.26GiB; native draft graphs remain
separate. The runner's existing corrected-length receipt fence is retained.

## Count sweep

`MTP_TOKENS=0..4` selects the bounded service experiment. Zero uses the existing
no-MTP product path; positive counts use candidate rows of width K+1. The
recurrence records at most three token updates per kernel, so K3/4 use two
ordered kernels within the graph, continuing directly in the state pool.
A direct four-token static expansion caused pathological compiler runtime and
was rejected. Operator state/publication checks cover all four positive counts;
consult the frozen service receipts before claiming model-level qualification.

`timeline_probe.py` installs diagnostic methods on the existing Worker (no new
Worker type). Profiling is opt-in via `MTP_PROFILE=1`, after timing, six steps
per window. `profile_counts.py` preserves native raw export and writes TraceLoom
Perfetto timelines plus exact target/draft costs. `summarize_counts.py` produces
a qualification-aware table, excludes failed/partial cases, and requires actual
prefix-hit equality within each completed pair.

Across K0 and positive K, native APC reuse differs: Eagle drops one full block.
Do not attribute the entire end-to-end difference to speculative execution.
Record actual scheduled/padded widths, accepted length, cache hits and target /
draft periods alongside output throughput. This fixture is synthetic, not SWE.

## Opt-in MTP2 APC lookahead experiment

`MTP_APC_BOUNDARY=1` in `service_smoke.py` selects `apc_boundary.BoundaryScheduler`
and the matching draft-input leaf hook in `service_adapter.py`. K2 candidate
only; no product defaults change. Diagnostic cache-reset routes are enabled only
on the harness's localhost service. Capsule must include `apc_protocol.py`,
`apc_boundary.py` and `apc_verification.py` alongside the existing prototype files.

Draft KV at i depends on hidden[i] and token[i+1]. Hashing one lookahead token
makes identical-continuation checkpoint reuse safe without blanket last-block
retreat. Unknown lookahead delays publication; different lookahead cannot alias
and falls back a checkpoint. Incomplete-prefill draft input is explicitly the
next known prompt token. This does not add a hidden-state cache or permit
arbitrary-token restoration of GDN. Salt remains part of the identity; multimodal,
LoRA, resumable input and external cache connectors are excluded.

Frozen `mtp-apc-20260920-v4/candidate-k2` passed both1536/3072 cold/warm and
changed-lookahead comparisons, restored3072-token hits, and measured C1/C4/C8
58.51/193.36/305.92 output tokens/s. Same synthetic workload as the count sweep;
not an interleaved control or SWE claim. Numerical service code in that capsule
is the qualification anchor; normal product admission remains unchanged.

## N+2 ownership boundary (work in progress)

Captured target and merged draft graphs alone are not a device-continuation
protocol. The current native runner still waits for hybrid accepted-count D2H,
corrects optimistic CPU lengths, chooses/migrates Mamba slots on CPU, and stages
those counts back to device before building the next metadata. Do not remove
these waits while their CPU consumers remain. Raw accepted output count,
APC-reset state-selection count and externally emitted token count are distinct.

LiveInference `05ac1541` is the reference, not an imported runtime dependency:
`runtime/meta_tensor.py::replay_construction_program` fixes the host construction
boundary; `runtime/shadow.py::publish_externalized` publishes stable roots;
`serve/qwen35/mtp/resources.py` separates mailbox banks, canonical continuation
and egress banks; its scheduler maintains two outstanding qualified waves.
Its BlockTableIngress still constructs Torch tensors: our NumPy views borrow
its ownership discipline, not an assertion that its own implementation is NumPy.

`host_metadata.py` fills one bank's pinned CPU views without per-request Torch
construction. It does not move acceptance back to host or change publication
fences. CPU equivalence and NPU state probes qualify this packing transformation;
real service qualification must also preserve cross-wave lifetimes. Faster host
preparation is not proof of N+2 scheduling and must not mask a FIA lifetime fault.

`fia_feedback_probe.py` tests a separate prerequisite: host-planned KV upper-bound
FIA tiling with actual lengths authored by device feedback inside two graphs.
It runs synthetic acceptance with ingress/compute/egress streams and two
outstanding waves; receipt collection never supplies the following wave's
lengths. It tests real native FIA in pure verification and mixed layouts, not
an integrated model, sampler, GDN/APC transition, request turnover or service
performance. Inside `torch.npu.graph`, native launches must use the CURRENT
capture stream, not a stream handle saved before entering capture.


`draft_banks.py` repairs a separate native ownership mismatch: the shared
runner dispatcher bank-qualifies draft graphs, so their update handles,
workspaces and attention parameters must also be bank-qualified. Both native
draft resource planes are scoped around the existing proposer entry points and
restored on exit. This removes cross-bank updates; it does not turn the native
runner's CPU acceptance/length consumers into device continuation.

`apc_device_preprocess_probe.py` tests GPU-selected running-state columns and
direct-pool migration using the pinned Ascend SD copy primitive. Device-generated
acceptance advances logical progress separately from APC-reset state-selection
counts. Full-pool snapshots are diagnostic egress only, NOT a proposed service
state gather/scatter or duplicated canonical pool. Upstream V2 generic Triton
precopy does not compile on the pinned Ascend runtime; see the knowledge guide
for the bounded alternative and its qualification limits.
