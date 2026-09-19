# Qwen3-30B-A3B joint-wave compatibility

This is a compatibility oracle, **not** a port of the complete device-continuation
protocol. It does not change the packaged BetterScale Worker or installed donor.
Qwen here means `/data/shared_models/Qwen3-30B-A3B` (48 layers, BF16, 128 experts,
top8, hidden2048, no shared expert), not Qwen3-Next or Qwen3.8. No speculative
draft model is configured; the joint graph is target → logits → native sampler.

## Two distinct questions

1. Can native Qwen + TP/DP/EP collectives + greedy sampling share one capture,
   matching an independently completed native invocation, including all KV bytes?
2. Can that capture consume changing device lengths and continue autonomously,
   as the DSV4 prototype does?

Do not label a pass of (1) as a pass of (2). The default `snapshot` mode tests (1):
six actual native decode invocations, one separate capture per fixed snapshot,
two checked replays per capture, and an eager composition control. Capture itself
is not counted as the first committed invocation. KV is restored before every
candidate and restored to the native result before serving continues.

Native reference: FULL_DECODE_ONLY with `pa_shape_list=[1]`, one request/DP rank,
32-token prompt, greedy12 outputs, maxlen256,256MiB KV/rank, FlashComm1 off,
HCCL_DETERMINISTIC=strict. Candidate uses native model/PA/sampler bodies with
runtime graph mode NONE inside one outer graph (no nested target-only replay).
No native attention kernel, expert dispatch, collective, logits or sampler is
reimplemented. This is specifically the native **PA decode** envelope, not a
qualification of default FIA, prefill/mixed, skew, idle EP peers or performance.
Native FULL remains the independent reference, not an eager-only baseline.

## Confirmed device-length gap

The pinned native builder sets `AscendMetadata.seq_lens` to a **CPU** tensor.
`seq_lens_list` is also a CPU list. PA's Tensor-typed `context_lens` alone does not
prove it can consume a device tensor:

- `qwen-tp2-smoke1`: fixture attempted compilation mode0 with native graph capture;
  startup failed because graph parameter workspace registry was uninitialized.
  Fixed by retaining native compilation setup, not by patching the registry.
- `qwen-tp2-smoke2`: continuation's `DeviceOnly` guard correctly rejected copying
  the device cursor into the native CPU length carrier.
- `qwen-tp2-smoke3`: private device-length carrier reached the native ATB PA body;
  both-rank run failed at `PagedAttentionOperation setup failed` in the eager
  composition gate, before joint graph capture. Native FULL startup had passed.
  This tests direct tensor substitution, not the impossibility of another backend.
- `qwen-tp2-smoke4`: two-layer dummy snapshot oracle passes both TP2/EP2 ranks,
  with sampled IDs and every KV backing byte exact for all12 replays/rank.

`QWEN_JOINT_MODE=continuation` retains the rejected experiment for diagnosis; it
is not an advertised working mode. No claim that ATB's Tensor schema implies
supported device length ownership. Native graph task-parameter updates remain a
host alternative; device-aware attention ABI/backend work is a separate decision.

## Matrix and reproduction

Use fresh capsule paths. The unchanged subset admission helper owns per-device
leases, fresh occupancy checks, bounded waiting and cleanup of its process group.

```bash
QWEN_TP=2 QWEN_DP=1 PROBE_DEVICES=0,1 \
 PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/joint-wave/CHOOSE-FRESH-TP2 \
 bash /workspace/strengthen-dsv4/prototypes/joint-wave/qwen/run.sh

QWEN_TP=1 QWEN_DP=2 PROBE_DEVICES=0,1 \
 PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/joint-wave/CHOOSE-FRESH-DP2 \
 bash /workspace/strengthen-dsv4/prototypes/joint-wave/qwen/run.sh

QWEN_TP=2 QWEN_DP=2 PROBE_DEVICES=0,1,2,3 \
 PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/joint-wave/CHOOSE-FRESH-TP2DP2 \
 bash /workspace/strengthen-dsv4/prototypes/joint-wave/qwen/run.sh
```

`QWEN_DUMMY=1` shrinks only to two layers with dummy weights for harness debugging.
It is not a real30B result. All DP clients stay alive until every partner finishes;
DP rank r uses prompt `[17+r]*32`, avoiding identical-activation-only EP coverage.

After successful exit, run `analyze.py CAPSULE --tp N --dp M`. It checks every
EP/TP/DP rank, all six oracle rows, both replays, TP token agreement and each
client's real completion. `summary.json` keeps the exact boundary. Raw evidence
is `engine/rank*.json`, `result-dp*.json`, `complete.json`, and `run/{run.log,exit.txt,release.txt}`.
Each capsule snapshots sources and checks nearby installed donor source identity.
The snapshots contain the executed adapter; repository base commit alone is not
its identity. The admission subprocess has an internal1500-second execution bound.

### Real-weight receipts (2026-09-16)

`qwen-tp2-real1`: TP2/DP1/EP2 **PASS**, both ranks48 layers, native reference FULL,
MoE ALLGATHER. Six fixed snapshots × two exact joint replays/rank; original12-token
completion retained. Process exits0; selected devices released. This includes
vocabulary-parallel logits gathering, not merely a local sampler leaf check.

`qwen-dp2-real1`: TP1/DP2/EP2 **PASS**, both ranks48 layers, same exact six-snapshot /
twelve-replay gates. Each DP engine keeps its own request/KV and participates in
the shared ALLGATHER expert group. Both clients finish12 tokens and exit0.

The combined TP2/DP2 command additionally accepts `QWEN_PROMPTS=language`: two
locally tokenizer-verified English completion prompts (18/19 tokens, numbers vs
weekdays), then the same six decode snapshots. This tests varied token output
rather than relying only on repeated numeric prompt IDs. Prompt text/IDs are
written into each capsule. It is still a balanced decode workload: both prefills
finish in one step; the difference in prompt length is NOT a skew qualification.

`qwen-tp2dp2-real1`: TP2/DP2/EP4 **PASS**, all four ranks48 layers, native FULL
reference and ALLGATHER experts. Language prompts exercise changing sampled IDs:
DP0 `[323,1221,3270,5109,504,825]` at positions18–23; DP1
`[11,8145,8500,315,2849,11]` at19–24. TP peers agree exactly. All six snapshots,
twelve replays/rank and complete KV backing-byte gates pass; both clients produce
12 tokens and the supervisor exits0. No copied constant sample can explain this
varied-token control.

| Real48-layer BF16 topology | Joint fixed-snapshot capture | Dynamic device-length continuation |
|---|---|---|
| TP2 / DP1 / EP2 | PASS on2 ranks | Not qualified; direct carrier smoke failed |
| TP1 / DP2 / EP2 | PASS on2 ranks | Not qualified |
| TP2 / DP2 / EP4 | PASS on4 ranks, varied tokens | Not qualified |

This isolates the tested collective/sampler composition from the length-carrier
seam. It does not qualify MC2/AllToAll, other sampling policies, changing cohorts,
new generations, prefill/mixed, default FIA, or an online scheduler replacement.

### MetaTensor comparison: source clarification

LiveInference local `05ac1541` distinguishes three things that must not be
collapsed into "attention length":

- Ordinary Ascend `llm/attention_metadata.py:build_attention_metadata` still
  constructs sequence lengths on CPU; query offsets, block table and slot mapping
  use MetaTensor recipes. Its ordinary attention class is an integration-owned
  execution stub, not proof of a working device-length Qwen numerical backend.
- DSV4 `metadata/attention.py:DSV4SequenceLengths` is a MetaTensor recipe resolving
  to a device int32 tensor. The separate continuous-wave metadata program derives
  live lengths from device continuation; the CPU-topology recipe alone does not
  establish autonomous progression.
- `core/meta_tensor.py:MetaTensor.tensor` resolves/binds a physical tensor. It
  establishes construction/ownership, not an automatic conversion of the native
  operator's host-side setup ABI into device-side execution inputs.

Public Ascend/op-plugin `op_plugin/utils/custom_functions/atb/AtbCommon.cpp`
(TorchTensor2AtbTensor conversion) assigns CPU tensors to `atb::Tensor.hostData`
and non-CPU tensors to `deviceData`. `ops/atb/PagedAttentionAtb.cpp` supplies the
context_lens tensor to this binding before Setup. Thus changing `.device` can
change the descriptor contract, not merely the address of an equivalent operand.
This upstream source is corroboration, not binary/source identity proof for our
installed torch-npu. We have not established how its final PA kernel receives
lengths (device payload versus tiling/launch metadata). The smoke failure is NOT
proof that device attention generally cannot consume device-authored lengths.
