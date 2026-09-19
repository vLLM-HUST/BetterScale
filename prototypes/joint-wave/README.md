# Joint native execute/sample/draft wave

An isolated runnable protocol transplant, not a change to the published Worker.
Grammar is outside the current envelope. No installed donor files are edited.

For the subsequent real Qwen3-30B-A3B TP/DP+EP compatibility matrix and the
separate device-length gap, see [qwen/README.md](qwen/README.md). Its passing
fixed-snapshot captures are NOT the DSV4 continuation gate described below.

## Numerical and protocol ownership

The graph contains native target forward → logits → greedy rejection sampler →
native DSpark K5 → continuation commit. `continuation.py` owns the next target
input position, anchor, draft, remaining length, EOS, generation and fail-stop bit
on device. Its cursor is **not** LiveInference's emitted-anchor cursor.
Device positions, slot mappings and native attention metadata are regenerated
inside the graph; `DeviceOnly` rejects H2D, D2H and device scalar reads there.

`window.py` authorizes at most two outstanding waves against assigned KV, without
predicting acceptance. The two-wave K5 horizon is `position + 2*6 + 5`.
`transport.py` owns separate ingress/compute/copy streams and two receipt banks:

- grant H2D waits for the previous bank invocation to finish reading;
- replay waits for its grant and the previous bank's D2H reader;
- copy-out waits for graph completion; host retirement waits for that copy;
- pinned grant and receipt buffers stay alive; sequence mismatch poisons transport.

The device validates sequence, generation and one-wave resource coverage before
allowing writes. Invalid grants poison the resident rather than allowing a retry
to advance it. EOS/length terminal generations mask target **and draft** writes;
already-issued waves drain with zero emitted tokens and unchanged KV/state.
This still executes dummy numerical kernels; it is not a hardware early exit.

## Independent native oracle

`worker.py` observes a native `_model_forward` input and snapshots its pre-forward
KV. The native runner completes execute/sample/draft as the independent reference.
The candidate then starts from the original KV and calls the original native
model bodies, hidden-row selection, logits and sampler/proposer. Comparisons cover
sample IDs, draft IDs and **every byte of each distinct KV backing allocation**,
including heterogeneous aliased views. Reference KV is restored before service
continues. This is diagnostic execution, not a production capture-on-miss policy.

The no-discard greedy envelope replaces padded next-token selection only to remove
its unreachable backup-token host ingress. The candidate suppresses count D2H;
it does not pretend host bookkeeping or final serving output is graph-safe.
Native count publication must be reconnected deliberately in a serving adapter.

Both target and draft use native `enforce_eager=True` numerical paths under the
outer graph. This is not qualification of nested native FULL graph dispatch.
CPU attention bounds are conservative specialization facts, not live progress.

## Hardware gates and limits

Local Ascend 910B2 physical 0, TP1, four dummy DSV4 layers (SWA/C4/C128), one dummy
DSpark layer, K5, one greedy resident; prompt128, max output32, max model length1024.

- `run002`: three native decode snapshots, two exact replays per separate capture.
- `run003`: **one capture**, six consecutive native-reference steps; cursor128→140.
- `run004`: six queued candidate waves with no native calls between replays,
  exact receipts and final KV; delayed copy-out tests two-bank ownership. This
  version did NOT yet enforce at most two unretired host submissions.
- `run005`: also EOS and length termination after one emitted token, followed by
  three drains each, with unchanged continuation and all KV backing bytes.
- **`run009`: PASS, process exit0.** One capture, six exact native-reference
  replays, then six autonomous replays with max two unretired waves, exact
  receipts/final KV, and one grant-backpressure retirement. EOS/length drains,
  wrong-generation and insufficient-grant fail-stop tests all pass without KV
  mutation. Grant end152 is derived from assigned tables, not a fabricated cap.

The newer authorization gate records per-group assigned-block evidence. Compressed
KV rows must be scaled by their compression ratio, not treated as original token
counts (`run006` caught this). Initial grants also expire as positions advance;
`run007` passed three reference steps and correctly stopped at position134 against
its stale144-token grant. `run008` then exposed insufficient two-wave lookahead;
`run009` handles it by retiring another pending receipt before authorizing.
The reference phase now refreshes grants from native
allocated rows, never from unassigned pool capacity. The autonomous episode
starts from the original token/KV seed **with the final reference step's allocated
block tables and grant**. It does not exercise live resource replenishment.

This is not yet a replacement serving scheduler: new-generation admission,
block-table replenishment/rebinding, prefill/mixed handoff, multi-seat batching,
TP/DP collectives, startup finite capture and real-weight acceptance remain
unqualified. The natural dummy sequence emits two tokens on every active wave;
terminal clipping covers one and drains cover zero, not every acceptance length.
No throughput claim: the oracle copies full KV and synchronizes for diagnostics.
The transport retains its bounded episode's tickets; it is not an unbounded
production receipt queue.

## Reproduce

```bash
PROBE_DEVICES=0 PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/joint-wave/CHOOSE-FRESH \
  bash /workspace/strengthen-dsv4/prototypes/joint-wave/run.sh

JOINT_WAVE_MODE=continuation PROBE_DEVICES=0 \
  PROBE_CAPSULE=/workspace/strengthen-dsv4/runs/joint-wave/CHOOSE-FRESH-CONTINUATION \
  bash /workspace/strengthen-dsv4/prototypes/joint-wave/run.sh
```

The existing donor runtime, model configuration and CANN installation are used;
dummy weights avoid real checkpoint loading. Per-device admission uses the existing
explicitly authorized subset launcher, bounded waiting, foreign-occupancy monitoring
and owned-process-group cleanup. Source is copied into each fresh capsule before
launch. `engine/oracle.json` records progress and failures; `engine/result.json`
exists only after required gates pass. `run/run.log` retains the complete log.
`run001` failed launcher setup (overwritten CANN PYTHONPATH), not numerical capture.

CPU contracts (14 tests):

```bash
/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env/bin/python \
  -m unittest discover -s /workspace/strengthen-dsv4/prototypes/joint-wave -p 'test_*.py'
```

These cover storage ownership, bank/failure rules and resource-window accounting;
fake CPU streams do not establish hardware ordering. The NPU receipts do that.
