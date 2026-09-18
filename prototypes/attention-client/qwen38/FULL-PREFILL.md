# Whole-model prefill replay

This is the owned Qwen3.8 Flash Next attention/client prototype, **not** the
public vLLM Worker and not the Qwen3.8 27B donor patch. Target attention, PLE,
GDN state maintenance, remote routed experts, local shared experts and the MTP
prefill update belong to one captured invocation. Expert servers stay persistent
on their own devices; capture does not move them into the attention graph.

Enable with `QWEN38_PREFILL_GRAPH=1` on `run_model.sh --trace-plan ...`.
The default remains eager until broader serving coverage is qualified. Existing
decode/MTP FULL replay is unchanged. This does **not** add mixed scheduling:
pending prefill and decode still take separate waves.

## What the 27B implementation taught this path

The reference is `feat/qwen38-serving` at `8b0aebb`, particularly the
`qwen_gdn` patch and its owned device-authored metadata. Carry its ownership
principle, not its operator/layout: that GDN pool is K-V; our current backend
has its own state/candidate contract. Equal square state shapes do not make
those layouts interchangeable. Never qualify padding solely by current output;
inspect retained state and subsequent decode/continuation too.

Here the LiveModule topology already takes device lengths and validity masks.
`trace_prefill.py::PrefillGraphRunner` owns stable input IDs, query lengths and
prefix cursors for two static variants (cold and continuation). Capture owns
all derived topology operations and retains the ForwardContext. It includes:

1. Canonicalizing the accepted speculative GDN endpoint before prefill.
2. Target forward and its PLE mailbox publication/receipt.
3. Pairing target rows for MTP, including the previous-turn boundary.
4. MTP state updates, recurrent carry and next-token sampling.

Only host input publication, result conversion and validity reporting stay
outside. They use the same stream in this qualification version; this is not an
asynchronous H2D or ping-pong claim. Moving the validity read past MTP also
removes an intermediate host synchronization from the refactored eager control.

There is one fixed physical bucket: per-seat width is `min(512, lanes / seats)`.
Logical tails are right-aligned and masked, **not** interpreted as a longer
logical sequence. Cold/continuation remain separate Python specializations;
this does not license arbitrary new request arrival into a continuation seat.
Both graphs are warmed/captured before workload timing. Their scratch pools are
currently independent; reset graphs before releasing buffers or closing PLE.

## State gate and current boundary

`QWEN38_PREFILL_SHADOW=1` additionally performs same-state eager/replay shadows,
restoring all root State and
`trace.multi`, while allowing mailbox generations to advance monotonically.
It compares tokens, publication validity and every retained tensor. Integer
state is exact; floating differences are reported and rejected above relative
L2 0.001. The gate exercises changed lengths, a one-token tail, the full bucket
and inactive alternating seats under each static variant. It is an opt-in
qualification cost, disabled by default: cloning an entire near-capacity State
pool twice would invalidate the serving memory budget. It is never a
per-serving-wave check.

The initial hw0 A1(TP1)+E3 gate used two seats,512 total physical rows,1GiB State,
K1 MTP and real full-model weights. The first two changed-length shadows were
**129/129 states exact in each variant**. A synthetic two-turn trace then
completed cold256 + continuation44 + decode/MTP + next-turn10 + decode/MTP.
Expanded shadow coverage and matched timing/profile receipts are tracked in
`PREFILL-PROFILE.md`; this is not OpenCompass or full SWE task completion.

## Qualified transport selection and remaining caution

The A2TP1+E3 four-trace/two-turn SWE smoke gate passed with parallel input pack,
fixed-order fused collect and shared overlap. Its12 state shadows (two sources ×
two variants × lengths1/2/256) were129/129 exact each. Preserve that selection
for this graph qualification:

```bash
export QWEN38_PREFILL_GRAPH=1
export QWEN38_PARALLEL_PACK=1 QWEN38_FUSED_COLLECT=1
export QWEN38_SHARED_OVERLAP=1 QWEN38_ONLINE_COLLECT=0
# Only for a correctness gate with spare memory, not capacity measurements:
export QWEN38_PREFILL_SHADOW=1
```

The arrival-order online collector is **not** silently included in that gate.
Its first dual-source whole-model shadow failed on compressed QSA index keys
(relativeL2~0.0209/~0.00555). A diagnostic rerun then passed every state and
completed the same smoke workload. That pass does not erase the original
failure. Arrival-order FP32 accumulation is a plausible source of amplified
BF16 differences, not an established cause; capture/overlap safety is not yet
exonerated for that combination. The failure path now records a repeated-eager
comparison before raising, to distinguish pre-existing nondeterminism from
replay-only corruption on the next natural occurrence. No threshold was relaxed.

The smoke fixture preserves full prompt deltas and prefix reuse but caps each
turn's output to8 tokens. It is not a full SWE trajectory or a throughput claim.
The public Worker defaults, online-collector selection and package release
remain unchanged.
