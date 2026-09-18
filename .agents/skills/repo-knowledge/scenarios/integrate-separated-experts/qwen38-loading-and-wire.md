# Integrate attention and persistent expert roles

Enter here for the separated-serving prototype, especially moving the existing
Next BF16 server to Qwen3.8 mixed W8A8/BF16. This is distinct from public Worker
patches. Do not change their defaults or upgrade the pinned donor to obtain an
experimental model implementation.

## Current Qwen38 topology result

Enter `prototypes/attention-client/qwen38/TOPOLOGY-RESULTS.md` for the completed
five-topology hw0 C40/K1 retained-prefix pilot and capacity matrix. TP1x5+E3 is
fastest in this prototype (66.13tok/s versus32.86 native TP1x8/EP8), but does not
have the greatest whole-machine KV capacity. The native control globally votes
phases; never call that ratio a stock-vLLM gain. Only the first two full turns
of40 distinct traces were replayed. Extreme-capacity gates use zero synthetic
history, independently of the9K-horizon real workload. The chronological notes
below preserve discovery boundaries; use the result page for current status.

## Owned sources and boundaries

- `prototypes/attention-client/qwen-next/README.md` owns the previously qualified
  80B Next A2/E4 baseline. Its priority subdirectory documents generation-tagged
  shared-completion promotion and bounded decode preference.
- `prototypes/attention-client/qwen38/README.md` owns the newer Qwen4Exp lane and
  compact passed receipts. Do not transfer the older model's full48 or
  fine-grained-prefix performance qualification onto this new ABI.
- Qwen38 Python source comes from owned LiveInfer branch
  `lumi/qwen38-flash-next-serving-plan`,820103bf. That original closure requires
  TP2 because QSA has two KV heads and adjacent-rank islands. The topology
  campaign now owns a disconnected TP1 extension; use its tested overlay rather
  than merely deleting the original TP2 guard. Each source is one TP group,
  with one publishing leader.
- The exact IPC helper appeared later than820103bf; `qwen38/ipc_acl.py` is a
  disconnected, attributed copy. The old branch does not contain that import.
- Keep native binary/config ABI together. The Qwen38 client config has17 words,
  target input is INT8 plus FP32 per-token scales, MTP remains BF16, and the
  weight catalog has four pointers per layer. Never point old Next Python at it.

## Paid numeric and loading lessons

The shared Eco-Tech model has target routed W8A8_DYNAMIC, but fused BF16 MTP
experts. Sixty target QSA projections are quantized too. Attention-side loading
must intercept scales/offsets and preserve the existing island weight slices;
removing routed experts alone does not implement this checkpoint.

Custom BF16-to-INT8 input quantization produced a one-integer difference from
native DynamicQuant. Native input quantization is the accepted ingress; the
server's FP32 SwiGLU quantization has its own passed integer/float reference.
The selected INT8 GMM uses installed CANN CATLASS and actual device group ends,
including empty groups and capacity tails. Preserve CANN header order when
formatting; alphabetical include sorting broke this compile once.

Group full expert loading by shard **within each layer**. Per-tensor safe_open
reparses enormous headers and becomes pathological for this222866-tensor model.
Keep only layer-local ND temporaries while retaining the NZ catalog.

**Observed September17:** Eco-Tech stores the three PLE integer hash/index
buffers as BF16. Original shared `Qwen3.8-Flash-Next` has exact int64 values;
text configs match and casting each original buffer to BF16 reproduces the new
buffer exactly. Sampled unquantized router/PLE embedding rows also match.
`qwen38/ple_metadata.py` restores only these exact original buffers, validating
shape/dtype/cast identity. This is explicitly a **repaired-checkpoint lane**,
not proof that the untouched published quantized model behaves identically.
Never round corrupted BF16 multipliers back and treat them as exact hashes.
Original shard/content identity matters; do not synthesize hash constants.

## Run small gates before full checkpoint loading

`qwen38/run_wire.sh` runs one client plus E4 real layer0 math/graph gate, with
per-device admission and owned-child fail-stop. Passed at rows1/4/32 with changed
inputs, max relative L2 about3.27e-6. This is neither language quality nor full
serving throughput. `run_model.sh ... --construct-only` isolates TP2 root loading
before spending six devices on full-layer integration.

The selected wheel/native overlay is recorded in the capsule receipt. Activate
its OPP before torch_npu initialization, then register the extension after device
admission/binding. **Append** to CANN's PYTHONPATH; replacing it erased `tbe` and
failed compiler initialization. No global OPP installation or LD_LIBRARY patch.

Admission-helper snapshots siblings into PYTHONPATH. Copy only the admission
script into an isolated helper directory, and keep the experiment source in its
own capsule. Otherwise unrelated helper modules can shadow the selected ABI.
On errors, the launcher stops only its children; never kill a foreign NPU job.

## Persistent lifetime is not the process timeout

Inspect the actual launch attributes before blaming queues. The reused microbench
`device-service/launch.cpp` explicitly sets per-kernel timeout10,000,000µs. This
remained active despite a process-level1200s setter. Full-root cold work outlived
the resident server; downstream `neural_collect` then timed out. Qwen38 owns a
longer launch wrapper and rejects old-lifetime ABI receipts. The external role
supervisor remains bounded. Do not turn a short successful leaf into a claim of
indefinite persistent service. The full-model gate also exposed K=0 metadata
carrying a clamped selector into an ordinary GDN backend that requires `None`;
the narrow target-only binding adapter preserves K>0 candidate semantics.

For single-card leaves, require a physical-device argument to agree with
`ASCEND_RT_VISIBLE_DEVICES` **before** set_device(0). Admission alone does not
bind a process. A missing binding was caught and the owned run stopped; that
capsule is rejected. Multi-role launchers bind each child explicitly.

The repaired full48 target lane subsequently passed prefill plus three decode
calls,192generations per E4 owner, with diagnostic per-layer sync disabled; see
`qwen38/full-target-result.json`. PLE Conv1d needed leaf-scoped ACLNN dispatch for
capture (`ple-conv-result.json`). Retain the entire graph input frame, not only
three convenient tensor fields, through graph reset. Full graph qualification
is distinct from these preceding gates; consult the current prototype receipt.

Final September17 gate: `qwen38/full-graph-result.json` passes all48 real target
layers on one TP2 attention group plus E4. Same-State eager/decode-graph hidden
relative L2 is0 on both ranks; three changed-input replays preserve the eager
output sequence. Each E4 owner drains240 calls, all six processes exit0.
This excludes full-model MTP, large prefill, independent-source batching,
language quality and throughput gains. Do not repeat the full model merely to
reconfirm those passed contracts; rerun for an affected change or a new risk.

## Two-source extension and admission boundary

Qwen38 `--sources 2` uses two independent TP2 groups plus E4 (eight cards),
not TP4. Each leader discovers all four output windows before exporting its
input to all server PIDs. Servers must send windows before waiting for input
registration; source IDs cannot be inferred from accept order. Shutdown waits
for both EOF generations before releasing either server's source mappings.

The coordinator's `tasks` field is bounded to32 and also sizes its rolling trace.
Do not enlarge it just to retain more trace records. Full-run co-batch count is
`sum(completed_counts) - waves`; sampled layer pairs are only a ring-tail check.
The two-source CPU integration passed syntax and receipt-analysis checks, but
September17 local attempts063838Z and064359Z were interrupted by foreign NPU
occupancy after admission. The latter loaded all E4 target weights and began
four attention ranks' loading; it did not reach model forward qualification.
No dual-source throughput/correctness claim follows. The launcher now handles
SIGTERM through its fail-stop/finally block to preserve role logs.

For the authorized hw0 migration, enter `qwen38/HW0.md`. Models download directly
on hw0; `/model` is read-only, so use `/workspace/betterscale-hw0/models` and the
explicit QWEN38 model/reference/Python environment overrides. Only two original
Qwen shards are needed for the three PLE integer fields. Do not copy or download
the entire original model to restore those buffers. A dummy INT8 FULL-graph gate
passed on hw0; full-model admission waits behind the direct download receipts.


hw0 subsequently passed full48 two-source graph service. A cold four-wave test
observed0 co-batches because independent cold/capture work shifted the sources;
do not use that as evidence that the coordinator cannot batch. A once-only
post-capture rendezvous plus63 replays/source produced689–694 paired waves per
server (calls[3168,3168]); no per-layer barriers were imposed. Same-State errors
were0 and complete output IDs matched single-source controls. Read the qwen38
README/result before quoting scaling: both single-source controls had a400–493ms
pause at wave52, making the raw >2x ratio unsuitable as a clean server gain.
At that point the cause was unproven; subsequent GC evidence is below.
Typical source step medians were42–43ms alone and46ms together. These are6-card
vs8-card, identical-prompt, target-only short windows, not equal-card or online
workload evaluation. Ring-tail records can contain no pairs despite nonzero
full-run pairing; never label a vacuous sample check as observed layer parity.


The pause is now causally localized:120404Z records a379.9ms generation2 GC
inside the425ms wave52, collecting0objects. Deferring cyclic GC only for the
bounded steady window removes it (wave52 becomes42.1ms). Keep default production
GC unchanged. `--observe-pauses` exposes timestamps; `--defer-steady-gc` is an
explicit <=96-step measurement control with collection before and after.
Both controls yield23.24tok/s(one TP2+E4,6cards) vs44.38tok/s(two TP2+E4,8cards),
zero shadow errors, identical token IDs. In this controlled dual run pairing is
only48waves/owner, not the earlier689–694: concurrent-source capacity gain is
not evidence that co-batching alone caused the scaling. See GC result and
comparison receipts. Do not discard the original pauses or report raw >2x.

