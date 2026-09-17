# Integrate attention and persistent expert roles

Enter here for the separated-serving prototype, especially moving the existing
Next BF16 server to Qwen3.8 mixed W8A8/BF16. This is distinct from public Worker
patches. Do not change their defaults or upgrade the pinned donor to obtain an
experimental model implementation.

## Owned sources and boundaries

- `prototypes/attention-client/qwen-next/README.md` owns the previously qualified
  80B Next A2/E4 baseline. Its priority subdirectory documents generation-tagged
  shared-completion promotion and bounded decode preference.
- `prototypes/attention-client/qwen38/README.md` owns the newer Qwen4Exp lane and
  compact passed receipts. Do not transfer the older model's full48 or
  fine-grained-prefix performance qualification onto this new ABI.
- Qwen38 Python source comes from owned LiveInfer branch
  `lumi/qwen38-flash-next-serving-plan`,820103bf. It requires TP2 minimum because
  QSA has two KV heads and adjacent-rank islands. The new client is one TP2 group
  with one publishing leader, not two independent TP1 sources.
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
