# Mixed GDN fusion: layout boundaries and bounded probes

Use this note when changing the experimental MTP mixed core. Numerical ownership
is unchanged: direct K-V state pool, separate prefill/verification state slots,
real device lengths, native convolution history, and two publication banks.
Normal released product admission is not changed by these prototypes.

## Whole inter-projection path

The input projection supplies packed X[T,5120] and a/b[T,24]. Two native
convolutions service disjoint prefill/verification rows into the same X-shaped
activation tensor. Their modes are genuinely different: prefill handles long
queries/initial flags; verification chooses temporal history from accepted count.
Simply deleting one call is not a valid fusion.

The old path then does:

1. Two mapped preprocess calls normalize Q/K, pack V and compute g/beta.
   BF16 Q/K and beta rounding boundaries are part of the contract.
2. Prefill chunk-local cumsum -> two head-major gate transposes -> KKT ->
   16x16 triangular solve -> 64x64 merge -> WY recomputation.
3. Four Q/K/W/U transposes -> owned AscendC H/O against the direct state pool.
4. Verification recurrence writes every speculative candidate into that same
   pool, using disjoint rows; it produces token-major activations.
5. Restore combines head-major prefill and token-major verification outputs in
   original token order. This is **activation routing**, not state restoration.
6. Existing output RMSNorm already fuses its z gate; the output projection/MC2
   is outside the recurrent block. Do not invent an unfused sigmoid/RMSNorm pair.

The implemented bounded layout experiment removes FIVE of six transposes:
Q normalization writes head-major Q; one cumsum kernel also packs beta and
writes cumulative g head-major; WY writes W/U head-major. Logical Python views
remain [1,T,H,D], so H/O receive their existing contiguous [1,H,T,D] inputs.
K remains token-major for pinned KKT/WY and still needs one H/O conversion.
Changing K universally would require changing those consumers too; a second
K output from preprocessing would trade a transpose launch for another write,
not eliminate bytes by magic.

`mixed_layout.restore_tiled` uses explicit token/head/channel axes and 16-token
blocks rather than flattened head arithmetic and a program per token/1024-vector.
It retains the same gather mapping, masks, padded-output semantics and separate
verification producer. It does not yet route directly from AscendC O/recurrent
producer into final model output. That larger ABI change is not needed for this
bounded improvement and must account for disjoint padded stores if attempted.

## Operator evidence (September21)

hw3 device5, native donors as in GUIDE, actual TP2-local qk8/v24/K=V128.
`mtp-gdn-layout-safe-20260921` PASS, exit0, released (address-safe restore;
the preliminary version is retained only as a diagnostic below). `layout_probe.py` checks six
independent CPU recurrence cases (two FULL banks, every candidate state and exact
conv history), then four complete-core old/new captured controls with three
changed-input/count/warm-start generations each. Active outputs and ENTIRE
state/conv pools are exactly equal between old/new in all twelve paired cases.
The CPU recurrence max output is7.6294e-6 and state7.50665e-5 (chunk BF16 math).

Unprofiled NPU events, twenty replays/block, ABBA, mean of two blocks per arm:

| capacity | actual request lengths | old / layout-fused core, us |
|---|---|---:|
|512|3,33,257,97|1182.378 / 744.826|
|1536|3,1024,509|2504.268 / 1293.752|
|2048|3,1536|3023.896 / 1550.260|
|64|3,17,2,1|567.101 / 437.576|

These include convolution, preprocessing, both recurrent branches, routing and
state handling, but NOT projection/RMSNorm/FIA/communication or HTTP serving.
Sequential bounded operator ABBA is not a population-level performance claim.

## Next-wave slot publication

`device_slots.py` fuses real GPU sequence length -> aligned block-table column ->
physical slots -> mixed role/initial-state publication. Host provides membership,
never accepted counts or corrected lengths. It writes existing bank tensors,
leaves inactive sentinels untouched, and keeps uploaded/consumed/APC fences.
N/NP/NV are non-specialized launch scalars to avoid first-use compilation for
all request-count partitions. Capture fixes those launch scalars; this kernel is
called during normal publication, not advertised as device-dynamic graph shape.
The ordinary Torch implementation remains an independent CPU/reference route.

`mtp-slot-fusion-20260921` PASS/exit0/released: two banks,24 changed device-table/
length waves for each of pure verification and mixed (48waves), exact all-field
CPU comparisons including zero physical slot and inactive tail. The earlier
layout probe additionally checks twelve eager membership/boundary cases.
Eight publications/capture,20replays/block, ABBA: pure~132.86->2.62us,
mixed~200.75->5.00us per publication. Not service seam timings; three groups and
other preparation work remain in the service. No accepted-count D2H is added.

## Remaining opportunities, not established improvements

- The 1216-token triangular-solve tiling still processes partially padded tasks
  even though wholly empty tasks already skip. Smaller task granularity or
  solve/merge fusion is a separate numerical/performance experiment.
- WY loops through heads and executes masked work in empty chunk tasks. Changing
  scheduling could reduce wasted work, but must be benchmarked against this
  layout-only version rather than bundled into an unexplained attribution.
- Convolution/preprocess and O/output-routing fusion require producer ABI or
  kernel changes; a lower operator count alone is not evidence of faster execution.

Frozen receipts and local sources:
`/root/my-ascend-workspace/runs/qwen-mtp-gdn-fusion-20260921/`.
Real TP2 qualification of the address-safe variant is recorded below; operator
success alone does not qualify APC/N+2/model integration.

### Startup boundary under investigation

The first combined service attempt failed before serving, during the first
2048-capacity capture warmup: device MTE address out of range, reported later
at `aclnnMatmulAllReduce` (507035). That reporting call is NOT yet identified as
the faulting operator. Do not classify it as an MC2 bug, nondeterminism, or a
passed service. The independent `mtp-gdn-dummy-20260921` also passed exact
output/full-pool checks for eight-prefill partitions [1]*7+[2041], [256]*8,
and [1]*7+[9]; thus partition shape alone did not reproduce the fault.
Its timing samples overlapped another owned service startup and are diagnostic,
not added to the performance table. The diagnostic service adds startup-only
field comparisons against the old metadata implementation and per-stage eager
synchronization, plus bounded CANN logs, to locate the failing boundary.

Diagnostic-v1 failed in the observer with a missing quoted string (`_audit_done`),
not numerical execution; no inference from that failed observer. Diagnostic-v2
locates the real fault at the first restore launch, after every preceding GDN
stage synchronized successfully. All three groups' actual startup slot fields
match the old implementation. A separate packed-slab probe also matches every
byte for three groups at capacities24/512/2048, ruling out that tested adjacent-
field overwrite hypothesis.

**Paid compiler boundary:** `restore_tiled.ttadapter` lowers BOTH gathered branch
loads to unconditional `memref.copy` loops and applies vector masks afterward
with `arith.select`. Its unselected verification offsets were negative; standalone
allocator slack hid the invalid read. CANN maps the fault PC exactly to
`restore_tiled_0` (diagnostic rank0 base0x124c009f7000, offset0xb58). Do not diagnose
this from the later MC2 error-reporting call. The fix uses `tl.where` to make both
branch addresses legal BEFORE either load, retaining masks for semantics. No
synchronization or state movement is part of the repair. The table above is the subsequent address-safe variant's fresh paired run,
not the unsafe version's earlier timings. Full compiler witness and
compact fault localization live in the local artifact's `fault/` directory.

Address-safe service-v2 completes26 captures and all17 smoke/boundary requests
without device faults. Its harness then fails storing the already-evaluated
boundary result (`receipt[boundary_checks]` instead of a quoted key); receipt is
FAIL, no profile was collected. This second shell-quoting error prompted switching
all subsequent harness generation to local files/scp plus Pyflakes preflight.
The next harness reusing the same numerical capsule fails at startup loading a
cached npugraph_ex callable: `too many values to unpack (expected 21)`. It does
not execute the GDN change. Preserve the failed cache; use a fresh per-run
VLLM_CACHE_ROOT, as with the earlier paid native AOT collision, rather than
clearing shared caches or interpreting cache ABI failure as kernel arithmetic.
The exact cached-signature collision mechanism is not diagnosed here.

## Real TP2 qualification and short timeline (PASS)

`mtp-gdn-fusion-final-20260921` reuses numerical source
`mtp-gdn-fusion-service-20260921-v2` unchanged, with repaired/linted harness and
fresh capsule-local VLLM compile cache. Same hw3 physical6/7, TP2/BF16/MTP2/APC
align/AIV,6GiB KV,8seats,2048budget,8192context.26 target captures;17 smoke/APC
requests, both1536/3072 cold/warm/branch assertions PASS; then warmed C8 decode
and joining mixed profiles, six steps each. Server/admission0; cards released.
No diagnostic stage synchronization or CPU slot comparisons in this final path.
The local prototype enables layout fusion by default (`MTP_GDN_LAYOUT_FUSION=0`
retains the explicit operator control); public product admission remains unchanged.

TraceLoom latest main before analysis was `bf6fb4912106147859d278e4f64177eed256106b`.
Fresh analysis of BOTH full old/new PROF inputs, with opt-in `qwen35-serving.yaml`
compute-layer landmarks. These labels are not scheduler ownership. Six exact
16-FIA target + six exact2-FIA draft graphs per phase/rank; native capture-task
updates remain absent, and pure verification still has zero MC2 kernels.

- Same six decode dispatches `[3]*8`: five post-draft seams mean rank0
  **1.06645 -> 0.56168ms**, rank1 **1.04524 -> 0.56597ms**. All three AICPU Index
  calls disappear in every seam, replaced by three AIVEC slot kernels (~7us
  total in the first decode seam). Twelve OTHER AIVEC Index calls remain; don't
  advertise zero Index across the entire runner. This is profiled preparation
  work reduction, not pure idle or a new unprofiled SWE throughput result.
- Excluding the first target on BOTH arms/ranks, decode target means are
  43.026/43.047ms old and43.048/43.048ms new. Its recurrent branch is unchanged.
  Preserve first-target outliers in raw data: old rank1~149ms, new rank0~45.54ms.
- Large mixed capture: old `[3,33,257,97]`, new `[3,257,97]` followed by
  `[3,3,3,33]`. Both large steps pad to512, but **arrival partitions differ**;
  170.596/166.810ms old versus142.937/143.023ms new is not a matched-work causal
  percentage. Use the independent matched complete-core ABBA table for gains.
- In those large512-capacity graphs, transpose count is **288 -> 48**, exactly
  five removals per GDN layer. Rank0 transpose sums5.952 ->1.164ms; restore
  20.345 ->3.628ms (rank1 20.328 ->3.402ms). Fused cumsum/packing~0.400ms versus
  old cumsum~0.191ms plus gate transposes already counted above. WY~6.57ms and
  the triangular solve/merge remain visible; the block is not “fully fused.”

Local `comparison.json` retains every schedule, target, seam and typed Index
population, plus semantic guards and caveats. `candidate-traceloom/` has the four
new readable exports; `control-traceloom/` reanalyzes the old bytes under the same
new analyzer. Final CPU suite123PASS. This task did not publish to PyPI/GitHub/
website, investigate eviction recovery, or qualify K3/K4 service.

## Follow-up: solve/WY fusion boundary

Read [solve-wy-fusion.md](solve-wy-fusion.md) before fusing the remaining solve/WY
chain or shrinking1216tasks. Fully fused small waves improve, long waves regress;
partial merge+WY fusion does not remove that tradeoff. It records matched
complete-core controls, compiler-workspace caveats and the unchanged-default
scope; do not infer universal wins from fewer explicit HBM tensors.
