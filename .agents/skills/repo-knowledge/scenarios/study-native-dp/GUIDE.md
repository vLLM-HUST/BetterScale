# Study native donor DP+EP

For released0.4.1 APC compatibility, enter `prototypes/prefix-caching/README.md`
and its retained193–195 capsules before re-running prefix experiments. Both TP
and DP cold/warm retrieval gates pass; use native engine affinity for DP. The
older APC-off qualification notes below are historical, not the current guard.


For the active percentage-free automatic KV sizing goal, first read
`prototypes/auto-kv-memory/README.md` and its run165/166 accounting results.

## Prefix/cache investigation: September14 source boundary

Before diagnosing missing SWA or enabling APC, inspect Ascend's
`patch/platform/patch_kv_cache_utils.py` and `patch_kv_cache_coordinator.py`,
not just native vLLM. At the pinned Ascend `9bf964cb` / vLLM `752a3a50`,
Ascend separates C4/C128 cache groups by compression ratio and replaces the
hybrid coordinator. Reading native grouping alone incorrectly suggests one
mixed-compression group. The coordinator converts compressed physical pages
back to logical token lengths.

Observed source: the v0.25.1 branch disables partial hash hits and returns
`lcm_block_size` as common hit alignment. Physical block128 implies16K;
physical block32 implies4K. **Run168's real model census shows the public native
command (no explicit block-size) uses32**, unlike earlier benchmark scripts
with explicit128. The initial inference that public15K/16K horizons could not
hit any prefix was wrong and is retracted. Native KV manager searches at most
prompt_length-1; a4K boundary CAN hit inside this envelope once APC is enabled.
Do not infer actual specs from an older command or treat group block units as
uncompressed tokens. Do not lower only coordinator alignment: compressed hash,
partial-page ownership/writes and SWA/compressor/draft resume must agree.

SWA specs, skipped-block recycling and admission caps DO exist. The SWA peak
bound includes sliding_window-1 plus the wave token budget (capped at maxlen),
rounded to pages with a safety page. Large prefill budgets can therefore hurt
capacity without SWA having been converted to full attention. Confirm the
actual hybrid-manager setting and installed layout before assigning blame.

As of main e584936 (0.3.2 namespace release), APC remains disallowed by the
package's qualification guard. Fletcher wants it supported in a subsequent
release; this investigation has not qualified that behavior or changed defaults.
For the separate eager-config/draft-graph question, read split_draft/README.md:
native generic draft graph exists, pinned DSpark forces it off, TP installs our
runner after warmup, DP does not install that runner.

Enter here for native DP/TP launch, cache ownership, skew and profile comparison,
not implementation of our graph patches. Read
[`DONOR_DP.md`](../../../../../prototypes/full-mixed/DONOR_DP.md) for the measured
matrix, native mechanism anchors and the bounded runner/report protocol.

The offline DP API needs one native client per DP rank with `VLLM_DP_*` env;
Ascend itself offsets local devices. TP1 rejects FlashComm1. All clients must
remain alive during EP drain. The runner preserves native scheduling; its worker
extension is observation plus the already-qualified K5/TP8 LCM startup repair,
not FULL-mixed or draft optimization. Never accidentally import `extension.py`
into this baseline while repairing a fixture.

Use the actual scheduler rows for global query-count comparisons. Profiles
contain dummy EP service after local completion and native32-step global finish
checks. Those are not generated tokens. Native reported cache tokens are hybrid,
max-length concurrency equivalents and depend on the prefill budget; do not
multiply a per-engine receipt and call it verified maximum active capacity.

Runs052/053 established a modest occupied decode-cycle DP advantage without a
stable cohort-throughput win. A1K per-DP-rank budget hurt the single8K skew case;
that is not proof that larger-budget DP cannot improve. Keep real-weight results,
synthetic workload scope and native profiler overhead distinct.

The longer native DP traces exposed a frozen TraceLoom export anti-join with a
missing child-edge index. Reuse `profile_tools/analyze.py --jobs 4 --resume`:
it adds the lookup index to derived DBs only and reuses completed native rank
analyses. Read its README before repeating a slow export; do not truncate real
waves or recapture the model to work around this CPU query issue.


For the opt-in native DSA FULL-target extension, enter
[`DP_FULL.md`](../../../../../prototypes/full-mixed/DP_FULL.md).
Do not apply DSACP hooks unchanged to DSA: native DSA splits query segments in
Python. Reusing its persistent ragged-query metadata is viable, but unifying all
queries onto decode also changes prefill norm/quant arithmetic under the default
attention overlap path. Keep replay-vs-unified, original-native, and quality
claims separate. Large buckets retain prefill arithmetic in the current probe;
runs065/066 show27% shorter balanced-prefill cohort completion and11% shorter
skew completion at the same1026 budget/rank; steady occupied decode is unchanged.
Both real runs pass the retained32-case retrieval quality set. Keep this bounded
DP8 experiment distinct from the packaged TP8/DSACP serving profile.

DP dummy shadow caveat: native `_dummy_run` builds DSA's copied slot mapping,
then clears the source mapping. Rebuilding metadata between replay and an eager
same-program reference therefore changes writes and produces false failures on
idle ranks. Use the exact prepared metadata for that oracle. A distinct native-
split reference still needs deliberate rebuilding; do not conflate its arithmetic
with the unified program. Run064 passes40 exact output/KV checks on all eight
ranks after this correction. Keep first-large-bucket checks independent of the
initial counter: DP's32-step drain can consume a fixed early-only check budget.

Run067 closes large-bucket same-program checks (42/rank,1018 valid owner rows,
all output/KV bytes exact). Native offline parsing must use fresh processes per
rank: reused torch-npu parser workers can leak rank singletons and label rank0
as rank2. Use `profile_tools/parse.py` and validate RANK_DEVICE_MAP; never repair
this by renaming DBs or mutating rank rows. Read that folder README for archive
and bounded export instructions.

For **native donor versus the retained package's HTTP E2E**, enter
`prototypes/retained-e2e/README.md` and `docs/E2E-20260914.zh-CN.md`.
Resolve arms from command AND program source before benchmarking: a capsule
called `control` can be an optimized incremental control, not stock donor.
Runs158–161 compare native/released DP8 and TP8;155 is a separately qualified
startup-prepared DP program. Its22 runtime files match experimental81ecde3,
not published0.3.0. Keep the published DP regression and cold-start behavior
beside the positive TP and qualified-DP numbers; do not discard first repeats.
All five arms pass32/32 retrieval, but latency tails are not universally better.
Reusing those complete same-host capsules is preferable to another blind reload.
The small native TP helper installs ONLY LCM startup compatibility. Preserve
CANN's Python path when prefixing the experiment closure;157 failed import when
its launcher replaced that path. The corrected bounded import check precedes158.

## Peak-memory attribution after automatic fitting

Before another full-model allocation trace, read `prototypes/peak-memory/README.md`.
The September15 TP8 dummy control completes; its target allocated/reserved peaks
match all8 instrumented ranks exactly. The trace later times out in draft startup,
so only the completed target interval is qualified for attribution. Broad Python
tracing across donor/compiler code is painfully slow; reuse retained watermarks
and isolate an operator rather than spending another full-model capture.
Highest target allocated water follows HC-pre; post-draft State clear raises the
clean run's high water by222MiB (543MiB above READY live bytes, a different metric).
Reserved slack and native temporary peaks are not automatically reclaimable KV.

The memory fixes and all-rank hw3 receipts are in
`prototypes/peak-memory/FIXES.md`. The retained two-fix arm (dense KV backing
clear + HC-pre workspace floor removal, original residual forward) passes TP8
FULL startup and short HTTP execution: about186MiB/rank more automatic KV and
360MiB/rank less READY reservation, with unchanged1GiB safety. These are dummy
memory/execution gates, not quality or throughput. Residual aliases have no
material integrated benefit and remain off. Do not add leaf savings together:
clear+alias alone frees540MiB READY reservation but does not increase fitted KV.
For native integration use `prototypes/peak-memory/NATIVE_HOST.md`: complete
original A2 vendor, unchanged API/schema/kernel files, rebuilt complete-A2 host
tilings only. Do not use a selected-op vendor or build ALL (duplicate indexer
symbols); use the pinned build_aclnn.sh A2 operator set. Keep the shared runtime
untouched. The V4 Python module's decoder class still has its legacy V2 name.
