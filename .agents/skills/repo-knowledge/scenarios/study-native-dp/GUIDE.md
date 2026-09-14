# Study native donor DP+EP

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
