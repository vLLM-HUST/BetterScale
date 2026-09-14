# Retained BetterScale versus native donor: HTTP service acceptance

This experiment measures the kept package against donor, NOT the last
experimental early-budget patch against an already optimized control.
It is an experiment harness, not a second package installation or serving CLI.
The product entry remains native `vllm serve ... --worker-cls betterscale.worker.Worker`.

## Frozen programs

`run_hw3.sh` records the exact single-host execution environment and native
commands used on hw3. It assumes the already prepared donor runtime and explicit
model path on that host; it does not install/upgrade donor or reserve foreign
resources. The local artifact root is
`/workspace/strengthen-dsv4/runs/tp-continuation-20260914/`; the corresponding
hw3 root is `/workspace/my-ascend-workspace/runs/tp-continuation-20260914/`.

The `retained-vs-native-v1` closure consists of:

- `src/strengthen_dsv4` and `src/betterscale` from release `v0.3.0`; source under
  main `7f0bbdb` has no changes relative to that release.
- `native_worker.py` from this directory. DP native uses NPUWorker directly.
  TP native installs ONLY the already-required K5/TP8 LCM startup repair;
  it installs no target, draft, metadata, or ordered-replay optimization.
- HTTP `service_bench.py` retained at Git object
  `81ecde3:prototypes/tp-continuation/service_bench.py`, with its startup timeout
  increased from540s to900s. The existing admission `launch.py` and home-lease /
  selected-device monitoring helpers are preserved in the source capsules;
  total execution bound1500s, idle admission polling2s, at most30min.
- The existing full32 original-input retrieval request file `quality-inputs.json`
  in the run root. Scoring uses `prototypes/full-mixed/score_quality.py` and its
  pinned upstream evaluator; no new metric or abbreviated request set.

Prefix the native CANN `PYTHONPATH` with the closure; do not replace it. Run157
lost CANN's ACL Python path and failed worker import before model loading.158
fixes only the launcher environment preservation, after a bounded import check
of all three worker classes. No donor/model change was needed.

## Comparison contract

DP: TP1/DP8/EP8,2 seats/rank,16 HTTP clients,budget1026/rank,8GiB KV/rank.
TP: TP8/DP1/EP8,4 seats,budget4128,12GiB KV/rank. Each topology has its OWN
matched control; do not compare raw DP/TP rates as identical-concurrency work.
Both use K5, standard rejection, W8A8, MP/async scheduling, no prefix caching,
normal HCCL and upstream shared-expert overlap.

Native uses its supported FULL_DECODE_ONLY mode and small decode buckets;
retained uses FULL target with the qualified additional prefill buckets.
Forcing unsupported native FULL prefill would not be a fair runnable baseline.
The scripts preserve all other within-topology model/service parameters.

Every process records the same short first HTTP cohort, then three repetitions
of occupied decode, balanced4K prefill and queued skew/turnover. Include all
three repeats; do not discard a slow first repetition to conceal lazy capture.
Generated tokens and draft acceptance remain native, not forced identical.
The32-question quality gate follows, outside timing. No profiler, shadow oracle,
early-budget agreement or per-wave diagnostic receipt is enabled.

Runs158/159 are DP native/released;160/161 are TP native/released. The previously
completed155 is the separate qualified-startup DP variant, NOT release0.3.0:
its metadata/producer banks are prepared before READY. Keep its source identity
and results separate rather than silently attributing them to the published wheel.

Summarize complete capsules with
`81ecde3:prototypes/tp-continuation/report_service.py`. Keep per-cohort output
counts, draft counters, latency tails and quality alongside throughput. Native
profiled/matched-step evidence is a different claim from these HTTP results.
