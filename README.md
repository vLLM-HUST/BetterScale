# strengthen-dsv4

Incremental DeepSeek V4 serving improvements on vLLM + vLLM-Ascend.
Keep the mature engine. Measure each change. Keep improvements that survive
correctness, memory and service-quality comparisons.

## Pinned upstreams

| Component | Release | Commit |
| --- | --- | --- |
| vLLM | v0.25.1 | `752a3a504485790a2e8491cacbb35c137339ad34` |
| vLLM-Ascend | v0.25.1rc1 | `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` |

These are unmodified release pins, not a claim that a new build reproduces
our previously installed donor packages byte-for-byte. The Ascend pin is a
release candidate, not a stable release. Submodule gitlinks are authoritative.

```sh
git clone --recurse-submodules git@github.com:vLLM-HUST/strengthen-dsv4.git
cd strengthen-dsv4
git submodule status
```

No build, package installation or accelerator job happens on checkout.
Upstreams retain their own licenses and build instructions. Before comparing a
fresh build with historical measurements, record the full software environment
and establish an unchanged baseline on the same workload and hardware.

## Working style

- Keep upstream pins unchanged while testing an optimization.
- Carry small, separately reversible patches in `patches/`; avoid a long-lived
  monolithic engine fork. Use native extension points where they fit.
- Compare unchanged and patched runs using the same inputs and settings.
  Report output throughput, TTFT, output gaps, memory and correctness—not just
  a favorable kernel duration. Profiled runs are diagnostic, not timing baselines.
- Package proven changes as a plugin when the actual extension boundary is
  clear. Core changes may remain explicit patches or become upstream contributions.
- Keep model weights, credentials, datasets, build products and raw profiles
  out of Git. Publish compact evidence and compressed timeline references.

## Starting evidence

[Donor C32 investigation](evidence/donor-c32-findings.md) records the current
cache, scheduler and FULL-graph observations. It does **not** establish a stable
C32 throughput collapse or a measured benefit from broader FULL graph capture.
The linked large artifacts remain in the original local workspace.

These starting hypotheses preceded the graph work below; they are not current
completion or performance claims. Prefix-cache granularity remains outside the
kept patch bundle.

## Maintained serving entry and report

The kept patches now live in **`src/strengthen_dsv4/`**, not just the historical
prototype tree. They use vLLM's explicit `worker_cls` lifecycle and normal OpenAI
server. No donor source files are modified, upgraded or rebuilt on startup.

```sh
export STRENGTHEN_PYTHON=/path/to/pinned-donor-env/bin/python
./bin/strengthen-dsv4 check
./bin/strengthen-dsv4 plan --model /models/DeepSeek-V4-Flash
./bin/strengthen-dsv4 serve --model /models/DeepSeek-V4-Flash \
  --profile optimized --artifacts runs/serve-optimized-001
```

Use a fresh process and `--profile baseline` to roll back to native target decode
FULL + eager DSpark (retaining only the K5/TP8 LCM startup fix). Current qualified
entry: TP8/EP/DSACP/K5, four seats,4128 budget, max length15104,12GiB KV/rank.
The launcher fails closed on unsupported private API sources or configurations.
It is a bounded serving integration, not certification of arbitrary production
traffic, concurrency or parallel layouts. Acquire the shared-host NPU lease and
perform admission before accelerator work.

- [Technical report, with mechanism diagrams (中文)](docs/REPORT.zh-CN.md)
- [Self-contained printable HTML report](docs/REPORT.zh-CN.html) (download and open locally)
- [Launch, checks and rollback (中文)](docs/RUNBOOK.zh-CN.md)
- [Patch inventory and native integration points](patches/README.md)
- [Exact TP8 results and comparator definitions](RESULTS.md)
- [DP8 FULL-prefill follow-up: same-budget comparison](prototypes/full-mixed/DP_FULL.md)
  (opt-in native DP probe; separate from the TP8 serving package)
- [Historical prototype and state experiments](prototypes/full-mixed/README.md)

Matched K5 studies isolate20–21% draft-graph and another10–12% stable-receipt
cycle improvement. Split context/query improves observed short mixed waves;
large prefill has no general speedup claim. Both historical arms pass the retained
32-item OpenCompass retrieval gate. These are not proven maximum-concurrency,
KV-capacity or stable end-to-end throughput improvements.

The Markdown report is canonical. Regenerate its self-contained HTML with
`python docs/render_report.py` in a separate documentation environment containing
`Markdown==3.8.2`; do not add documentation dependencies to the donor environment.
The SVG figures are editable vector sources under `docs/figures/`.
