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
git clone --recurse-submodules git@github.com:CubeLander/strengthen-dsv4.git
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

First candidates: inspect the observed timeline overheads, validate finer
prefix-cache granularity, and assess prefill/mixed graph coverage. These are
hypotheses, not implemented or promised speedups.

## Opt-in FULL mixed exploration

[The bounded prototype](prototypes/full-mixed/README.md) has passed TP2 and TP8 dummy
DSV4 + DSpark K5 target FULL replay, including mixed waves and~4K-token buckets.
TP8 deterministic controls include byte-exact shared KV pools and exact valid
output/MTP comparisons; the full real-weight model passed the bounded same-state target shadow. It is **not production enabled**
and does not yet establish full-model serving speedup. Release submodules and
installed runtimes remain unchanged.

## Decode/draft continuation

[DSpark FULL draft and stream-ordered replay](prototypes/full-mixed/DECODE.md)
now have bounded TP8 state checks and a corrected real-model comparison:
matched four-request K5 cycles fall from about65 to52 ms. Whole-cohort throughput
improvement is not yet stable; this remains opt-in, not a production claim.

[Cross-step authorization](prototypes/full-mixed/CROSS_STEP.md) adds conservative
CPU bounds and moves receipt bookkeeping after target submission while preserving
DMA ownership. Real TP8 matched K5 cycles improve another 10–12%; warmed short
cohort results and the narrower-than-production admission are recorded separately.
