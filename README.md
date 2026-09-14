# BetterScale

Less host waiting. More device execution.
Modular graph, replay and metadata-preparation optimizations for vLLM Ascend.

Install `vllm-betterscale` and use `betterscale.worker.Worker` with your native
vLLM serving command. The implementation lives directly in `src/betterscale/`.

Incremental DeepSeek V4 serving improvements on vLLM + vLLM-Ascend.
Keep the mature engine. Measure each change. Keep improvements that survive
correctness, memory and service-quality comparisons.

## Install and start

See [public package instructions](docs/PYPI.md) for the version-pinned install and
complete native TP8 / DP8 launch examples. This release changes packaging and naming,
not the qualified engine execution path.

## End-to-end service evidence

[September14 HTTP acceptance](docs/E2E-20260914.zh-CN.md) compares retained
programs with native donor, not one incremental patch against another.
On the bounded same-host workloads, the retained TP8 path improves aggregate output
throughput by **35.17%**; the DP8 startup-prepared implementation now shipped in
**0.3.1** improves it by **39.63%**. The DP figure reuses its qualified run155;
this is not a fresh wheel benchmark. DP8 0.3.0 measured −6.11% and remains in the
report, not relabelled. All measured arms pass the retained32-question retrieval
gate; decode latency tails do not improve universally. See all repeats,
configuration and [release provenance](docs/evidence/release-0.3.1.json).

## Pinned upstreams

| Component | Release | Commit |
| --- | --- | --- |
| vLLM | v0.25.1 | `752a3a504485790a2e8491cacbb35c137339ad34` |
| vLLM-Ascend | v0.25.1rc1 | `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` |

These are unmodified release pins, not a claim that a new build reproduces
our previously installed donor packages byte-for-byte. The Ascend pin is a
release candidate, not a stable release. Submodule gitlinks are authoritative.

```sh
git clone --recurse-submodules https://github.com/vLLM-HUST/BetterScale.git
cd BetterScale
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

The kept patches now live in **`src/betterscale/`**, not just the historical
prototype tree. They use vLLM's explicit `worker_cls` lifecycle and normal OpenAI
server. No donor source files are modified, upgraded or rebuilt on startup.
Each patch is a closed directory with its own `install`: `compat_lcm`,
`target_full`, `ordered_replay`, `qli_cpu`, `split_draft`, `cross_step`.
Worker owns their composition; importing a module does not install its hooks.
See each module's README under `src/betterscale/patches/` for its contract.

```sh
# In your existing vLLM-Ascend environment (no donor dependencies are upgraded):
python -m pip install --no-deps --no-build-isolation .

# Keep your native serving arguments; add only the worker class:
vllm serve /models/DeepSeek-V4-Flash <your-native-vllm-arguments> \
  --worker-cls betterscale.worker.Worker
```

`Worker` is the **only public integration entry**. There is no `BetterScale`
CLI, environment setup, automatic plugin discovery, private profile or mandatory
artifact directory. The package does not select Python/CANN, set HCCL/allocator
variables, repair library paths, or change network settings. Without an explicit
KV byte budget, it sizes KV from actual execution residency and physical headroom. Start with a
working donor environment. A source install with --no-build-isolation needs existing setuptools>=77.0.3.
Prefer the published wheel: `pip install --no-deps vllm-betterscale==0.4.0`.

The original TP admission remains bounded: TP8/EP/DSACP/K5, four seats,4128 token budget,
max length<=524288, target FULL, native scheduler, prefix caching off. This entry
change does not qualify arbitrary layouts or shapes. Manual KV bytes and bind
address remain native settings; there is no patch-imposed minimum of
four *full-length* resident requests. See the runbook for the complete native
example and source-compatibility boundary.

To remove the patch, stop the service and return to your original native worker
and command. Do not hot-unpatch a live process. Some pinned K5/TP8+SP combinations
need the patch's LCM fix even to start; removing it is not necessarily a runnable
same-configuration baseline. The historical isolated controls remain evidence,
not a second production entry. Shared hosts still require lease/admission.

- [Technical report, with mechanism diagrams (中文)](docs/REPORT.zh-CN.md)
- [Self-contained printable HTML report](docs/REPORT.zh-CN.html) (download and open locally)
- [Physical KV and context capacity (中文)](docs/CAPACITY-0.4.zh-CN.md)
- [Launch, checks and rollback (中文)](docs/RUNBOOK.zh-CN.md)
- [Patch inventory and native integration points](patches/README.md)
- [Exact TP8 results and comparator definitions](RESULTS.md)
- [DP8 FULL-prefill follow-up: same-budget comparison](prototypes/full-mixed/DP_FULL.md)
  (historical native DP probe; separate from TP8 measurements)
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


### DP8 stable decode continuation

The same Worker also has a separately gated TP1/DP8/EP8 path: two seats per
rank,1026 local token budget,context<=524288,FULL target,DSpark K5 with native
eager draft,DSACP off,prefix caching off. It combines native DSA FULL target
with owned input slots and captured device preparation/metadata. It does **not**
enable the rejected all-mode worker extension. Automatic physical KV sizing is
shared with the TP entry; only TP installs the finite split-draft catalog.
See [the ownership protocol](src/betterscale/patches/async_decode/README.md)
and [native launch parameters](docs/RUNBOOK.zh-CN.md).
Prototype matched-cycle evidence and the packaged-worker acceptance are reported
separately; a faster step is not by itself an end-to-end throughput claim.

## License

BetterScale is open source under the [Apache License 2.0](LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md) for the upstream execution paths
adapted by the patches. The pinned upstream repositories retain their own licenses.
