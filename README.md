# BetterScale

Less host waiting. More device execution.
Modular graph, replay and metadata-preparation optimizations for vLLM Ascend.

Formerly **strengthen-dsv4**. The project is now named **BetterScale**; the validated
Python package and Worker entry retain `strengthen_dsv4` for compatibility. The
repository URL below remains the existing address until the GitHub rename is complete.

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
Each patch is a closed directory with its own `install`: `compat_lcm`,
`target_full`, `ordered_replay`, `qli_cpu`, `split_draft`, `cross_step`.
Worker owns their composition; importing a module does not install its hooks.
See each module's README under `src/strengthen_dsv4/patches/` for its contract.

```sh
# In your existing vLLM-Ascend environment (no donor dependencies are upgraded):
python -m pip install --no-deps --no-build-isolation .

# Keep your native serving arguments; add only the worker class:
vllm serve /models/DeepSeek-V4-Flash <your-native-vllm-arguments> \
  --worker-cls strengthen_dsv4.worker.Worker
```

`Worker` is the **only public integration entry**. There is no `strengthen-dsv4`
CLI, environment setup, automatic plugin discovery, private profile or mandatory
artifact directory. The package does not select Python/CANN, set HCCL/allocator
variables, repair library paths, or change service/KV settings. Start with a
working donor environment. Installation needs existing setuptools>=68; the wheel
also works with `pip install --no-deps /path/to/strengthen_dsv4-0.3.0-py3-none-any.whl`.

The original TP admission remains bounded: TP8/EP/DSACP/K5, four seats,4128 token budget,
max length<=15104, target FULL, native scheduler, prefix caching off. This entry
change does not qualify arbitrary layouts or shapes. KV budget, bind address and
other native settings belong to the user; there is no patch-imposed minimum of
four *full-length* resident requests. See the runbook for the complete native
example and source-compatibility boundary.

To remove the patch, stop the service and return to your original native worker
and command. Do not hot-unpatch a live process. Some pinned K5/TP8+SP combinations
need the patch's LCM fix even to start; removing it is not necessarily a runnable
same-configuration baseline. The historical isolated controls remain evidence,
not a second production entry. Shared hosts still require lease/admission.

- [Technical report, with mechanism diagrams (中文)](docs/REPORT.zh-CN.md)
- [Self-contained printable HTML report](docs/REPORT.zh-CN.html) (download and open locally)
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
rank,1026 local token budget,context<=16384,FULL target,DSpark K5 with native
eager draft,DSACP off,prefix caching off. It combines native DSA FULL target
with owned input slots and captured device preparation/metadata. It does **not**
change the kept TP8 combination or enable the rejected all-mode worker extension.
See [the ownership protocol](src/strengthen_dsv4/patches/async_decode/README.md)
and [native launch parameters](docs/RUNBOOK.zh-CN.md#dp8-稳定-decode-continuation独立于-tp8-组合).
Prototype matched-cycle evidence and the packaged-worker acceptance are reported
separately; a faster step is not by itself an end-to-end throughput claim.
