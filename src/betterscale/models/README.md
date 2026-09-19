# One Worker, local behavior ownership

The public entry is `betterscale.worker.Worker`, directly derived from native
`NPUWorker`. Importing it does not install patches. `models.select` uses native
model/configuration fields and validates the entire admitted route before hook
installation. There is no registry, profile, Worker factory or subclass per patch.

| Native configuration | Composition | Numerical/state owner |
| --- | --- | --- |
| Qualified DSV4 TP8 / DSpark K5 | `dsv4.py` | Existing DSA, split draft and cross-step patches |
| Qualified DSV4 DP8 / DSpark K5 | `dsv4.py` | Existing native DSA and async-decode patches |
| Qualified Qwen27 TP2, no speculation | `qwen.py` | Owned K-V GDN and wave FIA |
| Qualified Qwen27 TP2, native MTP2 | `qwen.py` | Native model/proposer; packed conv weights |

These are the existing bounded admissions, not claims of arbitrary model support.
Native configuration remains authoritative; no invalid or missing-library route
silently falls back. Qwen checks only Qwen pins/libraries; DSV4 checks only DSV4.
The process remembers its selected route. Incompatible compositions or a failed
partial bootstrap require a fresh process, not monkey-patch rollback. Identical
installers retain their idempotence; this does not qualify concurrent runners or
multiple independently configured models in one process.

## Lifecycle

Worker delegates only native lifecycle seams: before/after initialization,
model loaded, device/memory initialization, and completion of native warmup.
Absent optional hooks preserve the native call and return value. The components
own the implementation, not more Worker subclasses.

- Qwen: install GDN before donor init, FIA after it, pack immutable weights after
  load. FIA preload and native libraries remain process-start prerequisites.
- DSV4: install target hooks before donor init; install execution continuation
  after native capture. Physical KV accounting is DSV4-only. `dsv4_draft.py`
  composes draft preparation/retirement; memory accounting receives explicit
  trial callbacks rather than calling custom methods on Worker.

Qwen GDN `publication.py` alone owns the shared dispatcher/runner overrides.
`graphs.py` supplies capacity policy, not another wrapper layer. FIA installs its
attention/update/replay leaves and supplies the wave-forward callback; publication
calls it explicitly around its graph-resource scope. Thus the existing order is
visible: FIA wave setup → GDN graph resources → native forward → GDN consumed
fence → FIA release. Host-source and device-reader reuse fences are unchanged.
Cross-layer operations stay at the wave boundary; they are not repeated per layer.

## Migration

Use `--worker-cls betterscale.worker.Worker` for every current configuration.
Old `betterscale.qwen_worker.Worker` and `MixedWorker` names are import aliases,
not separate classes. The old no-MTP single-prefill/native V-K route is retired
from entry selection: both aliases select owned K-V, including when APC is off.
Use the Qwen GDN launcher/environment and a fresh pool/process. Do not interpret
a missing library as permission to switch layouts. Native MTP2 still requires its
original admitted graph configuration and does not install owned GDN/FIA.

Historical prototype Workers and frozen measurement capsules remain experiments;
they are not public entrypoints and have not been mass-rewritten. Existing PyPI
artifacts are unchanged. Numerical kernels and native binary manifests are not
modified by this consolidation. Fresh correctness evidence and historical service
performance must be reported separately.

## Consolidation verification

[Bounded acceptance](../../../docs/evidence/worker-unification.json): 88 CPU
tests; fresh hw3 TP2 no-MTP FULL/NONE checks across 68 rank-steps and 8,772
hidden/cache comparisons (max absolute difference zero), cold/warm APC and
eight shared-prefix branches. DSV4 TP/DP and native Qwen MTP2 use CPU lifecycle
and unchanged/normalized-source evidence, not new hardware runs. No new service
throughput result is claimed.
