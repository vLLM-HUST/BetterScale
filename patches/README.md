# Maintained patch bundle

The kept implementation is now **`src/strengthen_dsv4/patches/`**, loaded by the
explicit native `worker_cls` integration in `strengthen_dsv4.worker.Worker`.
These are source-visible Python overrides, **not** patches silently applied to
release submodules or `site-packages`, and not a claim that upstream exposes a
fully public plugin API for all these hooks. Private API compatibility is checked
against `src/strengthen_dsv4/pins.json` before model loading.

Each directory is a closed feature module with its own `install` entry. Small
modules keep their implementation in `__init__.py`; only split-draft has private
helpers. No sibling patch imports or automatic import-time hook installation.
These are six Python packages in one wheel, not six separately versioned products.
Worker owns composition and lifecycle; independent source ownership is not a
claim that arbitrary patch combinations have been hardware-qualified.

| Module | Owned native boundary | Install phase | Mechanism evidence |
|---|---|---|---|
| `compat_lcm/` | joint K5/TP capture alignment only | Before runner construction | runs007/009/012 |
| `target_full/` | DSACP support/build/RoPE and FIA request padding | Before runner construction/capture | runs012/045 |
| `ordered_replay/` | ACLGraphWrapper hook and same-stream admission | After native warmup | policy026/031 |
| `qli_cpu/` | DSACP QLI metadata builder | After native warmup | shadow029 |
| `split_draft/` | DSpark runnable, private graph banks and context/query split | After warmup; lazy capture |022/045/046/049 |
| `cross_step/` | Runner input/state/forward receipt placement | After native warmup |028/029/031 |

Each directory's README explains scope, original-code interception, prerequisites
and tests. `split_draft/_graph.py` and `_metadata.py` belong solely to that module.
The reported seven mechanism IDs still include both draft-bank and split-context
improvements; they deliberately map to one cohesive split_draft implementation.

All paths in the code column are beneath `src/strengthen_dsv4/patches/`.
Install the package in the existing donor environment, then add
`--worker-cls strengthen_dsv4.worker.Worker` to the native `vllm serve` command.
Selecting this class installs the kept combination. There is no custom CLI,
private baseline/optimized profile, artifact requirement or environment rewrite.
Restart with your original native worker/command to roll back; do not hot-unpatch
live graph objects. The pinned K5/TP8 SP alignment issue can also affect an
unpatched baseline, so removing the worker is not a same-config A/B guarantee.

No `git apply` step is required: unchanged gitlinks stay the baseline. The source
manifest's digests bind specific private APIs, not a blanket binary reproducibility
claim. There is no automatic plugin discovery, donor upgrade or operator build.

The historical `prototypes/full-mixed/` tree remains a historical evidence/reproducer
surface, including experiments that were **not adopted**. It is not imported by
the maintained bundle. Whole-KV shadows, dummy fixtures, padded all-mode draft and
the N+2 scheduler are excluded from the serving default.

[Chinese mechanism report](../docs/REPORT.zh-CN.md) ·
[Launch/rollback runbook](../docs/RUNBOOK.zh-CN.md) ·
[Exact historical result ledger](../RESULTS.md)
