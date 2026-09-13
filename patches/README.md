# Maintained patch bundle

The kept implementation is now **`src/strengthen_dsv4/patches/`**, loaded by the
explicit native `worker_cls` integration in `strengthen_dsv4.worker.Worker`.
These are source-visible Python overrides, **not** patches silently applied to
release submodules or `site-packages`, and not a claim that upstream exposes a
fully public plugin API for all these hooks. Private API compatibility is checked
against `src/strengthen_dsv4/pins.json` before model loading.

| ID | Maintained code | Native boundary | Installation | Evidence |
|---|---|---|---|---|
| compat-lcm | `target.py` | `CompilationConfig.adjust_cudagraph_sizes_for_spec_decode` | Before runner construction, both profiles | K5/TP8 target/state runs007/009/012 |
| target-full | `target.py` | DSACP support/build/RoPE; runner FIA request padding | Before target capture | Target shadow, real TP8 runs012/045 |
| ordered-replay | `ordered_replay.py`, `target.py` | `ACLGraphWrapper.__call__` | Wrapper before capture; enable after native warmup | Same-stream guard; policy026/031 |
| cpu-qli | `qli_cpu.py` | DSACP `_build_qli_metadata` | After native warmup | CPU mirrors and conservative-bound shadow029 |
| private-draft-banks | `draft_graph.py` | DSpark `_runnable` | After native warmup; lazy first capture/replay | First-call and KV qualification022/045 |
| split-draft-context | `split_draft.py`, `metadata.py` | Native context hook before query body | Same lifecycle; no scheduler replacement |044/045 state;046 timing;049 quality |
| stable-receipt-cut | `cross_step.py` | Runner input/state/forward hooks | After native warmup, stable K5 only |028/029 state;031 isolated timing |

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
