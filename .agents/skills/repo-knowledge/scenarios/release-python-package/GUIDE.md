# Release the public BetterScale package

Use when building or publishing `vllm-betterscale`, or updating its install entry.
The native entry is `betterscale.worker.Worker`, defined directly in that module.
From0.3.2 the wheel contains only the `betterscale` package; no legacy alias package
is shipped. First public release was already `vllm-betterscale`0.3.0. Do not present
private prototype naming or migration advice as the public product's introduction.
Do not uninstall donor packages or upgrade their dependencies.

- `docs/PYPI.md` is the public package README and owns the bounded TP8/DP8 commands.
  Version changes must agree in pyproject and the implementation initializer.
- Build in a new staging directory from tracked package files plus pyproject,
  MANIFEST, public README, root README and license/notices. Do not reuse build/lib:
  it previously retained a removed CLI. `python -m build` builds the wheel from
  the sdist too; audit both archives. Include pins and per-patch READMEs, not
  prototypes, upstream checkouts, tests, credentials or profiles.
- Use separate Python 3.12 packaging tools; unset inherited PYTHONPATH for them.
  A tools-only venv lacks torch/numpy: use the existing runtime Python for the CPU
  test suite, without constructing NPU workers. The isolated Worker-import test needs no
  donor. Do not install torch just to repair the packaging-tools environment.
- Run strict Twine metadata checks, install the wheel with --no-deps outside the
  checkout, verify Worker is defined in betterscale.worker against a native-base
  fixture and the legacy namespace is absent, verify pins load and no console/plugin entry points are introduced.
  These are packaging checks, not new NPU correctness or performance evidence.
- Publishing requires Fletcher's authority. Credentials stay opaque in a mode600
  `.pypirc`; validate the official PyPI endpoint before upload. Never print token
  values or include them in commands/logs. With Twine, use --repository pypi after
  validating the configured endpoint: supplying --repository-url instead bypasses
  .pypirc credentials (observed as Credential not found, before any upload).
  Record the source commit, upload only
  the two audited artifacts, then download from official PyPI and compare to the
  local wheel before claiming availability. PyPI versions are immutable.
- Only after publication, update the website's normal MOD quickstart and detail
  integration section. Development repository visibility and PyPI source visibility
  are separate: do not advertise private GitHub source as publicly reachable.

## First public release

`vllm-betterscale==0.3.0` was published on 2026-09-14 from `d992b5f`.
Both wheel and sdist passed strict metadata/archive checks; 51 CPU tests passed.
The official PyPI wheel was downloaded, directly compared byte-for-byte with the
built artifact and installed into a clean Python 3.12 environment with --no-deps.
The clean install also verified alias identity and readable pins. No new NPU run
was performed for the alias-only packaging change.

## Public repository

Fletcher authorized opening `vLLM-HUST/BetterScale` on 2026-09-14. Before the
visibility change, six published branch/tag refs were inspected (553 unique
historical blobs, 3.95 MB). The offline secret scan's 48 entropy findings were
reviewed as 42 commit/digest values and six local artifact paths; no configured
PyPI credential, private key or credential-bearing URL was found. No GitHub
release assets, Actions artifacts, issues, wiki or discussions needed separate
publication treatment. This is a bounded pre-publication audit, not a guarantee
about future commits. Untracked local worktrees were not published.

GitHub now recognizes the root Apache-2.0 LICENSE (same text as the packaged
license). Public visibility and anonymous repository/license downloads were
verified. Do not restore private-repository notices in the website or install
instructions; keep immutable PyPI 0.3.0 artifacts and their measurement identity.

## DP startup fix: 0.3.1

The DP-only implementation is `db73468`; producer, metadata and startup preparation
are copied from the already-qualified run155 program at `81ecde3`, not reinvented.
`docs/evidence/release-0.3.1.json` owns the direct file/DP-install-branch comparison
and the retained differences. TP's release0.3.0 composition is unchanged; do not
merge the whole experimental branch or bring in early-budget/TP continuation.

Fletcher explicitly chose to reuse the completed same-host results: DP +39.63%,
TP +35.17%, each against its own native baseline. The new release does not imply a
new NPU measurement. The unnecessary run162 was cancelled and its lease released.
55 CPU tests and clean wheel/archive/alias/pin checks qualify packaging and scoped
integration; preserve run155/158–161 identities and all rounds in the E2E report.
Do not repeat hardware qualification merely because those identical DP programs
are shipped under a new package version. Reopen it only for a changed behavior or
a concrete unresolved risk, not to manufacture a fresh-looking measurement.

Published as tag `v0.3.1` / `fbfa963` on 2026-09-14. Official PyPI wheel and
sdist were anonymously downloaded and directly matched to both audited artifacts;
the official wheel passed the clean-install alias/pin check. Evidence is retained
under `runs/release-0.3.1/` (not tracked). No new NPU result is claimed.

## Namespace consolidation: 0.3.2

Implementation moved into `src/betterscale/`; the wrapper namespace is gone. Runtime
ASTs match0.3.1 after only package/log/profiler label and version normalization;
donor pins are byte-identical. The same55 CPU tests cover the new imports. Use
clean wheel and0.3.1-to0.3.2 upgrade checks, not another eight-card experiment for
this rename. Historical capsules/prototypes retain their original source identity
and require their frozen versions; do not mass-rewrite old evidence or real paths.

Published0.3.2 from `963f1c2` on2026-09-14. Both official PyPI artifacts directly
match audited builds. A0.3.1-to0.3.2 install confirms the old namespace is removed
and native Worker/pins load; no NPU rerun. Local receipt: `runs/release-0.3.2/`.

## Physical capacity: 0.4.0

Published/tagged from package source `80a75b5`; both official artifacts match the audited builds.
`docs/evidence/release-0.4.0.json` distinguishes real quality190, actual public-entry dummy192
and dummy capacity189. No extra real-weight load was needed for logging/pin/documentation-only
changes. The public Worker uses one target/draft pool, physical free-memory accounting and a
1GiB safety reserve; separate DP auxiliary metadata/producer lifetimes are unchanged.

Do not use sitecustomize to install a dummy loader: run191 polluted native CPU-info subprocess
stdout and failed before KV qualification. The corrected Worker-import-time fixture passed192;
neither dummy admission nor fixture is shipped. Native INFO records need the `vllm.` logger
namespace, not a new root namespace; absence of custom logs is not absence of native startup.

Capacity ceilings and bytes are documented in `docs/CAPACITY-0.4.zh-CN.md`. Keep96.84% labeled
TP dummy, keep DP timeout and the unrepaired small-budget preemption boundary explicit, and
never relabel historical throughput numbers as fresh0.4 benchmark results.

## Native APC admission:0.4.1

Package source/tag `a4c403f` removes only APC-off admission; no numerical execution
hook is added. Four additional native cache files are pinned (19 total), checked
on local and actual hw3 runtimes.61 package CPU tests and16 upstream CPU tests pass.
TP194/DP195 each pass real cold32/32 and warm32/32, with positive warm hits and
identical token sequences; DP also passes16 repeated requests at two per engine.
Read `prototypes/prefix-caching/README.md` for exact protocol and scope.
The wheel's runtime bodies are unchanged apart from admission/pins/version, so do
not reload eight cards to repeat a packaging-only gate.

Publication propagation has two gates: the version JSON/artifact URLs may be
public before the PyPI simple index used by pip has the release.0.4.1's first
website CI install saw only0.4.0 despite a successful official artifact download.
Verify the simple index as well before pushing a versioned installation job;
retain that failure and retry only after propagation, not by altering dependencies.
