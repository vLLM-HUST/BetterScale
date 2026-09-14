# Release the public BetterScale package

Use when building or publishing `vllm-betterscale`, or updating its install entry.
The native entry is `betterscale.worker.Worker`, an identity alias of the qualified
`strengthen_dsv4.worker.Worker`, not a second implementation. The old and new
**distributions** own overlapping files: stop the service and uninstall an old
`strengthen-dsv4` distribution before installing the new distribution. Do not
uninstall donor packages or upgrade their dependencies.

- `docs/PYPI.md` is the public package README and owns the bounded TP8/DP8 commands.
  Version changes must agree in pyproject and the implementation initializer.
- Build in a new staging directory from tracked package files plus pyproject,
  MANIFEST, public README, root README and license/notices. Do not reuse build/lib:
  it previously retained a removed CLI. `python -m build` builds the wheel from
  the sdist too; audit both archives. Include pins and per-patch READMEs, not
  prototypes, upstream checkouts, tests, credentials or profiles.
- Use separate Python 3.12 packaging tools; unset inherited PYTHONPATH for them.
  A tools-only venv lacks torch/numpy: use the existing runtime Python for the CPU
  test suite, without constructing NPU workers. The isolated alias test needs no
  donor. Do not install torch just to repair the packaging-tools environment.
- Run strict Twine metadata checks, install the wheel with --no-deps outside the
  checkout, verify both Worker imports are the same class against a native-base
  fixture, verify pins load and no console/plugin entry points are introduced.
  These are packaging checks, not new NPU correctness or performance evidence.
- Publishing requires Fletcher's authority. Credentials stay opaque in a mode600
  `.pypirc`; validate the official PyPI endpoint before upload. Never print token
  values or include them in commands/logs. Record the source commit, upload only
  the two audited artifacts, then download from official PyPI and compare to the
  local wheel before claiming availability. PyPI versions are immutable.
- Only after publication, update the website's normal MOD quickstart and detail
  integration section. Development repository visibility and PyPI source visibility
  are separate: do not advertise private GitHub source as publicly reachable.
