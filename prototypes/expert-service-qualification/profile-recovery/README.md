# Evacuated A2E2 capture helpers and compact evidence

Copied from `runs/a2e2-revival/20260930-stage11-profile-recovery/capsule/`
before machine cleanup. These are the historical executed helpers, not a portable
one-command launcher: inspect and relocate their explicit worktree/runtime/model/
build/SWE-client/output paths before using them elsewhere. Shell launchers also
refer to the old external admission wrapper; use the destination host's current
lease/admission protocol, never bypass it. No automatic NPU launch is authorized.

`stage9-protocol.json` is the original manifest (its pre-run status is historical).
`stage9-summary.json` preserves selected completed client fields.
`stage11-costs.json` preserves native-DB-derived rank-local observations.
Original raw databases, client traffic and compressed timelines are NOT in Git.
Absolute artifact paths identify the old machine, not remotely backed-up files.

Early `PROFILING_MODE=dynamic`, validated same-namespace PIDs and acknowledged
start/stop fixed the attach failure. `analyze_profile.py` uses the full captured
session and counts interval unions; collect is not network-only time. Persistent
expert internal execution is absent from the attention-rank captures. No device
performance measurement of the new finite-wave protocol exists.
