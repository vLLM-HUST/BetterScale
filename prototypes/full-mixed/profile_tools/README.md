# Native eight-rank decode profile

These tested helpers reuse frozen TraceLoom37323af; they do not implement a
second clock fitter or export a hand-built substitute timeline.

1. Collect outside initialization/warmup with `--profile-after`. Workers are
 daemonic, so torch-npu cannot parse in their stop callback. Use the existing
 runtime's `torch_npu.profiler.profiler.analyse(PROFILE_ROOT)` offline.
2. Preserve the eight rank roots, official `ASCEND_PROFILER_OUTPUT/*.db` and
 neighboring `PROF_*/host/sqlite` / `device_*/sqlite` DBs. Archive them as
 `profile-sqlite.tar.gz` and extract under `CAPSULE/profile`. Keep raw binary
 sources on the execution host. Do not treat an isolated monolithic DB as a
 complete cross-layer evidence package.
3. Run `python analyze.py CAPSULE`. This validates each source DB, records
 rank/device maps and the source archive identity, and materializes bounded
 native analysis into `CAPSULE/analysis/rank0.db` through `rank7.db`.
4. Run `python align.py CAPSULE --export`. It joins ONLY unique identical
 provider collective names/types/groups/counts across ranks, samples at most256
 matched endpoints, then invokes TraceLoom's affine fitter and native exporter.
 The result is `analysis/target-draft-tp8-end-aligned.json.gz`.

The fitted clocks are **candidate_only, display-only**, not a physical
simultaneity or causal claim. Markers, holdout residuals and raw timestamps remain
available. Endpoint uncertainty uses the larger observed collective duration
(with a20us floor); no nearest-timestamp pairing is used.

The export can take several minutes. Full logs stay in the capsule. Link the
compressed timeline, never paste/load its raw JSON into the conversation.
Use the corrected run026 capsule for user-facing results. Run019 remains an
older profile, not an initial-call numerical qualification.
