# Native eight-rank decode profile

These tested helpers reuse frozen TraceLoom37323af; they do not implement a
second clock fitter or export a hand-built substitute timeline.

1. Collect outside initialization/warmup with `--profile-after`. Workers are
 daemonic, so torch-npu cannot parse in their stop callback. Use the existing
 runtime's Python with `parse.py CAPSULE --jobs 2` offline (fresh process per rank).
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

Run030 exposed a marker-identity trap: some replay-associated provider names
matched across ranks but their endpoints differed by 56–291 ms on rank2. The
all-provider holdout P95 was 113 ms even though the robust median fit looked
near identity. That is NOT evidence that rank2's physical clock moved or that
its device computation stalled by that amount.

`align.py` now refuses export above 50 us holdout P95. Inspect identity rather
than deleting high-residual points. For this trace, `--eager-markers` restricts
candidates independently of timestamp residual: require linked native tasks and
exclude every op with a captured model ID. Then match the same unique provider
name/type/group/count as before. This produced 0.86–1.51 us holdout P95 across
ranks. The discarded all-provider fit is preserved in run030's
`analysis/clock-all-provider/`; its old candidate export has a distinct name.
The eager-only fit remains display-only candidate alignment, not calibration.

For larger windows use `analyze.py CAPSULE --jobs 4 --resume` to run bounded
CPU-only native analyses concurrently and reuse completed rank databases.
Resume verifies the native embedded source path/digest and SQLite quick-check;
incomplete `.tmp` files are never accepted. This preserves the source/archive
receipts and clock gate. No NPU lease is held during offline parsing/export.

Frozen37323af also lacks a child-edge lookup index for the recursive tree view's
root anti-join. On run052's larger profiles, `EXPLAIN QUERY PLAN` showed a
correlated `SCAN e`; export timed out at300s. A bounded stack sample was inside
SQLite `load_rank`, not gzip or NPU work. `analyze.py` now adds
`strengthen_viz_edge_child(child_node_id)` **only to derived analysis DBs**.
The plan becomes `SEARCH e USING COVERING INDEX`; run052 skew rank0's43562 atom
nodes enumerate in0.57s including index creation. No profile rows, graph
hierarchy, clock markers, or original source DBs are changed. Do not merely
raise export timeouts for this stable query-plan failure.

Run053 skew's unrestricted eager endpoints exceeded the50us gate (P95 up to76us).
The rejected fit is retained in `clock-all-eager/`. Large-payload collective
completion need not be simultaneous across ranks. The semantic small-control
filter `--eager-markers --max-marker-count 128` (chosen by payload count, never
residual) gives0.60–1.15us holdout P95 here. It does not relax the clock gate.
Retain this filter in markers.json; do not reinterpret large-collective end
skew as clock error or manufacture timestamp-nearest matches.

Exports now write a `.partial.json.gz` and rename only after native success.
A killed exporter can leave a **valid gzip stream containing truncated JSON**:
`gzip -t` alone is not completion evidence. The largest run052 decode window
also exhausted300s while emitting content after the index fix, so export has a
600s bound. Preserve failures and verify the native completion receipt before
linking an artifact; never hand out the partial file.

Run065 parsing lessons: torch-npu's reused worker pool can leak rank identity
across rank roots (rank0/rank7 DBs named rank2, RANK_DEVICE_MAP wrong). Fresh
Python processes per rank restore correct native identity; `parse.py` checks
the filename and rank/device map instead of editing them. Copied profiles must
be owned by the parsing user: extract trusted run archives with
`tar --no-same-owner`, not the execution host's unrelated numeric UID.
A native parse can report errors yet exit0: the checked DB contract is required.

Archive only canonical profiler/native sqlite DBs and profiler-info metadata;
exclude `REUSED_PARSER_OUTPUT` backups. Use gzip level1 (e.g. tarfile
`compresslevel=1`), not default Python gzip9: the latter cost minutes of CPU
without improving the user-facing evidence. Short eight-forward windows may
have fewer than20 small-control markers. Run065's unrestricted *eager-only*
identity set has63 markers and passes the unchanged50us gate at1.60–6.84us.
Do not relax the gate or choose pairs based on residual to force an export.

Run066 native skew fails unrestricted eager alignment (rank7 P95 164.53us),
while small-control-only identities are too few. Preserve the rejected fit;
export without a collective clock model, explicitly named `unaligned` (native
first-event normalization per rank). It supports within-rank
inspection, NOT cross-rank lateness comparisons. Do not keep trying filters
until a visually pleasing alignment appears.

For the DSV4/eager-draft continuation studies, `continuation.py WINDOW` links
observed FULL forward order to graph compute-task occurrences and checks the
counts. It attributes the first ReduceScatter only to that target graph's model,
requires matching provider name/type/group/count for cross-rank comparisons,
and preserves turnover name mismatches without inventing timestamp-nearest
matches. Source `COMMUNICATION_TASK_INFO.opId` can link multiple replay instances;
OP membership alone is not a particular physical occurrence. Keep occurrence
bounds and use the existing clock transform (reference-target plus scaled delta;
`offset_ns` is descriptive, not an extra additive term). The >1000-task target
model selector is specific to these full-model profiles with eager draft; do
NOT apply it unchanged once draft graphs are also enabled.
