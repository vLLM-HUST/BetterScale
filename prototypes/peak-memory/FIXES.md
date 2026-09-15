# September15 memory fixes — staged acceptance

Do not conflate a native leaf result with a complete serving/graph-pool saving.
Baseline attribution is in `evidence.json`; raw new receipts live in
`runs/peak-memory-fixes-20260915/`.

## Dense KV backing clear

The startup hook in `patches/auto_kv/_draft.py` now delegates to `_state.py`:
clear each current KV-exclusive untyped storage once through a contiguous byte
view. No data allocation, remapping, second pool, or change of graph addresses.
The pinned native compressed-cache allocator owns these backings independently;
this is not safe as a general-purpose clearer for arbitrary graph-pool views.
All warmup work is drained before request admission as before.

CPU tests:62 passed, including overlapping dtype views, page gaps, unchanged
pointer/stride/offset, and an unrelated allocation left intact.

Single-card constructed-layout probe (`clear_probe.py`):1GiB backing,512MiB
logical strided view, also exposed through an INT32 view. Original clearing
reaches1,073,743,360 additional allocated bytes; backing clearing reaches0.
Reserved-peak increments are1,069,547,520 and0 bytes respectively. Both pass
changed-input FULL graph replay against the same addresses. Single-shot clear
times8.402ms and0.743ms are illustrative, not a benchmark distribution or a
whole-model speedup. Full-model state-clear/READY comparison remains pending.

## HC-pre native workspace floor

`hc_pre_workspace.patch` applies to Ascend9bf964c. It removes only the208MiB
minimum and retains the three kernel offset extents plus the existing16MiB
provision. It does not change tiling, arithmetic, accumulation or output dtype.
The private selected-op build under `source/csrc` is not installed into the
shared donor runtime. `hc_probe.py` compares exact old/new output bits at
1/24/255/256/257/516/4128 rows, and changing-input FULL graph behavior.
All seven shapes pass exact old/new output equality and changing-input FULL
replay. Original eager allocated increments versus candidate:24 rows208.191→
22.492MiB;516 rows212.072→55.346MiB;4128 rows240.567→98.834MiB. These include
operator outputs, not only scratch. Original used local card4, candidate used
card1 after card4 became occupied: this is a numerical/allocation comparison,
NOT a timing claim. Compact exact-byte receipts are in `leaf-results.json`.

Private native build completed in about26 minutes, largely compiling fresh
protobuf/abseil prerequisites. Retain its task-local dependency products rather
than paying for them again. Package was extracted with `--noexec --extract=...`
into this task only, not installed globally. The isolated HcPre candidate uses
ONE effective custom OPP root (selected-op); the extension's ABI-compatible
original API library stays unchanged. It does not invoke HC-post or other
custom operators. A selected-op package is a diagnostic artifact, not a
complete model OPP; a complete native delivery/integration gate remains.

## Residual aliases

`hc_residual.py` copies the native decoder-layer forward, changing only the two
`hidden_states.clone()` residual saves to aliases. HC-pre reads its input;
HC-post produces a separate output. Attention and MLP consume HC-pre's new
output, not the preserved residual. This is NOT in-place HC-post.

The leaf fixture composes two native HC-pre/post pairs with identity inner
transforms. Original native HC-pre/post passed all seven row counts: readonly
inputs, exact clone/alias outputs, and repeated changing-input FULL replay. At
4128 rows the pair capture allocated increment falls531.135MiB→402.135MiB;
reserved increment644MiB→424MiB. At516 rows reserved falls284MiB→244MiB;
at256 rows both are244MiB despite lower allocated peak. This illustrates why
allocated savings do not translate proportionally to reservation. Real
attention/MLP integration and whole-model graph-pool comparison remain required; the public Worker does not install this candidate yet. Source-level
clone removal may yield nothing if the native compiler already removes it.

## Integration admission boundary

The prepared TP8 dummy capsule combines dense backing clear and residual aliases,
using the original complete native OPP and the same24/4128 buckets and HTTP work
as the earlier control. It does not enable the selected-op HC-pre package.
Local admission polled under `/root/tp8.lock` for1800 seconds and timed out before
launching any model (`Selected-card admission wait expired`). No model, capture,
HTTP or reservation result exists for this arm. The watcher exited and released
its lease; no task-owned NPU work remains. Reuse the prepared capsule in
`runs/peak-memory-fixes-20260915/integration/` with a fresh output name when a
whole-eight-card window is available, rather than repeating leaf probes.

Publication boundary: the dense-clear Python source and focused CPU/leaf tests
are committed locally; no new PyPI release was made. The residual alias and
HC-pre native change remain explicit prototypes, not public Worker defaults.
