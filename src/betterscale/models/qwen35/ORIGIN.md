# Qwen35 baseline execution origin

These leaves come from the completed September24 small-fish qualification's
`candidate` source (the `serve-candidate3` and `optimized35b-c16` gates), not the
later standalone live model/scheduler. The migration first makes local imports
relative and separates model admission from the older dense-Qwen hook installer.
Numerical kernels are retained, including BF16 Q/K and beta rounding. The shared
GDN/FIA infrastructure receives explicit Qwen35 geometry and capacity.

This source migration is not, by itself, a serving qualification or permission
to change model arithmetic. State binding and graph lifecycle are separate
substitution seams; keep native model forward, MTP, sampling and async scheduling.

The full four-file adaptation from the pinned Ascend donor is bundled as
`runtime.patch`, with exact input/output identities in`runtime.json`. The inert
`python -m betterscale.models.qwen35.runtime` command stages it in a new isolated
directory. It does not modify installed donor files or import accelerator modules.
The Qwen35 source pins still check every qualified dependency. The older
`runtime-imports.patch` records only the namespace migration and is not the complete
runtime preparation recipe. See README.md for the current serving entry.
