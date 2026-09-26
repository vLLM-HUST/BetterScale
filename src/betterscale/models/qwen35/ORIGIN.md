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

The pinned runner has two calls into `device_apc`. Apply the bundled
`runtime-imports.patch` to the frozen baseline runtime before use; the Qwen35
source pins require that exact owned-namespace form. This changes imports only,
not runner control flow. Do not publish top-level module aliases or retain a
PYTHONPATH dependency on the experimental capsule.
