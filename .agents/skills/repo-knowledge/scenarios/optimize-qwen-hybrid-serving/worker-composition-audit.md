# Worker composition audit (2026-09-19)

Read before consolidating production Worker entries. Source inspected at
`e0e061f`; read-only runtime investigation, no new hardware qualification.
Fletcher wants one public Worker with locally owned overrides selected from
native configuration, not another Worker subclass per optimization combination.
The initial audit below is retained as rationale. The route was subsequently
implemented: see `src/betterscale/models/README.md` and
`docs/evidence/worker-unification.json` for the exact fresh verification boundary.

## Observed migration traps

- `qwen_worker.Worker` without speculation and `MixedWorker` can accept the same
  native configuration (APC off, FULL, TP2). The former installs native-layout
  single-prefill hooks; the latter owns K-V GDN state across eager and graph
  execution. Configuration cannot reconstruct the old class choice. A unified
  entry must explicitly choose the current implementation; no catch-and-fallback
  into native V-K after owned-state initialization. Native MTP2 is a separate
  existing route, not permission to enable MTP on owned GDN.
- `qwen_gdn/graphs.py` and `publication.py` wrap the same dispatcher descriptor,
  runner execution/padding, and attention-metadata methods. Installation order is
  meaningful: capacity selection first, bank identity/publication around it.
  `qwen_fia/wave.py` then wraps the publication `_model_forward`. Preserve actual
  call order, frame ownership and reuse fences, not just a set of patch names.
- `MixedWorker` installs GDN before donor initialization and FIA afterward.
  GDN execution first imports the donor Qwen patch so its import-time assignment
  cannot overwrite our `_forward_core`. Weight packing follows model loading.
  FIA's preload is a process-start requirement; Worker installation cannot repair
  a library loaded too late. Missing qualified native artifacts must fail closed.
- Several hooks replace process-global classes, not selected instances:
  GDN builder, dispatcher, runner and ACLGraphWrapper. Their scope is currently
  protected primarily by entry admission. One public entry must NOT eagerly
  install every model's hooks. Repeated same-route installation must be harmless;
  incompatible second installation must reject before mutation. This does not
  promise multiple independently configured models in one process.
- DSV4 `PhysicalMemoryMixin` is not generic memory policy: it knows `_native_dp`,
  captures/retires DSpark draft programs through Worker methods, clears trial
  graph singletons, and prepares TP communication even with explicit KV bytes.
  Do not inherit this mixin unconditionally on a unified Qwen-capable Worker.
- DSV4 TP `ordered_replay`, DSV4 DP `async_decode/_target`, and Qwen FIA all
  override ACLGraphWrapper.__call__, with different stream/metadata ownership.
  A shared name is not evidence their synchronization contracts are interchangeable.

## Smallest plausible route

Keep a single public `betterscale.worker.Worker` and explicit bounded composition
selected before native initialization. Selection/admission should be pure and
complete before side effects. Keep model-specific compatibility/native-library
checks lazy, so Qwen does not acquire DSV4 dependencies and vice versa.

Move actual behavior to its owning component, rather than hiding existing Worker
subclasses behind a factory. Use only lifecycle hooks genuinely required by the
native Worker: bootstrap, model-loaded, memory sizing, final warmup. State and
capture ownership belong together. The GDN core, metadata and native library form
one coherent state protocol, not independently optional toggles.

Within Qwen, give each shared runner seam one owner; that owner calls capacity,
publication and FIA helpers in explicit order. Keep cross-layer wave operations
at the wave boundary, not redundantly in every attention leaf. Ordinary numerical
leaves remain local. No discovery framework, dependency solver or feature powerset.

Pinned donor source already has `AscendGDNAttentionBackend.get_builder_cls()` and
GDN `get_attn_backend()` selection. These are candidate narrower seams for the
owned builder; actual backend selection/capture behavior still needs verification.
ModelRegistry.register_model exists but swaps a whole model architecture, too
broad merely to replace GDN execution. No donor source modification proposed.

## Verification that would distinguish a real refactor

First use CPU tests to verify configuration selection, admission before mutation,
installation order, import isolation, duplicate installation and illegal route
switches. Keep numerical kernels/native binaries unchanged for the first pass.
Existing package-import tests alone do not establish lifecycle equivalence.
If moving hook ownership or ordering, run bounded affected-path hardware gates:
Qwen mixed graph/eager state oracle plus cold/warm APC; native MTP if its lifecycle
changes; DSV4 TP and DP startup/capture if their lifecycle changes. Reuse retained
performance evidence only for unchanged runtime behavior, not as proof of a newly
composed executor. Historical prototype Workers are experiment fixtures, not new
public entries to consolidate indiscriminately.
