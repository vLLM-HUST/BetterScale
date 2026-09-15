# Qualify native prefix reuse with BetterScale FULL execution

Starting point: released0.4.0 / `80a75b5`, native vLLM `752a3a50`,
Ascend `9bf964cb`. Work is isolated from the published APC-off entry.

## Reuse the native mechanism

The pinned Ascend hybrid coordinator already handles C4/C128 logical-page hashes,
common hit boundaries, and propagates the EAGLE bit to same-spec sibling managers.
That propagation keeps SWA checkpoint retention consistent with read-side draft
lookahead. Do not copy these fixes into another patch or claim them as BetterScale
inventions. Physical block32 implies a4K common boundary, not the historical16K
from block128. Do not lower this boundary without a new partial-page protocol.

The shipped cross_step and DP producer cuts admit stable pure K5 only. A new
cache-hit request must first use native preparation; cached length alone must not
admit it as an already-established decode continuation. Once it becomes stable,
it may re-enter the existing fast path. Target FULL and TP draft metadata still
need an actual hit/resume gate; source plausibility is not qualification.

## First bounded gate

`runs/prefix-cache-20260915-tp/` (main checkout's ignored run directory) freezes
public code with only experimental APC/dummy admission relaxed. Dummy wo_a layout
repair is imported at the known-safe Worker entry, never sitecustomize. Native
hashing, cache managers and scheduler are unchanged. Existing leased eight-card
supervisor owns timeout, collision rejection and reclamation on hw3.

HTTP cases: cold12345 tokens; identical repeat; append256; divergence at token5000;
divergence at token100; original repeat after both branches. Each generates64
native K5 tokens. Request usage explicitly exposes cached_tokens; metrics and
outputs remain in the capsule. Require a real positive repeated hit, no hit past
a divergent token, and all completions. Do not infer reuse from lower TTFT alone.

This is a dummy execution/cache-routing gate, not output-quality or throughput
qualification. Follow with real retained retrieval cold/warm comparisons and DP
routing/ownership before changing public flags. DP caches are engine-local; a
request landing on another engine is not a cache-manager miss bug.

## CPU evidence

Pinned `test_compressed_prefix_cache.py`:3 passed on the installed local donor
runtime, exercising logical hash sensitivity, identical hits and rejection of
partial compressed-page matches. No NPU execution or approximate hash fixture.
The independent coordinator tests and actual device resume gate are recorded
separately as they complete.
